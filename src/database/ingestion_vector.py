"""
ingestion_vector.py  —  Hybrid Filter + Vector Search MLS Store
================================================================

Architecture
────────────
                          ┌─────────────────────────────────┐
  Buyer inquiry text ───► │  Parser Agent (LLM)             │
                          │  → BuyerProfile (structured)    │
                          └────────────┬────────────────────┘
                                       │
                          ┌────────────▼────────────────────┐
                          │  Stage 1: Hard Filters          │
                          │  • Active listings only         │
                          │  • Budget gate (≤ 120%)         │
                          │  • Bedroom gate (min - 1)       │
                          │  • Neighbourhood resolution     │
                          │  • Property type filter         │
                          └────────────┬────────────────────┘
                                       │ candidate set
                          ┌────────────▼────────────────────┐
                          │  Stage 2: Vector Reranking      │
                          │  • Build semantic query text    │
                          │  • Embed query on the fly       │
                          │  • Cosine similarity vs index   │
                          │  • Blend: vector + hard signals │
                          └────────────┬────────────────────┘
                                       │ top-K results
                          ┌────────────▼────────────────────┐
                          │  Strategist Agent (LLM)         │
                          │  → LeadBrief with rationale     │
                          └─────────────────────────────────┘

Why hybrid (not pure vector)?
  • Hard filters enforce non-negotiables (budget, bedrooms). A semantically
    perfect listing that is $200K over budget is not a match.
  • Vector search handles the semantic layer: "home office" ↔ "remote work",
    "luxury" ↔ "prestigious", "near water" ↔ "waterfront / bay view".
  • Together they avoid both false positives (pure vector) and missed synonyms
    (pure keyword).

Embedding model
  The system already uses Google Vertex AI for its LLM agents. We reuse the
  same provider and credentials with the `text-embedding-004` model, keeping
  the stack single-vendor. The index is built once at startup and cached in
  memory; at 206 listings it fits comfortably in RAM (~1 MB for 768-d floats).

Vector index
  NumPy matrix multiplication + sklearn cosine_similarity. FAISS would add
  speed at 100K+ vectors; for sub-1K inventories it is unnecessary overhead.
"""

import logging
import os
import numpy as np
import pandas as pd
from difflib import get_close_matches
from typing import Dict, List, Any, Optional, Tuple

from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai import types
from dotenv import load_dotenv

from src.agent.schemas import BuyerProfile

load_dotenv()
logger = logging.getLogger("MLSVectorStore")

# ── Constants ────────────────────────────────────────────────────────────────

MAX_RESULTS      = 8     # Max listings passed to the Strategist Agent
VECTOR_TOP_K     = 25    # Candidate pool size after hard filters, before reranking
EMBEDDING_MODEL  = "text-embedding-004"  # Vertex AI text embedding model

# Blend weights for the final composite score.
# Tune these to shift emphasis between semantic relevance and hard-signal fit.
WEIGHT_VECTOR    = 0.50   # Semantic similarity (cosine)
WEIGHT_FEATURE   = 0.20   # Must-have keyword coverage
WEIGHT_BUDGET    = 0.20   # Price vs stated budget
WEIGHT_BEDROOM   = 0.10   # Bedroom count fit

ACTIVE_STATUSES  = {"Active"}

NEIGHBORHOOD_ALIASES: Dict[str, str] = {
    "downtown":      "Downtown Miami",
    "south beach":   "South Beach",
    "sobe":          "South Beach",
    "mid beach":     "Mid-Beach",
    "midbeach":      "Mid-Beach",
    "north beach":   "North Beach",
    "grove":         "Coconut Grove",
    "coconut grove": "Coconut Grove",
    "coral gables":  "Coral Gables",
    "gables":        "Coral Gables",
    "aventura":      "Aventura",
    "bal harbour":   "Bal Harbour",
    "bal harbor":    "Bal Harbour",
    "key biscayne":  "Key Biscayne",
    "pinecrest":     "Pinecrest",
    "doral":         "Doral",
    "north miami":   "North Miami",
    "miami beach":   "Miami Beach",
    "brickell":      "Brickell",
    "edgewater":     "Edgewater",
    "wynwood":       "Wynwood",
}

PROPERTY_TYPE_ALIASES: Dict[str, str] = {
    "condo":         "Condo",
    "apartment":     "Condo",
    "apt":           "Condo",
    "house":         "Single Family",
    "home":          "Single Family",
    "single family": "Single Family",
    "sfh":           "Single Family",
    "townhouse":     "Townhouse",
    "townhome":      "Townhouse",
    "multi family":  "Multi-Family",
    "multifamily":   "Multi-Family",
    "duplex":        "Multi-Family",
    "villa":         "Villa",
}


# ── Embedding client (singleton) ─────────────────────────────────────────────

def _get_embedding_client() -> genai.Client:
    return genai.Client(
        vertexai=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION"),
    )


def embed_texts(texts: List[str], client: genai.Client) -> np.ndarray:
    """
    Embeds a list of strings using Vertex AI text-embedding-004.

    Returns an (N, D) float32 array where D is the embedding dimension (768).
    Batches of up to 250 texts are sent per API call (Vertex AI limit).

    Each vector is L2-normalised so that dot product == cosine similarity,
    allowing us to use fast matrix multiplication instead of per-pair division.
    """
    BATCH_SIZE = 250
    all_embeddings: List[np.ndarray] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        batch_vecs = np.array(
            [e.values for e in response.embeddings], dtype=np.float32
        )
        # L2-normalise each vector in-place
        norms = np.linalg.norm(batch_vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)   # guard against zero vectors
        batch_vecs /= norms
        all_embeddings.append(batch_vecs)

    return np.vstack(all_embeddings)


def embed_query(text: str, client: genai.Client) -> np.ndarray:
    """
    Embeds a single query string.

    Uses task_type=RETRIEVAL_QUERY so the model biases the vector toward
    retrieval-style representations rather than document-style, which
    improves cosine similarity alignment with the document index.

    Returns a (1, D) normalised float32 array.
    """
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[text],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    vec = np.array(response.embeddings[0].values, dtype=np.float32).reshape(1, -1)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ── Document text builder ─────────────────────────────────────────────────────

def build_listing_document(row: pd.Series) -> str:
    """
    Constructs a rich natural-language document string for each listing.

    This is what gets embedded into the vector index. Richer text produces
    better semantic representations. We include structured fields (neighbourhood,
    type, price, bedrooms) alongside free-text (description, features) so the
    embedding captures both dimensions.

    Example output:
        "Brickell Condo | 2 bedrooms | $695,000 | Active
         Modern condo in the heart of Brickell. Features city views and...
         Features: Gym; Pool; Balcony; City View; Concierge"
    """
    return (
        f"{row.get('neighborhood', '')} {row.get('property_type', '')} | "
        f"{row.get('bedrooms', '')} bedrooms | "
        f"${row.get('price', ''):,} | "
        f"{row.get('listing_status', '')} | "
        f"{row.get('description', '')} | "
        f"Features: {row.get('features', '')}"
    )


def build_query_document(inquiry_text: str, profile: BuyerProfile) -> str:
    """
    Constructs a semantic query string from the buyer's raw inquiry and
    structured profile. Combining both gives the embedding model maximum
    context: the free-text captures tone and implicit preferences, while
    the profile adds concrete structured signals.

    Example output:
        "Buyer inquiry: Hi, I'm relocating to Miami for a new job...
         Requirements: 2+ bedrooms in Brickell or Downtown Miami.
         Budget up to $700,000. Must haves: gym, balcony, city view."

    ── FIX 3: include reference_price in the semantic query ─────────────────
    PROBLEM: When a buyer mentions a specific listing by address and price
    (e.g. "1820 Bay Road, asking $1.25M"), that price tier signals the
    lifestyle and feature class they are shopping in. Without it in the
    query text, the embedding model has no price-tier signal beyond budget_max,
    and may retrieve semantically similar properties from a completely
    different price bracket.

    FIX: Append a sentence about the referenced property's price when
    reference_price is present on the profile. This steers the vector
    similarity toward listings in the same market tier without hard-filtering
    by price (which the user explicitly does not want).
    ── END FIX 3 ────────────────────────────────────────────────────────────
    """
    parts = [f"Buyer inquiry: {inquiry_text}"]

    if profile.neighborhoods:
        parts.append(f"Preferred neighborhoods: {', '.join(profile.neighborhoods)}.")
    if profile.bedrooms_min:
        parts.append(f"Needs at least {profile.bedrooms_min} bedrooms.")
    if profile.budget_max:
        parts.append(f"Budget up to ${profile.budget_max:,}.")

    # FIX 3: steer vector search toward the buyer's reference price tier.
    reference_price = getattr(profile, "reference_price", None)
    if reference_price and reference_price != profile.budget_max:
        parts.append(
            f"Buyer is actively considering a property listed at "
            f"${reference_price:,}. Prioritise alternatives in this price tier "
            f"with similar features and lifestyle profile."
        )

    if profile.must_haves:
        parts.append(f"Must haves: {', '.join(profile.must_haves)}.")

    return " ".join(parts)


# ── Main store ────────────────────────────────────────────────────────────────

class MLSVectorStore:
    """
    Drop-in replacement for MLSDataStore that adds a vector search layer.

    Startup sequence
    ────────────────
    1. Load + clean the CSV (same as before).
    2. Build a natural-language document string for every active listing.
    3. Batch-embed all documents via Vertex AI text-embedding-004.
    4. Store the (N, 768) embedding matrix in RAM alongside the DataFrame.

    At startup this makes one batched API call (≈ 1 second for 206 listings).
    All subsequent queries hit only the in-memory index — no further API calls
    are needed for the retrieval step itself (only one call per query to embed
    the query text).

    search_properties() interface
    ──────────────────────────────
    Identical to the original MLSDataStore.search_properties() — same input
    (BuyerProfile), same output schema (List[Dict]). The orchestrator requires
    no changes.
    """

    def __init__(self, csv_path: str):
        self._embedding_client = _get_embedding_client()
        self.df = self._load_and_clean(csv_path)
        self._canonical_neighborhoods = set(
            self.df["neighborhood_clean"].unique()
        )
        self._index, self._index_ids = self._build_vector_index()

    # ── Data loading ─────────────────────────────────────────────────────────

    def _load_and_clean(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip().str.lower()

        # Drop rows missing fields critical for display and hard filtering.
        before = len(df)
        df.dropna(subset=["price", "bedrooms"], inplace=True)
        dropped = before - len(df)
        if dropped:
            logger.warning(f"Dropped {dropped} row(s) with missing price/bedrooms.")

        # ── FIX 1: Drop 0-bedroom listings ───────────────────────────────────
        # PROBLEM: The CSV contains listings with bedrooms=0. These are bad data
        # entries — no real residential property has zero bedrooms. They pass
        # every bedroom hard filter (bedrooms_min is skipped when null, and
        # 0 >= any negative threshold), receive a neutral bedroom_score of 0.5,
        # and can rank highly on vector similarity alone if their features text
        # matches the query. This caused a $320K, 0-bed condo to appear as a
        # top match for a buyer inquiring about a $1.25M single-family home.
        #
        # FIX: Discard any listing with bedrooms < 1 at load time. We do this
        # here rather than in the hard-filter stage so the bad rows never enter
        # the vector index either — embedding a 0-bedroom listing pollutes the
        # semantic space with an anomalous document.
        before_zero = len(df)
        df = df[df["bedrooms"] >= 1].copy()
        zero_dropped = before_zero - len(df)
        if zero_dropped:
            logger.warning(
                f"Dropped {zero_dropped} listing(s) with 0 bedrooms "
                f"(bad data — not real residential properties)."
            )
        # ── END FIX 1 ────────────────────────────────────────────────────────

        df["price"]    = df["price"].astype(int)
        df["bedrooms"] = df["bedrooms"].astype(int)

        # Keep only purchasable listings.
        before_active = len(df)
        df = df[df["listing_status"].isin(ACTIVE_STATUSES)].copy()
        logger.info(
            f"Active inventory: {len(df)} listings "
            f"(removed {before_active - len(df)} non-active)."
        )

        # Pre-compute normalised text columns used by hard filters.
        df["neighborhood_clean"]  = df["neighborhood"].astype(str).str.strip().str.lower()
        df["property_type_clean"] = df["property_type"].astype(str).str.strip().str.lower()
        df["description_clean"]   = df["description"].astype(str).str.lower()
        df["features_clean"]      = df["features"].astype(str).str.lower()

        df.reset_index(drop=True, inplace=True)
        return df

    # ── Vector index construction ─────────────────────────────────────────────

    def _build_vector_index(self) -> Tuple[np.ndarray, List[int]]:
        """
        Embeds every active listing once at startup.

        Returns
        ───────
        index    : (N, D) float32 ndarray — L2-normalised embedding matrix.
        index_ids: list[int] — DataFrame integer positions matching each row
                   in `index`. Used to map similarity scores back to listings.

        The index is built from `build_listing_document()` strings rather than
        raw CSV text, giving the embedding model a clean, structured input that
        captures all relevant fields in a consistent format.
        """
        logger.info(f"Building vector index for {len(self.df)} listings...")
        documents = [build_listing_document(row) for _, row in self.df.iterrows()]

        # embed_texts returns an already L2-normalised (N, D) matrix.
        embeddings = embed_texts(documents, self._embedding_client)

        # index_ids maps position i in the embedding matrix → DataFrame iloc index.
        index_ids = list(range(len(self.df)))
        logger.info(
            f"Vector index ready. Shape: {embeddings.shape} "
            f"(~{embeddings.nbytes / 1024:.1f} KB in RAM)."
        )
        return embeddings, index_ids

    # ── Neighbourhood and property-type helpers ───────────────────────────────

    def _resolve_neighborhoods(self, raw_names: List[str]) -> List[str]:
        resolved: List[str] = []
        for raw in raw_names:
            token = raw.strip().lower()
            if token in NEIGHBORHOOD_ALIASES:
                resolved.append(NEIGHBORHOOD_ALIASES[token].lower())
            elif token in self._canonical_neighborhoods:
                resolved.append(token)
            else:
                close = get_close_matches(
                    token, self._canonical_neighborhoods, n=1, cutoff=0.75
                )
                if close:
                    logger.info(f"Fuzzy neighbourhood: '{raw}' → '{close[0]}'")
                    resolved.append(close[0])
                else:
                    for canonical in self._canonical_neighborhoods:
                        if token in canonical or canonical in token:
                            logger.info(f"Substring neighbourhood: '{raw}' → '{canonical}'")
                            resolved.append(canonical)
                            break
                    else:
                        logger.warning(f"Could not resolve neighbourhood '{raw}'.")
        return list(dict.fromkeys(resolved))

    def _resolve_property_type(self, must_haves: List[str]) -> Optional[str]:
        for item in must_haves:
            token = item.strip().lower()
            if token in PROPERTY_TYPE_ALIASES:
                return PROPERTY_TYPE_ALIASES[token].lower()
        return None

    # ── Hard filters ──────────────────────────────────────────────────────────

    def _apply_hard_filters(
        self, profile: BuyerProfile
    ) -> Tuple[pd.DataFrame, str]:
        """
        Applies non-negotiable filters to narrow the candidate pool before
        vector reranking.

        Tier 1 (strict):  budget ≤ 105%, bedrooms ≥ min, neighbourhood, type.
        Tier 2 (relaxed): budget ≤ 120%, bedrooms ≥ min−1, same geo/type.
        Tier 3 (broad):   neighbourhood + type only — no price/bedroom gate.

        Returns the filtered DataFrame and the tier label applied.
        """
        results      = self.df.copy()
        resolved_nb  = self._resolve_neighborhoods(profile.neighborhoods or [])
        resolved_pt  = self._resolve_property_type(profile.must_haves or [])

        # Geography filter (preserved across tiers).
        if resolved_nb:
            mask = results["neighborhood_clean"].isin(resolved_nb)
            if mask.any():
                results = results[mask]

        # Property type filter (preserved across tiers).
        if resolved_pt:
            mask = results["property_type_clean"] == resolved_pt
            if mask.any():
                results = results[mask]

        # Tier 1 — strict price + bedroom gate.
        strict = results.copy()
        if profile.budget_max:
            strict = strict[strict["price"] <= profile.budget_max * 1.05]
        if profile.bedrooms_min:
            strict = strict[strict["bedrooms"] >= profile.bedrooms_min]

        if not strict.empty:
            return strict, "Tier 1: Strict Match"

        # Tier 2 — relaxed gate.
        relaxed = results.copy()
        if profile.budget_max:
            relaxed = relaxed[relaxed["price"] <= profile.budget_max * 1.20]
        if profile.bedrooms_min and profile.bedrooms_min > 1:
            relaxed = relaxed[relaxed["bedrooms"] >= profile.bedrooms_min - 1]

        if not relaxed.empty:
            return relaxed, "Tier 2: Relaxed Fallback"

        # Tier 3 — geography + type only.
        if not results.empty:
            return results, "Tier 3: Broad Fallback"

        # Nothing survived — return full active inventory as last resort.
        return self.df.copy(), "Tier 3: Broad Fallback"

    # ── Scoring helpers ───────────────────────────────────────────────────────

    def _feature_score(self, row: pd.Series, profile: BuyerProfile) -> float:
        """Fraction of non-type must_haves found in listing text."""
        if not profile.must_haves:
            return 1.0
        feature_items = [
            item for item in profile.must_haves
            if item.strip().lower() not in PROPERTY_TYPE_ALIASES
        ]
        if not feature_items:
            return 1.0
        matched = sum(
            1 for item in feature_items
            if item.lower().strip() in row["description_clean"]
            or item.lower().strip() in row["features_clean"]
        )
        return matched / len(feature_items)

    def _budget_score(self, price: int, profile: BuyerProfile) -> float:
        """Linear decay from 1.0 (at/under budget) to 0.0 (at 120% of budget)."""
        if not profile.budget_max or profile.budget_max <= 0:
            return 0.5
        over = max(0.0, (price - profile.budget_max) / profile.budget_max)
        return max(0.0, 1.0 - over / 0.20)

    def _bedroom_score(self, bedrooms: int, profile: BuyerProfile) -> float:
        """
        Exact = 1.0; each surplus bedroom −0.15; deficit = 0.0.

        ── FIX 2: hard zero for 0-bedroom listings, even when bedrooms_min is null ──
        PROBLEM: When bedrooms_min is null (buyer didn't state a preference),
        the original code returned 0.5 unconditionally — including for listings
        with 0 bedrooms that slipped past Fix 1 or future bad rows. A 0-bedroom
        property getting a neutral 0.5 bedroom_score means it can still rank
        alongside real 2-3 bedroom properties when its vector similarity is high.

        FIX: Return 0.0 for any listing with 0 bedrooms regardless of whether
        the buyer stated a minimum. A property with no bedrooms is never a valid
        residential recommendation and should score last in any ranking.
        This is a belt-and-suspenders guard: Fix 1 removes them at load time,
        but this ensures a score of 0.0 if any ever reach the scoring stage.
        ── END FIX 2 ────────────────────────────────────────────────────────────
        """
        # Belt-and-suspenders: 0-bedroom listings should never be recommended.
        if bedrooms < 1:
            return 0.0

        if not profile.bedrooms_min:
            return 0.5
        delta = bedrooms - profile.bedrooms_min
        if delta < 0:
            return 0.0
        return max(0.0, 1.0 - delta * 0.15)

    # ── Core search ───────────────────────────────────────────────────────────

    def search_properties(
        self, profile: BuyerProfile, inquiry_text: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search: hard filters → vector reranking → blended scoring.

        Parameters
        ──────────
        profile      : Structured buyer profile from the Parser Agent.
        inquiry_text : Original free-text inquiry. Used to build a richer
                       semantic query. Pass an empty string if unavailable
                       (the method falls back to profile fields only).

        Returns
        ───────
        List of at most MAX_RESULTS property dicts, sorted by composite score.

        Step-by-step
        ────────────
        1. Apply hard filters → candidate DataFrame + tier label.
        2. Build a natural-language query from inquiry_text + profile.
        3. Embed the query with task_type=RETRIEVAL_QUERY.
        4. Compute cosine similarity between the query vector and every
           candidate's pre-built embedding (using the DataFrame index positions
           to look up the right rows in the full embedding matrix).
        5. Blend vector similarity with feature, budget, and bedroom scores.
        6. Sort by composite score, cap at MAX_RESULTS, return.
        """

        # ── Step 1: Hard filters ──────────────────────────────────────────────
        candidates, tier_label = self._apply_hard_filters(profile)

        if candidates.empty:
            return []

        # ── Step 2: Build semantic query ──────────────────────────────────────
        query_text = build_query_document(inquiry_text, profile)

        # ── Step 3: Embed the query ───────────────────────────────────────────
        # One lightweight API call per search — the index itself is in RAM.
        query_vec = embed_query(query_text, self._embedding_client)

        # ── Step 4: Vector similarity against candidates only ─────────────────
        #
        # candidates.index holds the original DataFrame integer positions,
        # which are the same as self._index_ids positions (we reset_index after
        # loading). We slice the pre-built embedding matrix to only the rows
        # that survived the hard filters — avoiding O(N) similarity against
        # the full corpus when the candidate set is small.
        #
        candidate_positions = candidates.index.tolist()
        candidate_embeddings = self._index[candidate_positions]      # (M, D)
        cosine_scores = cosine_similarity(query_vec, candidate_embeddings)[0]  # (M,)

        # ── Step 5: Blended scoring ───────────────────────────────────────────
        matches: List[Dict[str, Any]] = []
        for local_idx, (df_pos, row) in enumerate(candidates.iterrows()):
            vector_sim   = float(cosine_scores[local_idx])
            feature_sig  = self._feature_score(row, profile)
            budget_sig   = self._budget_score(int(row["price"]), profile)
            bedroom_sig  = self._bedroom_score(int(row["bedrooms"]), profile)

            composite = (
                WEIGHT_VECTOR  * vector_sim  +
                WEIGHT_FEATURE * feature_sig +
                WEIGHT_BUDGET  * budget_sig  +
                WEIGHT_BEDROOM * bedroom_sig
            )

            matches.append({
                "listing_id":            row.get("listing_id", "N/A"),
                "address":               row["address"],
                "price":                 int(row["price"]),
                "neighborhood":          row["neighborhood"],
                "bedrooms":              int(row["bedrooms"]),
                "features":              row["features"],
                "property_type":         row.get("property_type", ""),
                "fallback_tier_applied": tier_label,
                "match_score":           round(composite, 3),
                # Signal breakdown surfaced for transparency / debugging.
                "score_breakdown": {
                    "vector_similarity": round(vector_sim,  3),
                    "feature_coverage":  round(feature_sig, 3),
                    "budget_efficiency": round(budget_sig,  3),
                    "bedroom_fit":       round(bedroom_sig, 3),
                },
                "match_rationale": "",   # Populated downstream by Strategist Agent.
            })

        # ── Step 6: Sort and cap ──────────────────────────────────────────────
        matches.sort(key=lambda x: -x["match_score"])
        return matches[:MAX_RESULTS]