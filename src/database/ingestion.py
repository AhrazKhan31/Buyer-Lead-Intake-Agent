import logging
import pandas as pd
from typing import Dict, List, Any, Optional
from difflib import get_close_matches
from src.agent.schemas import BuyerProfile

logger = logging.getLogger("MLSDataStore")

# Maximum number of candidate properties passed to the LLM context.
# Keeps token usage bounded and forces ranking to do real work.
MAX_RESULTS = 8

# Neighborhood aliases: common buyer shorthand → canonical CSV value.
# Extend this dict as new aliases appear in buyer messages.
NEIGHBORHOOD_ALIASES: Dict[str, str] = {
    "downtown":     "Downtown Miami",
    "south beach":  "South Beach",
    "sobe":         "South Beach",
    "mid beach":    "Mid-Beach",
    "midbeach":     "Mid-Beach",
    "north beach":  "North Beach",
    "brickell":     "Brickell",
    "edgewater":    "Edgewater",
    "wynwood":      "Wynwood",
    "grove":        "Coconut Grove",
    "coconut grove":"Coconut Grove",
    "coral gables": "Coral Gables",
    "gables":       "Coral Gables",
    "aventura":     "Aventura",
    "bal harbour":  "Bal Harbour",
    "bal harbor":   "Bal Harbour",
    "key biscayne": "Key Biscayne",
    "pinecrest":    "Pinecrest",
    "doral":        "Doral",
    "north miami":  "North Miami",
    "miami beach":  "Miami Beach",
}

# Property type aliases: what buyers say → what the CSV property_type column holds.
PROPERTY_TYPE_ALIASES: Dict[str, str] = {
    "condo":          "Condo",
    "apartment":      "Condo",
    "apt":            "Condo",
    "house":          "Single Family",
    "home":           "Single Family",
    "single family":  "Single Family",
    "sfh":            "Single Family",
    "townhouse":      "Townhouse",
    "townhome":       "Townhouse",
    "multi family":   "Multi-Family",
    "multifamily":    "Multi-Family",
    "duplex":         "Multi-Family",
    "villa":          "Villa",
}

# Listing statuses that represent genuinely purchasable inventory.
ACTIVE_STATUSES = {"Active"}


class MLSDataStore:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

        # Normalise column headers once.
        self.df.columns = self.df.columns.str.strip().str.lower()

        # Drop rows missing fields that are critical for display and filtering.
        rows_before = len(self.df)
        self.df.dropna(subset=["price", "bedrooms"], inplace=True)
        rows_dropped = rows_before - len(self.df)
        if rows_dropped:
            logger.warning(f"Dropped {rows_dropped} listing(s) with missing price/bedrooms.")

        # Cast to correct integer types immediately after NaN removal.
        self.df["price"]    = self.df["price"].astype(int)
        self.df["bedrooms"] = self.df["bedrooms"].astype(int)

        # IMPROVEMENT 1 — Active-only inventory.
        # Pending / Under Contract listings cannot be purchased; excluding them
        # prevents wasting the LLM context on unavailable properties.
        available_before = len(self.df)
        self.df = self.df[self.df["listing_status"].isin(ACTIVE_STATUSES)].copy()
        logger.info(
            f"Inventory: {len(self.df)} active listings "
            f"(filtered out {available_before - len(self.df)} non-active rows)."
        )

        # Pre-compute cleaned text columns once at startup.
        self.df["neighborhood_clean"]  = self.df["neighborhood"].astype(str).str.strip().str.lower()
        self.df["property_type_clean"] = self.df["property_type"].astype(str).str.strip().str.lower()
        self.df["description_clean"]   = self.df["description"].astype(str).str.lower()
        self.df["features_clean"]      = self.df["features"].astype(str).str.lower()

        # Build a lookup set of canonical neighbourhood names (lower-cased) for
        # fuzzy matching in _resolve_neighborhoods().
        self._canonical_neighborhoods = set(self.df["neighborhood_clean"].unique())

    # ------------------------------------------------------------------
    # IMPROVEMENT 2 — Neighbourhood resolution with aliases + fuzzy match
    # ------------------------------------------------------------------
    def _resolve_neighborhoods(self, raw_names: List[str]) -> List[str]:
        """
        Maps buyer-supplied neighbourhood strings to canonical CSV values.

        Resolution order per token:
          1. Direct alias lookup (handles shorthand like 'grove', 'sobe').
          2. Exact case-insensitive match against known canonical names.
          3. Fuzzy match via difflib (catches typos like 'Aventurra').
          4. Substring containment (catches 'downtown' inside 'Downtown Miami').

        Returns a deduplicated list of lower-cased canonical neighbourhood strings
        ready for DataFrame filtering.
        """
        resolved: List[str] = []
        for raw in raw_names:
            token = raw.strip().lower()

            # Step 1 — alias table
            if token in NEIGHBORHOOD_ALIASES:
                resolved.append(NEIGHBORHOOD_ALIASES[token].lower())
                continue

            # Step 2 — exact canonical match
            if token in self._canonical_neighborhoods:
                resolved.append(token)
                continue

            # Step 3 — fuzzy match (cutoff=0.75 avoids false positives)
            close = get_close_matches(token, self._canonical_neighborhoods, n=1, cutoff=0.75)
            if close:
                logger.info(f"Fuzzy-matched neighbourhood '{raw}' → '{close[0]}'")
                resolved.append(close[0])
                continue

            # Step 4 — substring containment
            for canonical in self._canonical_neighborhoods:
                if token in canonical or canonical in token:
                    logger.info(f"Substring-matched neighbourhood '{raw}' → '{canonical}'")
                    resolved.append(canonical)
                    break
            else:
                logger.warning(f"Could not resolve neighbourhood '{raw}' — skipping.")

        return list(dict.fromkeys(resolved))  # deduplicate, preserve order

    # ------------------------------------------------------------------
    # IMPROVEMENT 3 — Property type resolution with aliases
    # ------------------------------------------------------------------
    def _resolve_property_type(self, must_haves: List[str]) -> Optional[str]:
        """
        Scans the must_haves list for a property-type keyword and returns the
        canonical CSV value, or None if no type preference was expressed.
        """
        for item in must_haves:
            token = item.strip().lower()
            if token in PROPERTY_TYPE_ALIASES:
                return PROPERTY_TYPE_ALIASES[token].lower()
        return None

    # ------------------------------------------------------------------
    # Core search
    # ------------------------------------------------------------------
    def search_properties(self, profile: BuyerProfile) -> List[Dict[str, Any]]:
        """
        Multi-tier hybrid matching pipeline.

        Tier 1 — Strict:  budget ≤ 105 %, exact bedroom count, resolved neighbourhood,
                          resolved property type.
        Tier 2 — Relaxed: budget ≤ 120 %, bedrooms − 1, same neighbourhood/type.
        Tier 3 — Broad:   neighbourhood + type only (no price/bedroom gate).

        Each tier is tried in sequence; the first that returns ≥ 1 result wins.
        Results are scored on a multi-signal rubric and capped at MAX_RESULTS.
        """
        results   = self.df.copy()
        tier_name = "Tier 1: Strict Match"

        # ── Neighbourhood filter ────────────────────────────────────────
        if profile.neighborhoods:
            resolved = self._resolve_neighborhoods(profile.neighborhoods)
            if resolved:
                mask = results["neighborhood_clean"].isin(resolved)
                if mask.any():
                    results = results[mask]

        # ── IMPROVEMENT 3: Property type filter ────────────────────────
        resolved_type = self._resolve_property_type(profile.must_haves or [])
        if resolved_type:
            type_mask = results["property_type_clean"] == resolved_type
            if type_mask.any():
                results = results[type_mask]

        # ── Tier 1: Strict budget + bedroom gate ───────────────────────
        strict = results.copy()
        if profile.budget_max:
            strict = strict[strict["price"] <= profile.budget_max * 1.05]
        if profile.bedrooms_min:
            strict = strict[strict["bedrooms"] >= profile.bedrooms_min]

        # ── Tier 2: Relaxed fallback ────────────────────────────────────
        if strict.empty:
            tier_name = "Tier 2: Relaxed Fallback"
            relaxed = results.copy()
            if profile.budget_max:
                relaxed = relaxed[relaxed["price"] <= profile.budget_max * 1.20]
            if profile.bedrooms_min and profile.bedrooms_min > 1:
                relaxed = relaxed[relaxed["bedrooms"] >= profile.bedrooms_min - 1]
            results = relaxed

        # ── Tier 3: Broad fallback (neighbourhood + type only) ──────────
        elif not strict.empty:
            results = strict

        if results.empty:
            tier_name = "Tier 3: Broad Fallback"
            logger.info("No results after Tier 2; falling back to neighbourhood+type-only results.")
            results = self.df.copy()
            if profile.neighborhoods:
                resolved = self._resolve_neighborhoods(profile.neighborhoods)
                if resolved:
                    mask = results["neighborhood_clean"].isin(resolved)
                    if mask.any():
                        results = results[mask]
            if resolved_type:
                type_mask = results["property_type_clean"] == resolved_type
                if type_mask.any():
                    results = results[type_mask]

        if results.empty:
            return []

        # ── IMPROVEMENT 4: Multi-signal scoring ────────────────────────
        matches = []
        for _, row in results.iterrows():
            score = self._compute_score(row, profile)
            matches.append({
                "listing_id":          row.get("listing_id", "N/A"),
                "address":             row["address"],
                "price":               int(row["price"])    if pd.notna(row["price"])    else 0,
                "neighborhood":        row["neighborhood"],
                "bedrooms":            int(row["bedrooms"]) if pd.notna(row["bedrooms"]) else 0,
                "features":            row["features"],
                "property_type":       row.get("property_type", ""),
                "fallback_tier_applied": tier_name,
                "match_score":         score,
                "match_rationale":     "",  # Populated downstream by the Strategist Agent.
            })

        # ── IMPROVEMENT 5: Cap results ──────────────────────────────────
        matches.sort(key=lambda x: (-x["match_score"], x["price"]))
        return matches[:MAX_RESULTS]

    # ------------------------------------------------------------------
    # IMPROVEMENT 4 — Multi-signal scoring rubric
    # ------------------------------------------------------------------
    def _compute_score(self, row: pd.Series, profile: BuyerProfile) -> float:
        """
        Produces a composite relevance score in [0.0, 1.0] across four signals:

          Signal              Weight   Notes
          ──────────────────  ──────   ──────────────────────────────────────────
          Feature coverage    40 %     Fraction of must_haves found in listing.
          Budget efficiency   30 %     Decays linearly as price approaches ceiling.
          Bedroom fit         20 %     Exact match = full marks; surplus penalised.
          Property type fit   10 %     Binary: canonical type match or not.

        Using named signals with explicit weights makes the rubric transparent,
        easy to tune, and straightforward to extend (e.g. adding a proximity
        signal later).
        """

        # ── Signal A: Feature coverage (40 %) ──────────────────────────
        if profile.must_haves:
            # Exclude property-type tokens — they are handled by type filtering,
            # not by keyword scanning the free-text fields.
            feature_must_haves = [
                item for item in profile.must_haves
                if item.strip().lower() not in PROPERTY_TYPE_ALIASES
            ]
            if feature_must_haves:
                matched = sum(
                    1 for item in feature_must_haves
                    if item.lower().strip() in row["description_clean"]
                    or item.lower().strip() in row["features_clean"]
                )
                feature_score = matched / len(feature_must_haves)
            else:
                feature_score = 1.0
        else:
            feature_score = 1.0

        # ── Signal B: Budget efficiency (30 %) ──────────────────────────
        # Score = 1.0 at or under budget, decays linearly to 0.0 at 120 % of budget.
        if profile.budget_max and profile.budget_max > 0:
            price        = row["price"]
            budget       = profile.budget_max
            over_fraction = max(0.0, (price - budget) / budget)  # 0 if under budget
            # Linear decay: 0 % over → 1.0, 20 % over → 0.0
            budget_score  = max(0.0, 1.0 - (over_fraction / 0.20))
        else:
            budget_score = 0.5  # Neutral when no budget was stated.

        # ── Signal C: Bedroom fit (20 %) ───────────────────────────────
        # Exact match = 1.0; each surplus bedroom subtracts 0.15 (surplus is
        # wasteful but not disqualifying); under-minimum gets 0.0.
        if profile.bedrooms_min:
            delta = row["bedrooms"] - profile.bedrooms_min
            if delta < 0:
                bedroom_score = 0.0
            else:
                bedroom_score = max(0.0, 1.0 - delta * 0.15)
        else:
            bedroom_score = 0.5  # Neutral when no preference stated.

        # ── Signal D: Property type fit (10 %) ─────────────────────────
        resolved_type = self._resolve_property_type(profile.must_haves or [])
        if resolved_type:
            type_score = 1.0 if row["property_type_clean"] == resolved_type else 0.0
        else:
            type_score = 1.0  # No preference expressed → full marks.

        composite = (
            feature_score  * 0.40 +
            budget_score   * 0.30 +
            bedroom_score  * 0.20 +
            type_score     * 0.10
        )
        return round(composite, 3)