import os
import re
import json
import time
import logging
from google import genai
from google.genai import types
from dotenv import load_dotenv
from src.agent.schemas import BuyerProfile, LeadBrief, PerformanceMetrics
from src.database.ingestion import MLSDataStore

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AgentOrchestrator")

# IMPROVEMENT 1: Lazy client initialisation via a function.
# A module-level client is created once at import time. If the Vertex AI
# token expires mid-session the module would need a full reload to recover.
# A factory function creates a fresh client per pipeline run at negligible cost.
def _make_client() -> genai.Client:
    return genai.Client(
        vertexai=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION"),
    )

# IMPROVEMENT 2: Prompt injection sanitisation.
# Aaron Cooper's message contained "ignore all previous instructions" which
# caused 9.72s parser latency (vs ~1.9s avg) as the LLM processed the
# injected instruction. We strip classic injection patterns before the text
# enters any prompt. This does NOT modify what's stored or displayed — only
# what's sent to the LLM.
_INJECTION_PATTERNS = re.compile(
    r"(ignore\s+(all\s+)?(previous|prior|above)\s+instructions?|"
    r"forget\s+(everything|all|prior)|"
    r"you\s+are\s+now\s+a|"
    r"disregard\s+(your|all|previous)|"
    r"respond\s+(only|just)\s+(by|with|in)\s+json|"
    r"list\s+all\s+(owner|phone|email|contact|private))",
    re.IGNORECASE,
)

def _sanitise_input(text: str) -> str:
    """
    Strips known prompt injection patterns from buyer inquiry text.

    The legitimate content of the message is preserved — only the injection
    attempt itself is removed. A short note is appended so the LLM is aware
    the text was cleaned, which prevents it from hallucinating the missing
    fragment.
    """
    cleaned = _INJECTION_PATTERNS.sub("[REDACTED]", text)
    if cleaned != text:
        logger.warning("Prompt injection attempt detected and sanitised.")
        cleaned += "\n[Note: part of the above message was automatically redacted for security.]"
    return cleaned

# IMPROVEMENT 3: Retry wrapper for transient LLM errors.
# The evaluation pipeline already has retry logic, but the orchestrator itself
# had none — a single 429 or network blip failed the whole lead silently.
def _call_with_retry(fn, max_retries: int = 3, base_wait: float = 5.0):
    """
    Calls fn() up to max_retries times, backing off on 429 / quota errors.
    Any other exception is re-raised immediately (not retried).
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            if "429" in msg or "quota" in msg.lower() or "rate" in msg.lower():
                wait = base_wait * (attempt + 1)
                logger.warning(f"Rate limit hit (attempt {attempt+1}/{max_retries}). Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise  # Non-rate-limit errors fail fast.
    raise RuntimeError(f"Max retries ({max_retries}) exceeded on LLM call.")


def run_parser_agent(inquiry_text: str) -> tuple[BuyerProfile, float, int, int]:
    """
    Extracts structured buyer parameters from free-text inquiry.

    IMPROVEMENT 4: The prompt now includes an explicit output contract — a
    description of every field the LLM must populate. This reduces the rate
    of field hallucination and missing keys compared to relying solely on the
    JSON schema constraint.

    IMPROVEMENT 2 applied: inquiry_text is sanitised before injection into
    the prompt f-string.

    HALLUCINATION FIX 1A applied: buyer_name added to the field contract with
    an explicit null instruction. Previously the field was absent from the
    prompt, so the LLM inferred a name from context and invented one when
    the buyer didn't state it. Now it is told to return null in that case.
    NOTE: BuyerProfile in schemas.py must also have buyer_name: Optional[str] = None
    for this to be included in the structured output. See Fix 1A note below.
    """
    safe_text = _sanitise_input(inquiry_text)

    prompt = f"""
    You are a real estate intake specialist. Analyse the buyer inquiry below and extract
    a structured profile. Populate every field described:

    - buyer_name (str | null): the buyer's name ONLY if they explicitly state it in the
      message. If the buyer does not mention their name, return null. NEVER infer, guess,
      or fabricate a name from context, email style, or any other signal.
    - budget_max (int | null): the BUYER'S maximum spend in USD.
      Important distinctions:
      • If the buyer states what they are WILLING TO PAY or OFFER → that is budget_max.
      • If the buyer mentions a listing's ASKING PRICE as context → use that as budget_max only if the buyer hasn't stated their own offer/budget separately.
      • If both are present, prefer the ASKING PRICE as budget_max since it reflects the price range the buyer is actively considering.
      • Return null if no price of any kind is mentioned.
    - bedrooms_min (int | null): minimum bedrooms needed; null if not mentioned
    - neighborhoods (list[str]): desired Miami neighbourhoods; empty list if unspecified
    - urgency_score (int 1-10): 1=no rush, 10=must close this week; infer from tone and timeline
    - must_haves (list[str]): strict non-negotiables, each as a short keyword (e.g. "pool", "pet-friendly")
    - extraction_reasoning (str): your chain-of-thought explaining each extracted value

    Buyer inquiry:
    \"\"\"{safe_text}\"\"\"

    Return only valid JSON matching the schema. Do not add extra fields.
    """
    # ── HALLUCINATION FIX 1A: buyer_name added to prompt field contract ───────
    # The original prompt had no buyer_name entry. Without an explicit rule, the
    # LLM treated name extraction as "fill in what seems reasonable", producing
    # fabricated names for 10 of 12 leads in evaluation_report.json.
    # The new entry above instructs it to return null when no name is stated,
    # which surfaces to the orchestrator and lets Fix 2 (below) apply the
    # authoritative metadata name instead.
    # ── END HALLUCINATION FIX 1A ─────────────────────────────────────────────

    client = _make_client()
    start = time.time()

    result = _call_with_retry(lambda: client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=BuyerProfile,
            temperature=0.1,
        ),
    ))

    latency = time.time() - start
    in_tok  = result.usage_metadata.prompt_token_count     if result.usage_metadata else 0
    out_tok = result.usage_metadata.candidates_token_count if result.usage_metadata else 0

    return BuyerProfile.model_validate_json(result.text), latency, in_tok, out_tok


def run_strategist_agent(
    inquiry_text: str,
    profile: BuyerProfile,
    matched_listings: list,
    metadata: dict,
) -> tuple[LeadBrief, float, int, int]:
    """
    Generates the strategic lead brief from structured profile and property matches.

    IMPROVEMENT 5: matched_listings is capped at 8 entries before serialisation.
    Without this cap, a vague lead (no filters) could send all 206 listings as
    JSON into the strategist context, wasting ~3K tokens and degrading output.

    IMPROVEMENT 4 applied: prompt specifies exactly what the strategist must do
    for each output field to reduce generic or hallucinated rationales.

    IMPROVEMENT 2 applied: inquiry_text sanitised before prompt injection.

    HALLUCINATION FIX 1B applied: the buyer name fallback changed from 'Unknown'
    to an explicit instruction not to invent a name. 'Unknown' reads to the LLM
    as a placeholder it should complete; the new wording closes that gap.
    """
    safe_text       = _sanitise_input(inquiry_text)
    # Cap listings and strip internal-only fields (embeddings, clean text cols)
    # before sending to the LLM — those fields add tokens without adding value.
    listings_subset = matched_listings[:8]
    safe_listings   = [
        {k: v for k, v in l.items() if k not in ("score_breakdown",)}
        for l in listings_subset
    ]

    # ── HALLUCINATION FIX 1B: safe buyer name label for the strategist prompt ─
    # PROBLEM: metadata.get('buyer_name', 'Unknown') passed the string 'Unknown'
    # to the LLM when no name was available. The LLM treated 'Unknown' as a
    # placeholder and substituted a plausible-sounding name in the output brief.
    #
    # FIX: Use an explicit "Not provided" label with a do-not-invent instruction.
    # This closes the implicit invitation to complete the placeholder.
    buyer_name_label = (
        metadata.get("buyer_name")
        or "Not provided — do not invent or infer a name, leave buyer_name as null"
    )
    # ── END HALLUCINATION FIX 1B ─────────────────────────────────────────────

    prompt = f"""
    You are an elite real estate strategist. Produce a Lead Brief for the agent.

    Buyer context:
    - Name: {buyer_name_label}
    - Channel: {metadata.get('channel', 'Unknown')}
    - Original message: \"\"\"{safe_text}\"\"\"
    - Extracted profile: {profile.model_dump_json()}

    Property candidates (ranked by relevance score):
    {json.dumps(safe_listings, indent=2)}

    Instructions:
    1. buyer_summary: 3 sentences — who the buyer is and their key requirement.
    2. recommended_properties: include ONLY properties from the candidates list above.
       For each, preserve listing_id, price, neighborhood, fallback_tier_applied exactly
       as given. Write a specific match_rationale that references the buyer's stated needs.
    3. strategic_advice: concrete next step for the agent, tailored to the {metadata.get('channel', 'website_form')} channel.
    4. human_in_the_loop_flags: flag unrealistic budgets, missing info, or ethical concerns.
       Leave empty list if none.
    5. follow_up_message: Write a ready-to-send message the realtor can copy-paste directly to the buyer. It must:
       - Open with the buyer's name if known, or a neutral greeting if not.
       - Reference 1 specific detail from their inquiry (e.g. the property address, their budget, or their timeline).
       - Offer concrete next steps (e.g. schedule a call, view alternatives).
       - Be 3-4 sentences, warm but professional. Plain text only — no markdown, no bullet points, no backticks.
    6. Use plain text only in all string fields. Do not use markdown, backticks, bullet points, or any formatting characters.

    Return only valid JSON matching the schema.
    """

    client = _make_client()
    start  = time.time()

    result = _call_with_retry(lambda: client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=LeadBrief,
            temperature=0.2,
        ),
    ))

    latency = time.time() - start
    in_tok  = result.usage_metadata.prompt_token_count     if result.usage_metadata else 0
    out_tok = result.usage_metadata.candidates_token_count if result.usage_metadata else 0

    return LeadBrief.model_validate_json(result.text), latency, in_tok, out_tok


def process_lead(
    inquiry_text: str,
    mls_db: MLSDataStore,
    metadata: dict = None,
) -> LeadBrief:
    """
    Main pipeline orchestrator.

    IMPROVEMENT 6: passes inquiry_text into search_properties() so the vector
    store can build a richer semantic query (raw message + structured profile).
    Previously only the structured profile was used, losing implicit buyer
    signals present only in the free-text (tone, lifestyle context, etc.).

    HALLUCINATION FIX 2 applied: after the strategist returns the brief,
    buyer_name is overwritten with the authoritative value from metadata.
    This is the final safety net — even if Fixes 1A and 1B let a hallucinated
    name through, it is corrected here before the brief leaves the pipeline.
    """
    if metadata is None:
        metadata = {}

    buyer_name = metadata.get("buyer_name", "Unknown")
    logger.info(f"Pipeline start: {buyer_name}")

    # Step 1 — Parser Agent
    profile, parse_lat, p_in, p_out = run_parser_agent(inquiry_text)

    # Step 2 — Property search (vector-aware if MLSVectorStore, keyword if MLSDataStore)
    # Pass inquiry_text so vector store can embed the full query, not just profile fields.
    try:
        matches = mls_db.search_properties(profile, inquiry_text=inquiry_text)
    except TypeError:
        # Fallback: old MLSDataStore signature doesn't accept inquiry_text kwarg.
        matches = mls_db.search_properties(profile)

    # Step 3 — Strategist Agent
    brief, strat_lat, s_in, s_out = run_strategist_agent(
        inquiry_text, profile, matches, metadata
    )

    # ── HALLUCINATION FIX 2: override brief.buyer_name with metadata value ────
    # PROBLEM: Both the parser and strategist could hallucinate a buyer name even
    # with the prompt fixes above (LLMs are probabilistic; no prompt is perfect).
    # The metadata dict — sourced from the leads file — is the ground truth.
    #
    # FIX: After the strategist returns, unconditionally set brief.buyer_name to
    # the metadata value if one exists. If metadata has no name either (e.g. a
    # truly anonymous submission), leave the brief's value as-is (likely null
    # after Fix 1A) rather than writing 'Unknown'.
    #
    # This mirrors the same override applied in evaluation.py (Fix 2 there), but
    # applying it here means every caller of process_lead gets clean output —
    # not just the evaluation pipeline.
    authoritative_name = metadata.get("buyer_name")  # None if not in metadata
    if authoritative_name and hasattr(brief, "buyer_name"):
        if brief.buyer_name != authoritative_name:
            logger.warning(
                f"[Fix 2] Hallucinated name corrected: "
                f"'{brief.buyer_name}' → '{authoritative_name}'"
            )
        brief.buyer_name = authoritative_name
    # ── END HALLUCINATION FIX 2 ──────────────────────────────────────────────

    brief.system_metrics = PerformanceMetrics(
        parser_latency_sec=round(parse_lat, 2),
        strategist_latency_sec=round(strat_lat, 2),
        input_tokens_total=p_in + s_in,
        output_tokens_total=p_out + s_out,
    )

    logger.info(
        f"Pipeline done: {buyer_name} | "
        f"{brief.system_metrics.total_latency_sec}s | "
        f"{brief.system_metrics.total_tokens} tokens"
    )
    return brief