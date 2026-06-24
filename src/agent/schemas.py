from pydantic import BaseModel, Field, field_validator, computed_field
from typing import List, Optional

class PerformanceMetrics(BaseModel):
    parser_latency_sec: float = Field(default=0.0)
    strategist_latency_sec: float = Field(default=0.0)
    input_tokens_total: int = Field(default=0)
    output_tokens_total: int = Field(default=0)

    # IMPROVEMENT 1: computed convenience fields so app.py and evaluation.py
    # don't have to manually add two fields every time they display latency/tokens.
    @computed_field
    @property
    def total_latency_sec(self) -> float:
        return round(self.parser_latency_sec + self.strategist_latency_sec, 2)

    @computed_field
    @property
    def total_tokens(self) -> int:
        return self.input_tokens_total + self.output_tokens_total


class BuyerProfile(BaseModel):
    # IMPROVEMENT 2: field validators enforce LLM output sanity.
    # Without these, the LLM can return budget_max=-1, urgency_score=11,
    # or bedrooms_min=99 — all of which silently corrupt downstream logic.

    # ── HALLUCINATION FIX 1A: buyer_name added to BuyerProfile ───────────────
    # PROBLEM: BuyerProfile had no buyer_name field. Because the Gemini
    # structured-output schema is derived directly from this Pydantic model,
    # buyer_name was absent from the JSON schema sent to the LLM. The parser
    # prompt in orchestrator.py (Fix 1A there) now instructs the LLM to extract
    # buyer_name and return null when not stated — but that instruction is
    # silently ignored if the field isn't in the response schema.
    #
    # FIX: Add buyer_name as Optional[str] = None so that:
    #   1. Gemini includes it in the structured output schema it enforces.
    #   2. The parser can return null (Python None) when the buyer is anonymous.
    #   3. The orchestrator's Fix 2 override (brief.buyer_name = authoritative)
    #      has a real attribute to write to instead of creating a dynamic one.
    #
    # The field_validator below strips accidental whitespace and rejects
    # placeholder strings ('Unknown', 'N/A', etc.) that the LLM might still
    # produce despite the prompt instruction, converting them to None so the
    # downstream override in orchestrator.py always has a clean starting point.
    buyer_name: Optional[str] = Field(
        default=None,
        description=(
            "The buyer's name, extracted verbatim from the inquiry text. "
            "Return null if the buyer does not explicitly state their name. "
            "Never infer, guess, or fabricate a name."
        ),
    )

    @field_validator("buyer_name", mode="before")
    @classmethod
    def validate_buyer_name(cls, v):
        """
        Sanitise the LLM-returned buyer name.

        Even with a strict prompt, the LLM occasionally returns placeholder
        strings ('Unknown', 'N/A', 'Not mentioned') instead of null. This
        validator normalises those to None so the orchestrator's metadata
        override (Fix 2) always operates on a clean null rather than a
        misleading string.
        """
        if v is None:
            return None
        cleaned = str(v).strip()
        # Reject placeholder strings the LLM uses when it can't find a name
        _PLACEHOLDER_NAMES = {
            "", "unknown", "n/a", "na", "not provided", "not mentioned",
            "not stated", "anonymous", "buyer", "null", "none",
        }
        if cleaned.lower() in _PLACEHOLDER_NAMES:
            return None
        return cleaned
    # ── END HALLUCINATION FIX 1A ─────────────────────────────────────────────

    budget_max: Optional[int] = Field(
        default=None,
        description="Maximum budget in USD. Null if not specified."
    )
    # Add to BuyerProfile, after budget_max:
    reference_price: Optional[int] = Field(
        default=None,
        description=(
            "The asking/listing price of a property the buyer explicitly mentions "
            "as context — distinct from their own budget or offer. Used to anchor "
            "property search to the correct price tier when the buyer's offer price "
            "is below the market price they are actually shopping in."
        ), 
    )
    bedrooms_min: Optional[int] = Field(
        default=None,
        description="Minimum number of bedrooms needed."
    )
    neighborhoods: List[str] = Field(
        default_factory=list,
        description="List of desired neighborhoods or cities."
    )
    urgency_score: int = Field(
        default=5,
        description="Calculated urgency from 1-10 based on context clues like timeline and language."
    )
    must_haves: List[str] = Field(
        default_factory=list,
        description="Strict non-negotiable requirements (e.g., pool, waterfront, pet-friendly)."
    )
    extraction_reasoning: str = Field(
        default="",
        description="Chain-of-thought detailing why these parameters were extracted from the raw text."
    )

    @field_validator("urgency_score", mode="before")
    @classmethod
    def clamp_urgency(cls, v):
        """LLM occasionally returns 0 or values above 10. Clamp silently."""
        try:
            return max(1, min(10, int(v)))
        except (TypeError, ValueError):
            return 5  # Safe default if the LLM returns a non-integer.

    @field_validator("budget_max", mode="before")
    @classmethod
    def validate_budget(cls, v):
        """Reject nonsensical budgets (negative, zero, or astronomically high)."""
        if v is None:
            return None
        try:
            v = int(v)
        except (TypeError, ValueError):
            return None
        if v <= 0 or v > 500_000_000:  # $500M upper bound for Miami luxury market
            return None
        return v
    
    @field_validator("reference_price", mode="before")
    @classmethod
    def validate_reference_price(cls, v):
        """Same sanity check as budget_max."""
        if v is None:
            return None
        try:
            v = int(v)
        except (TypeError, ValueError):
            return None
        return v if 0 < v <= 500_000_000 else None

    @field_validator("bedrooms_min", mode="before")
    @classmethod
    def validate_bedrooms(cls, v):
        """Clamp bedroom count to a realistic range."""
        if v is None:
            return None
        try:
            v = int(v)
        except (TypeError, ValueError):
            return None
        return max(0, min(10, v))


class PropertyMatch(BaseModel):
    listing_id: str
    address: str
    price: int
    neighborhood: str
    bedrooms: int
    features: str
    # IMPROVEMENT 3: property_type added to schema to match the field now
    # produced by MLSVectorStore. Without this, Pydantic silently drops it.
    property_type: str = Field(default="")
    fallback_tier_applied: str = Field(
        description="Tier 1: Strict Match or Tier 2/3: Fallback."
    )
    match_score: float = Field(
        description="Composite relevance score between 0.0 and 1.0."
    )
    # IMPROVEMENT 4: score_breakdown surfaces individual signal contributions
    # for transparency in the UI and debugging in evaluation.
    score_breakdown: dict = Field(
        default_factory=dict,
        description="Per-signal score breakdown: vector_similarity, feature_coverage, budget_efficiency, bedroom_fit."
    )
    match_rationale: str = Field(
        description="A concise sentence explaining exactly why this property fits the buyer."
    )


class LeadBrief(BaseModel):
    # ── HALLUCINATION FIX 1A (continued): buyer_name added to LeadBrief ──────
    # PROBLEM: LeadBrief had no buyer_name field. The strategist LLM was
    # producing a buyer name inside buyer_summary prose, where it couldn't be
    # validated or overridden programmatically. The orchestrator's Fix 2
    # (brief.buyer_name = authoritative_name) had no attribute to write to,
    # so hasattr() returned False and the override was silently skipped.
    #
    # FIX: Add buyer_name as a top-level Optional[str] field on LeadBrief.
    # This gives the Gemini schema a named slot for the name (reducing inline
    # hallucination in buyer_summary), and gives orchestrator.py Fix 2 a real
    # attribute to overwrite with the authoritative metadata value.
    #
    # The strategist prompt in orchestrator.py already passes the correct name
    # via buyer_name_label (Fix 1B). This field ensures the LLM writes it into
    # a structured, overrideable slot rather than embedding it in free text.
    buyer_name: Optional[str] = Field(
        default=None,
        description=(
            "The buyer's name as provided in the inquiry metadata. "
            "Copy exactly from the 'Name' field in buyer context. "
            "Return null if name was listed as 'Not provided'."
        ),
    )
    # ── END HALLUCINATION FIX 1A (LeadBrief) ─────────────────────────────────

    buyer_summary: str = Field(
        description="A 3-sentence summary of the buyer's situation and key requirements."
    )
    extracted_profile: BuyerProfile
    recommended_properties: List[PropertyMatch] = Field(default_factory=list)
    strategic_advice: str = Field(
        description="Suggested next action and outreach script for the realtor, tailored to the inbound channel."
    )
    follow_up_message: str = Field(
        default="",
        description=(
            "A ready-to-send follow-up message the realtor can copy and send "
            "to the buyer verbatim. Should be warm, professional, and reference "
            "1-2 specific details from the buyer's inquiry. 3-4 sentences max."
        ),
    )
    human_in_the_loop_flags: List[str] = Field(
        default_factory=list,
        description="Risks, unrealistic expectations, or concerns requiring realtor intervention."
    )
    system_metrics: PerformanceMetrics = Field(default_factory=PerformanceMetrics)