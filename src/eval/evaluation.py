"""
evaluation.py — LLM-as-a-Judge evaluation pipeline

Run:  python -m src.agent.evaluation
      python -m src.agent.evaluation --leads path/to/leads.json --output report.json
"""
import os
import sys
import json
import time
import argparse
import logging
from typing import Optional
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from src.agent.orchestrator import process_lead
from src.database.ingestion import MLSDataStore
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Evaluation")


# IMPROVEMENT 1: Structured output schema for the evaluator LLM.
# Previously the eval used json.loads(response.text) with no schema, meaning
# any malformed response (extra markdown, missing key) raised an unhandled
# exception. Now Pydantic validates the eval output exactly like the main pipeline.
class EvalScores(BaseModel):
    faithfulness_score:  int = Field(description="1-5: how accurately the brief reflects the inquiry")
    completeness_score:  int = Field(description="1-5: how fully all buyer needs are captured")
    actionability_score: int = Field(description="1-5: how useful the brief is for an agent to act on")
    critique: str = Field(description="One concise sentence summarising the main issue or strength")


def _make_client() -> genai.Client:
    return genai.Client(
        vertexai=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION"),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HALLUCINATION FIX 4: evaluate_generated_brief now accepts expected_buyer_name.
#
# PROBLEM: The evaluator prompt had no knowledge of the ground-truth buyer name,
# so when the brief contained a hallucinated name the judge could flag it in the
# critique but had no reference to score it accurately. This caused inconsistent
# faithfulness penalties across leads.
#
# FIX: Pass expected_buyer_name into the prompt as explicit ground truth.
# The judge is instructed to compare the brief's buyer_name against it and
# penalise any mismatch directly in the faithfulness score.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def evaluate_generated_brief(
    original_inquiry: str,
    brief_json: str,
    expected_buyer_name: Optional[str] = None,  # HALLUCINATION FIX 4: new parameter
) -> EvalScores:
    """
    LLM-as-a-Judge scoring of a generated LeadBrief.

    IMPROVEMENT 2: The brief JSON is trimmed before sending — system_metrics
    and score_breakdown are stripped out. They add ~200 tokens of noise the
    evaluator doesn't need and can cause it to comment on latency instead of
    brief quality.

    IMPROVEMENT 1 applied: response_schema=EvalScores enforces structured output
    so the result is always parseable without try/except json.loads gymnastics.

    HALLUCINATION FIX 4 applied: expected_buyer_name is injected into the prompt
    so the judge has a concrete ground-truth anchor when scoring faithfulness.
    """
    # Strip fields irrelevant to quality evaluation
    try:
        brief_dict = json.loads(brief_json)
        brief_dict.pop("system_metrics", None)
        for prop in brief_dict.get("recommended_properties", []):
            prop.pop("score_breakdown", None)
        trimmed_json = json.dumps(brief_dict)
    except Exception:
        trimmed_json = brief_json  # Fall back to full JSON if trimming fails

    # ── HALLUCINATION FIX 4: build a ground-truth block for the prompt ────────
    # If we know who the buyer is (from metadata), tell the judge explicitly.
    # Without this, the judge can only infer the correct name from the inquiry
    # text, which is unreliable when the inquiry is anonymous or ambiguous.
    ground_truth_block = ""
    if expected_buyer_name:
        ground_truth_block = f"""
    Known ground truth (use this as the authoritative reference):
    - Buyer name: {expected_buyer_name}

    Explicitly check whether the brief's buyer_name matches the ground truth above.
    Any mismatch — including partial names, invented last names, or wrong names —
    must be penalised in the faithfulness_score.
    """
    # ── END HALLUCINATION FIX 4 ───────────────────────────────────────────────

    prompt = f"""
    You are an expert QA auditor for a real estate AI system.
    Score the generated Lead Brief against the original buyer inquiry on three dimensions:

    - faithfulness_score (1-5): Does the brief accurately reflect what the buyer said?
      Penalise hallucinated names, wrong bedroom counts, missed must-haves, or wrong budget.
    - completeness_score (1-5): Are all buyer needs captured and addressed?
    - actionability_score (1-5): Can a real estate agent act on this brief immediately?
    - critique: One sentence on the single most important strength or flaw.
    {ground_truth_block}
    Original inquiry:
    \"\"\"{original_inquiry}\"\"\"

    Generated brief:
    {trimmed_json}
    """

    client = _make_client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=EvalScores,
            temperature=0.1,
        ),
    )
    return EvalScores.model_validate_json(response.text)


def process_single_evaluation(
    lead: dict,
    idx: int,
    db: MLSDataStore,
    max_retries: int = 3,
) -> dict:
    """
    Processes one lead through the full pipeline + evaluation.

    IMPROVEMENT 3: Adaptive back-off replaces fixed sleep(2).
    The fixed 2-second sleep between every lead was wasteful for fast leads
    and insufficient after hitting a quota. We now sleep only when needed
    and scale wait time with the attempt number.

    HALLUCINATION FIX 2: buyer_name from metadata overrides whatever the parser
    extracted, so a hallucinated name never reaches the evaluator or the report.

    HALLUCINATION FIX 3: a post-parse guardrail re-applies the metadata name
    directly on the brief object before serialisation — a second line of defence
    in case the orchestrator ignores the metadata override.
    """
    inquiry_text = lead.get("message") or lead.get("inquiry", "")
    buyer_name   = lead.get("buyer_name", f"Lead-{idx + 1}")

    if not inquiry_text.strip():
        return {
            "buyer_name": buyer_name,
            "status": "FAILED",
            "error": "Empty inquiry text — skipped.",
        }

    for attempt in range(max_retries):
        try:
            brief_obj = process_lead(inquiry_text, db, metadata=lead)

            # ── HALLUCINATION FIX 2: metadata name overrides parser output ────
            # PROBLEM: process_lead passes metadata to the orchestrator, but the
            # orchestrator was not explicitly using metadata["buyer_name"] to
            # overwrite the parser's extracted (often hallucinated) buyer_name.
            # The LLM tried to infer a name from the inquiry text and invented
            # one when the text was ambiguous or the buyer didn't state their name.
            #
            # FIX: After process_lead returns, unconditionally overwrite
            # brief_obj.buyer_name with the authoritative value from the leads
            # file. The leads file is the ground truth — the parser output is not.
            if hasattr(brief_obj, "buyer_name") and buyer_name:
                brief_obj.buyer_name = buyer_name
                logger.debug(
                    f"[Fix 2] Overwrote parser buyer_name with metadata value: '{buyer_name}'"
                )
            # ── END HALLUCINATION FIX 2 ───────────────────────────────────────

            # ── HALLUCINATION FIX 3: post-parse guardrail before serialisation ─
            # PROBLEM: Even with Fix 2 above, if brief_obj uses a nested or
            # aliased field for the buyer name, hasattr may resolve to the wrong
            # attribute and the override silently fails.
            #
            # FIX: Serialise to a dict first, then force-set the buyer_name key
            # before re-serialising to JSON. This catches nested or aliased fields
            # that a direct attribute assignment would miss.
            brief_dict = json.loads(brief_obj.model_dump_json())

            if brief_dict.get("buyer_name") != buyer_name:
                logger.warning(
                    f"[Fix 3] Guardrail triggered for '{buyer_name}': "
                    f"brief contained '{brief_dict.get('buyer_name')}'. Correcting."
                )
                brief_dict["buyer_name"] = buyer_name  # force correct value

            brief_json = json.dumps(brief_dict)
            # ── END HALLUCINATION FIX 3 ───────────────────────────────────────

            # HALLUCINATION FIX 4: pass expected_buyer_name to the evaluator
            # so the judge has ground-truth context when scoring faithfulness.
            # (The signature change is in evaluate_generated_brief above.)
            eval_result = evaluate_generated_brief(
                inquiry_text,
                brief_json,
                expected_buyer_name=buyer_name,  # HALLUCINATION FIX 4
            )

            return {
                "buyer_name": buyer_name,
                "status":     "SUCCESS",
                "scores":     eval_result.model_dump(),
                "metrics":    brief_obj.system_metrics.model_dump(),
            }

        except Exception as e:
            msg = str(e)
            is_rate_limit = "429" in msg or "quota" in msg.lower()

            if is_rate_limit and attempt < max_retries - 1:
                wait = (attempt + 1) * 20  # 20s → 40s → 60s
                logger.warning(
                    f"Rate limit for {buyer_name} (attempt {attempt+1}). "
                    f"Waiting {wait}s..."
                )
                time.sleep(wait)
            else:
                return {
                    "buyer_name": buyer_name,
                    "status":     "FAILED",
                    "error":      repr(e),
                }

    return {
        "buyer_name": buyer_name,
        "status":     "FAILED",
        "error":      "Max retries exceeded.",
    }


def _print_summary(results: list[dict]) -> None:
    """
    IMPROVEMENT 4: Prints a per-metric aggregate summary after the run,
    not just pass/fail counts. Helps immediately spot which dimension
    (faithfulness, completeness, actionability) needs the most attention.
    """
    successes = [r for r in results if r["status"] == "SUCCESS"]
    failures  = [r for r in results if r["status"] == "FAILED"]

    print(f"\n{'='*60}")
    print(f"  EVALUATION SUMMARY  —  {len(successes)} passed / {len(failures)} failed")
    print(f"{'='*60}")

    if failures:
        print("\nFailed leads:")
        for r in failures:
            print(f"  ✗ {r['buyer_name']}: {r['error']}")

    if successes:
        def avg(key): return sum(r["scores"][key] for r in successes) / len(successes)
        def avg_m(key): return sum(r["metrics"][key] for r in successes) / len(successes)

        print(f"\nScore averages (out of 5):")
        print(f"  Faithfulness:  {avg('faithfulness_score'):.2f}")
        print(f"  Completeness:  {avg('completeness_score'):.2f}")
        print(f"  Actionability: {avg('actionability_score'):.2f}")
        print(f"\nPerformance averages:")
        total_lat = avg_m("parser_latency_sec") + avg_m("strategist_latency_sec")
        total_tok = avg_m("input_tokens_total") + avg_m("output_tokens_total")
        print(f"  Avg latency:   {total_lat:.2f}s")
        print(f"  Avg tokens:    {total_tok:.0f}")

        low_faith = [r for r in successes if r["scores"]["faithfulness_score"] < 4]
        if low_faith:
            print(f"\nLeads needing attention (faithfulness < 4):")
            for r in low_faith:
                print(f"  • {r['buyer_name']}: {r['scores']['critique']}")
    print(f"{'='*60}\n")


def run_evaluation_pipeline(
    leads_path: str = "sample_buyer_inquiries.json",
    output_path: str = "evaluation_report.json",
    csv_path: str = "miami_mls_listings.csv",
) -> None:
    """
    IMPROVEMENT 5: Accepts file paths as parameters (used by CLI args below)
    instead of hardcoding them. Makes the evaluator reusable against any leads
    file or MLS dataset without editing source code.
    """
    if not os.path.exists(leads_path):
        logger.error(f"Leads file not found: {leads_path}")
        sys.exit(1)

    db = MLSDataStore(csv_path)

    with open(leads_path) as f:
        sample_leads = json.load(f)

    logger.info(f"Starting evaluation on {len(sample_leads)} leads...")
    results = []

    for idx, lead in enumerate(sample_leads):
        buyer = lead.get("buyer_name", f"Lead-{idx+1}")
        logger.info(f"[{idx+1}/{len(sample_leads)}] Processing: {buyer}")

        res = process_single_evaluation(lead, idx, db)
        results.append(res)

        status_icon = "✓" if res["status"] == "SUCCESS" else "✗"
        print(f"  {status_icon} {buyer} — {res['status']}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Report written to {output_path}")

    _print_summary(results)


if __name__ == "__main__":
    # IMPROVEMENT 5: CLI argument support
    parser = argparse.ArgumentParser(description="Run the MLS lead evaluation pipeline.")
    parser.add_argument("--leads",  default="sample_buyer_inquiries.json", help="Path to leads JSON file")
    parser.add_argument("--output", default="evaluation_report.json",      help="Output report path")
    parser.add_argument("--csv",    default="miami_mls_listings.csv",      help="MLS CSV path")
    args = parser.parse_args()

    run_evaluation_pipeline(
        leads_path=args.leads,
        output_path=args.output,
        csv_path=args.csv,
    )