import streamlit as st
import json
import os
from src.database.ingestion_vector import MLSVectorStore as MLSDataStore
from src.agent.orchestrator import process_lead

st.set_page_config(
    page_title="Buyer Lead Intake Platform",
    layout="wide",
    page_icon="🏡",
)

@st.cache_resource
def initialize_system_datastore():
    return MLSDataStore("miami_mls_listings.csv")

@st.cache_data
def load_historical_leads():
    if os.path.exists("sample_buyer_inquiries.json"):
        with open("sample_buyer_inquiries.json") as f:
            return json.load(f)
    return []

db          = initialize_system_datastore()
leads_pool  = load_historical_leads()

st.title("🏡 Buyer Lead Intake Platform")
st.caption("Convert raw multi-channel buyer inquiries into structured property briefings.")

st.sidebar.header("📥 Lead Input")
input_mode = st.sidebar.radio("Source", ["Lead Pool", "Manual Entry"])

selected_lead_metadata = {}
active_inquiry_text    = ""

if input_mode == "Lead Pool" and leads_pool:
    lead_options = [
        f"{l.get('buyer_name', 'Unknown')} ({l.get('lead_id', 'N/A')})"
        for l in leads_pool
    ]
    selected_idx = st.sidebar.selectbox(
        "Select lead:",
        range(len(lead_options)),
        format_func=lambda x: lead_options[x],
    )
    selected_lead_metadata = leads_pool[selected_idx]
    active_inquiry_text    = selected_lead_metadata.get("message", "")

    st.sidebar.markdown("---")
    st.sidebar.subheader("👤 Client Profile")
    st.sidebar.caption(f"**Lead ID:** {selected_lead_metadata.get('lead_id')}")
    st.sidebar.caption(f"**Name:** {selected_lead_metadata.get('buyer_name')}")
    st.sidebar.caption(f"**Message:** {selected_lead_metadata.get('message')}")
    st.sidebar.markdown(f"**Email:** `{selected_lead_metadata.get('buyer_email')}`")
    st.sidebar.markdown(f"**Phone:** {selected_lead_metadata.get('buyer_phone')}")
    st.sidebar.markdown(f"**Channel:** `{selected_lead_metadata.get('channel')}`")
    st.sidebar.markdown(f"**Received:** {selected_lead_metadata.get('received_at')}")
else:
    active_inquiry_text    = st.sidebar.text_area("Paste buyer message:", height=250)
    selected_lead_metadata = {"buyer_name": "Walk-in", "channel": "manual_entry"}

if st.sidebar.button("⚡ Process Lead", type="primary"):
    if not active_inquiry_text.strip():
        st.warning("Message is empty — please enter a buyer inquiry.")
    else:
        # IMPROVEMENT 1: Real error boundary around the pipeline call.
        # Previously any exception (network, validation, quota) crashed the
        # entire Streamlit page with an unformatted Python traceback. Now
        # errors are caught and shown as a readable error message with the
        # raw exception for debugging.
        try:
            with st.spinner("Running pipeline..."):
                # IMPROVEMENT 2: Pass inquiry_text explicitly so the vector
                # store builds a richer semantic query (raw message + profile).
                # The old call only passed the structured profile, losing tone
                # and implicit context present in the free-text message.
                brief = process_lead(
                    active_inquiry_text,
                    db,
                    metadata=selected_lead_metadata,
                )

        except Exception as e:
            st.error(
                f"**Pipeline error:** {type(e).__name__}: {e}\n\n"
                "Check your API credentials and network connection, then try again."
            )
            st.stop()

        # ── KPI row ──────────────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Urgency", f"{brief.extracted_profile.urgency_score} / 10")

        # IMPROVEMENT 3: Budget display distinguishes "not stated" (None → "Not stated")
        # from genuinely flexible budgets. Previously both showed "Flexible".
        budget_val = brief.extracted_profile.budget_max
        m2.metric(
            "Max budget",
            f"${budget_val:,.0f}" if budget_val else "Not stated",
        )

        # IMPROVEMENT 4: Use computed_field from PerformanceMetrics instead of
        # manually summing two fields everywhere.
        m3.metric("Latency", f"{brief.system_metrics.total_latency_sec:.2f}s")
        m4.metric("Tokens",  f"{brief.system_metrics.total_tokens:,}")

        # ── Main panels ───────────────────────────────────────────────────────
        left_panel, right_panel = st.columns([1, 1])

        with left_panel:
            st.subheader("📋 Buyer Profile")
            st.info(brief.buyer_summary)

            neighborhoods = brief.extracted_profile.neighborhoods
            must_haves    = brief.extracted_profile.must_haves

            st.markdown(
                f"**Neighbourhoods:** {', '.join(neighborhoods) if neighborhoods else 'Not specified'}"
            )
            st.markdown(
                f"**Must-haves:** {', '.join(must_haves) if must_haves else 'None stated'}"
            )

            if brief.human_in_the_loop_flags:
                st.subheader("⚠️ Risk Flags")
                for flag in brief.human_in_the_loop_flags:
                    st.error(flag)

        with right_panel:
            st.subheader("🎯 Strategic Advice")
            st.success(brief.strategic_advice)

            st.markdown("---")
            st.markdown("**📋 Suggested Follow-up Message**")
            st.text_area("Copy and send to buyer:", value=brief.follow_up_message, height=120)

            st.subheader("🏘️ Property Matches")
            if not brief.recommended_properties:
                st.warning("No properties matched the search criteria.")
            else:
                for prop in brief.recommended_properties:
                    tier_badge = "🟢" if "Strict" in prop.fallback_tier_applied else "🟡"
                    with st.expander(
                        f"{tier_badge} {prop.address} — ${prop.price:,.0f}"
                    ):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.caption(f"**Tier:** {prop.fallback_tier_applied}")
                            st.caption(f"**Score:** {prop.match_score:.2f}")
                            st.caption(f"**Neighbourhood:** {prop.neighborhood}")
                        with col_b:
                            st.caption(f"**Beds:** {prop.bedrooms}")
                            # IMPROVEMENT 5: property_type now shown — it's in
                            # the schema (PropertyMatch.property_type) and produced
                            # by MLSVectorStore but was never displayed.
                            if prop.property_type:
                                st.caption(f"**Type:** {prop.property_type}")

                        st.markdown(f"**Features:** `{prop.features}`")
                        st.markdown(f"**Rationale:** {prop.match_rationale}")

                        # IMPROVEMENT 6: score_breakdown expander for transparency.
                        # Shows the four signal contributions so agents can see
                        # *why* a property scored the way it did.
                        if prop.score_breakdown:
                            with st.expander("Score breakdown"):
                                for signal, val in prop.score_breakdown.items():
                                    label = signal.replace("_", " ").title()
                                    st.progress(float(val), text=f"{label}: {val:.2f}")

        # ── Debug expanders ───────────────────────────────────────────────────
        st.markdown("---")
        c_tree, c_trace = st.columns(2)

        with c_tree:
            with st.expander("👁️ Parser chain-of-thought"):
                # IMPROVEMENT 7: Use st.markdown with unsafe_allow_html so the
                # text renders in a plain <div> rather than through Streamlit's
                # Markdown parser. This prevents underscores in snake_case words
                # (e.g. budget_max, offer_strategy) from being parsed as italic
                # markers, which was causing words to run together visually.
                reasoning = brief.extracted_profile.extraction_reasoning
                st.markdown(
                    f"<div style='white-space:pre-wrap; font-size:0.875rem;'>"
                    f"{reasoning}</div>",
                    unsafe_allow_html=True,
                )

        with c_trace:
            with st.expander("🛠️ Full telemetry log"):
                st.json(brief.model_dump())

else:
    st.info("Select a lead from the sidebar and click **Process Lead** to begin.")