# Buyer Lead Intake Platform

An LLM-powered real estate lead processing system that converts raw multi-channel buyer inquiries into structured, actionable property briefings. Built on Google Vertex AI (Gemini) with a hybrid vector + hard-filter property search engine and a Streamlit UI.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Local Setup](#local-setup)
- [Running the App](#running-the-app)
- [Running the Evaluation Pipeline](#running-the-evaluation-pipeline)
- [Docker](#docker)
- [Deploying to Google Cloud Run](#deploying-to-google-cloud-run)
- [Environment Variables](#environment-variables)
- [Architecture Overview](#architecture-overview)
- [Troubleshooting](#troubleshooting)

---

## Project Structure

```
.
├── app.py                          # Streamlit UI entry point
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container definition for Cloud Run
├── miami_mls_listings.csv          # MLS property dataset (206 active listings)
├── sample_buyer_inquiries.json     # 12 sample buyer leads for the lead pool
├── .env                            # Local secrets (not committed — see below)
└── src/
    ├── agent/
    │   ├── orchestrator.py         # Two-agent pipeline (Parser + Strategist)
    │   ├── schemas.py              # Pydantic models for all data types
    │   └── evaluation.py           # LLM-as-a-Judge evaluation pipeline
    └── database/
        ├── ingestion.py            # Keyword-based MLS store (baseline)
        └── ingestion_vector.py     # Hybrid vector + filter MLS store (production)
```

---

## Prerequisites

| Requirement          | Version | Notes                               |
| -------------------- | ------- | ----------------------------------- |
| Python               | 3.11+   | Match the Dockerfile runtime        |
| Google Cloud project | —       | With Vertex AI API enabled          |
| `gcloud` CLI         | Latest  | For local auth and Cloud Run deploy |
| Docker               | 20.10+  | Only needed for container builds    |

### Enable Vertex AI

```bash
gcloud services enable aiplatform.googleapis.com --project YOUR_PROJECT_ID
```

### Authenticate locally

```bash
gcloud auth application-default login
```

This writes credentials to `~/.config/gcloud/application_default_credentials.json`, which the `google-genai` SDK picks up automatically without an explicit API key.

---

## Local Setup

### 1. Clone and enter the repo

```bash
git clone <your-repo-url>
cd <repo-directory>
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create your `.env` file

Create a file named `.env` in the project root:

```env
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-central1
```

> `GOOGLE_CLOUD_LOCATION` must be a region where Gemini models are available on Vertex AI. `us-central1` works for all models used in this project.

### 5. Verify your data files are present

```bash
ls miami_mls_listings.csv sample_buyer_inquiries.json
```

Both files must be in the project root. The app will fail to start without the CSV.

---

## Running the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` by default.

**First run note:** If you are using `MLSVectorStore` (the production ingestion backend), the app embeds all 206 listings via the Vertex AI embedding API on startup. This takes approximately 1–3 seconds and is cached for the session. Subsequent leads process without re-embedding.

### Switching between ingestion backends

The default `app.py` imports `MLSDataStore` (keyword-based). To switch to the vector backend, update the import in `app.py`:

```python
# Keyword baseline (default):
from src.database.ingestion import MLSDataStore

# Hybrid vector + filter (production):
from src.database.ingestion_vector import MLSVectorStore as MLSDataStore
```

No other code changes are needed — both classes expose the same `search_properties()` interface.

---

## Running the Evaluation Pipeline

The evaluation pipeline processes all leads in `sample_buyer_inquiries.json` through the full agent pipeline and scores each output using a separate LLM-as-a-Judge call.

### Basic run (uses default file paths)

```bash
python -m src.agent.evaluation
```

### Custom paths

```bash
python -m src.agent.evaluation \
  --leads path/to/your_leads.json \
  --output path/to/report.json \
  --csv path/to/mls_listings.csv
```

### Output

The pipeline writes a JSON report (default: `evaluation_report.json`) and prints a summary to stdout:

```
============================================================
  EVALUATION SUMMARY  —  12 passed / 0 failed
============================================================

Score averages (out of 5):
  Faithfulness:  4.58
  Completeness:  4.75
  Actionability: 4.42

Performance averages:
  Avg latency:   18.34s
  Avg tokens:    3721
============================================================
```

**Rate limit note:** The evaluation pipeline processes leads sequentially with adaptive back-off (20s → 40s → 60s on 429 errors). Running all 12 leads takes 3–8 minutes depending on quota.

---

## Docker

### Build the image

```bash
docker build -t lead-intake-platform .
```

### Run locally with Docker

```bash
docker run -p 8080:8080 \
  -e GOOGLE_CLOUD_PROJECT=your-project-id \
  -e GOOGLE_CLOUD_LOCATION=us-central1 \
  -v ~/.config/gcloud:/root/.config/gcloud:ro \
  lead-intake-platform
```

The `-v` flag mounts your local gcloud credentials into the container so it can authenticate with Vertex AI without an explicit service account key.

Open `http://localhost:8080` in your browser.

### Using a service account key (alternative auth for Docker)

```bash
docker run -p 8080:8080 \
  -e GOOGLE_CLOUD_PROJECT=your-project-id \
  -e GOOGLE_CLOUD_LOCATION=us-central1 \
  -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/key.json \
  -v /path/to/your-key.json:/secrets/key.json:ro \
  lead-intake-platform
```

---

## Deploying to Google Cloud Run

### 1. Configure your project and region

```bash
export PROJECT_ID=your-gcp-project-id
export REGION=us-central1
export IMAGE=gcr.io/$PROJECT_ID/lead-intake-platform
```

### 2. Build and push to Google Container Registry

```bash
gcloud builds submit --tag $IMAGE
```

Or build and push manually:

```bash
docker build -t $IMAGE .
docker push $IMAGE
```

### 3. Create a service account for the Cloud Run service

```bash
gcloud iam service-accounts create lead-intake-sa \
  --display-name="Lead Intake Platform SA"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:lead-intake-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

### 4. Deploy to Cloud Run

```bash
gcloud run deploy lead-intake-platform \
  --image $IMAGE \
  --platform managed \
  --region $REGION \
  --service-account lead-intake-sa@$PROJECT_ID.iam.gserviceaccount.com \
  --set-env-vars GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GOOGLE_CLOUD_LOCATION=$REGION \
  --memory 1Gi \
  --timeout 300 \
  --allow-unauthenticated
```

**Memory note:** The vector index for 206 listings uses approximately 50MB at runtime. `512Mi` is sufficient, but `1Gi` gives comfortable headroom if the MLS dataset grows.

**Timeout note:** `--timeout 300` allows up to 5 minutes per request. The strategist agent on a cold start with vector index build can approach 30–40 seconds on large lead sets; this timeout prevents premature termination.

### 5. Get the service URL

```bash
gcloud run services describe lead-intake-platform \
  --platform managed \
  --region $REGION \
  --format 'value(status.url)'
```

---

## Environment Variables

| Variable                         | Required | Description                                                                                                                              |
| -------------------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `GOOGLE_CLOUD_PROJECT`           | Yes      | GCP project ID where Vertex AI is enabled                                                                                                |
| `GOOGLE_CLOUD_LOCATION`          | Yes      | Vertex AI region (e.g. `us-central1`)                                                                                                    |
| `GOOGLE_APPLICATION_CREDENTIALS` | No       | Path to service account key JSON. Not needed when using `gcloud auth application-default login` or Cloud Run's attached service account. |

---

## Architecture Overview

```
Buyer inquiry (free text)
        │
        ▼
┌───────────────────┐
│   Input sanitiser  │  Strips prompt injection patterns before any LLM call
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   Parser Agent    │  Gemini Flash — extracts BuyerProfile (structured JSON)
│  (gemini-flash)   │  Fields: buyer_name, budget_max, reference_price,
└────────┬──────────┘  bedrooms_min, neighborhoods, urgency_score, must_haves
         │
         ▼
┌───────────────────────────────────────────┐
│         MLSVectorStore.search_properties() │
│                                            │
│  Stage 1: Hard filters                     │
│    • Active listings only                  │
│    • Budget gate (≤ 105% → 120% → none)   │
│    • Bedroom gate (min → min-1 → none)     │
│    • Neighbourhood + property type         │
│                                            │
│  Stage 2: Vector reranking                 │
│    • Build semantic query from text+profile│
│    • Embed query (text-embedding-004)      │
│    • Cosine similarity vs in-memory index  │
│    • Blend: 50% vector + 20% features     │
│             + 20% budget + 10% bedrooms   │
└────────┬──────────────────────────────────┘
         │  Top-8 candidates
         ▼
┌───────────────────┐
│ Strategist Agent  │  Gemini Flash — generates LeadBrief (structured JSON)
│  (gemini-flash)   │  Fields: buyer_summary, recommended_properties,
└────────┬──────────┘  strategic_advice, follow_up_message, risk_flags
         │
         ▼
┌───────────────────┐
│  Metadata override │  buyer_name always sourced from leads metadata,
│  + metrics attach  │  never from LLM output. Latency + token counts attached.
└────────┬──────────┘
         │
         ▼
    LeadBrief → Streamlit UI / Evaluation pipeline
```

### Models used

| Model                   | Role                               | Temperature                         |
| ----------------------- | ---------------------------------- | ----------------------------------- |
| `gemini-3.1-flash-lite` | Parser Agent                       | 0.1 (near-deterministic extraction) |
| `gemini-3.1-flash-lite` | Strategist Agent                   | 0.2 (slight creativity for advice)  |
| `gemini-2.5-flash`      | Evaluation Judge                   | 0.1 (consistent scoring)            |
| `text-embedding-004`    | MLS vector index + query embedding | —                                   |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'src'`**
Run the app from the project root, not from inside `src/`. The `src/` directory is a package imported as `src.agent.orchestrator`.

**`google.api_core.exceptions.PermissionDenied`**
Your credentials don't have the `aiplatform.user` role on the project. Run `gcloud auth application-default login` again, or check the service account IAM bindings.

**`429 Resource exhausted` during evaluation**
The evaluation pipeline has adaptive back-off built in. If you hit persistent quota errors, reduce the number of leads in your test file or request a quota increase in the GCP console under Vertex AI → Quotas.

**`FileNotFoundError: miami_mls_listings.csv`**
The CSV must be in the working directory you launch the app from. If you reorganise the project, update the path passed to `MLSDataStore` / `MLSVectorStore` in `app.py` and `evaluation.py`.

**Vector index seems stale after updating the CSV**
`MLSVectorStore` builds the vector index once per Python process using `@st.cache_resource`. Clear Streamlit's cache (top-right menu → Clear cache) or restart the process to force a rebuild.

**Streamlit reruns the whole page on every button click**
This is Streamlit's execution model. The `@st.cache_resource` on `initialize_system_datastore()` ensures the MLS store and vector index are not rebuilt on every rerun.
