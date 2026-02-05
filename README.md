# bayut-competitor-ai

Streamlit app for competitor gap analysis.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Google Cloud (Cloud Run from Cloud Shell)

1. Open Google Cloud Shell and clone this repo.
2. Set your project and enable required APIs:

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

3. Deploy to Cloud Run (builds from source using the Dockerfile):

```bash
gcloud run deploy bayut-competitor-ai \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

4. (Optional) Add API keys as environment variables:

```bash
gcloud run services update bayut-competitor-ai \
  --region us-central1 \
  --update-env-vars "SERPAPI_API_KEY=YOUR_KEY,DATAFORSEO_LOGIN=YOUR_LOGIN,DATAFORSEO_PASSWORD=YOUR_PASSWORD"
```

Notes:
- The app listens on port 8080 in Cloud Run.
- You can use any of the supported DataForSEO keys (e.g. `DATAFORSEO_API_LOGIN` / `DATAFORSEO_API_PASSWORD`).