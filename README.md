# bayut-competitor-ai

Streamlit app for Bayut competitor gap analysis. This repository includes a
Cloud Run deployment configuration so the latest version of the app can be
built and deployed on Google Cloud.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Google Cloud Run

### One-time setup

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com
```

### Deploy with Cloud Build (recommended)

```bash
gcloud builds submit \
  --config cloudbuild.yaml \
  --substitutions _REGION=us-central1,_SERVICE=bayut-competitor-ai \
  .
```

### Deploy manually (Docker build + run)

```bash
gcloud builds submit --tag gcr.io/$PROJECT_ID/bayut-competitor-ai:latest .
gcloud run deploy bayut-competitor-ai \
  --image gcr.io/$PROJECT_ID/bayut-competitor-ai:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated
```

## Runtime configuration

Set these environment variables in Cloud Run if you use the related features:

- `SERPAPI_API_KEY`
- `DATAFORSEO_LOGIN` or `DATAFORSEO_EMAIL`
- `DATAFORSEO_PASSWORD` or `DATAFORSEO_API_PASSWORD`
- `DATAFORSEO_LOCATION_CODE` (optional)
- `DATAFORSEO_LOCATION_NAME` (default: United Arab Emirates)
- `DATAFORSEO_LANGUAGE_CODE` (default: en)
- `DATAFORSEO_SE_DOMAIN` (default: google.ae)
- `DATAFORSEO_DEPTH` (default: 50)