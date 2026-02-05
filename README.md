# bayut-competitor-ai

Streamlit app for Bayut competitor gap analysis.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Google Cloud Run (from scratch)

The commands below use the provided project ID:

```bash
gcloud auth login
gcloud config set project cool-album-478007-n4
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

Create an Artifact Registry repo (one-time):

```bash
gcloud artifacts repositories create bayut-competitor-ai \
  --repository-format=docker \
  --location=us-central1 \
  --description="Docker images for bayut-competitor-ai"
```

Build and push the container image:

```bash
gcloud builds submit \
  --tag us-central1-docker.pkg.dev/cool-album-478007-n4/bayut-competitor-ai/app
```

Deploy to Cloud Run:

```bash
gcloud run deploy bayut-competitor-ai \
  --image us-central1-docker.pkg.dev/cool-album-478007-n4/bayut-competitor-ai/app \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars SERPAPI_API_KEY=YOUR_KEY,DATAFORSEO_LOGIN=YOUR_LOGIN,DATAFORSEO_PASSWORD=YOUR_PASSWORD
```

Fetch the service URL:

```bash
gcloud run services describe bayut-competitor-ai \
  --region us-central1 \
  --format "value(status.url)"
```

## Configuration

These environment variables are optional but unlock more features:

- `SERPAPI_API_KEY`
- `DATAFORSEO_LOGIN`
- `DATAFORSEO_PASSWORD`
- `DATAFORSEO_LOCATION_CODE` (optional override)
- `DATAFORSEO_LOCATION_NAME` (default: "United Arab Emirates")
- `DATAFORSEO_LANGUAGE_CODE` (default: "en")
- `DATAFORSEO_SE_DOMAIN` (default: "google.ae")
- `DATAFORSEO_DEPTH` (default: 50)