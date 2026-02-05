# Google Cloud Run deployment

This repository is a Streamlit app (`app.py`). The Dockerfile included here is
ready for Cloud Run.

## Prerequisites

- Install and authenticate the gcloud CLI.
- Select your project: `gcloud config set project YOUR_PROJECT_ID`
- Enable APIs:
  - `run.googleapis.com`
  - `cloudbuild.googleapis.com`
  - `artifactregistry.googleapis.com`

## Build and deploy (new service)

```bash
PROJECT_ID="$(gcloud config get-value project)"
SERVICE_NAME="bayut-competitor-ai"
REGION="us-central1"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

gcloud builds submit --tag "${IMAGE}"

gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --port 8080
```

After deployment, fetch the service URL:

```bash
gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --format="value(status.url)"
```

## Environment variables

Some features need API keys. Set them during deploy or later with `services update`.

```bash
gcloud run services update "${SERVICE_NAME}" \
  --region "${REGION}" \
  --set-env-vars "SERPAPI_API_KEY=YOUR_KEY"
```

DataForSEO is supported as an alternative (login + password):

```bash
gcloud run services update "${SERVICE_NAME}" \
  --region "${REGION}" \
  --set-env-vars "DATAFORSEO_LOGIN=YOUR_LOGIN,DATAFORSEO_PASSWORD=YOUR_PASSWORD,DATAFORSEO_LOCATION_CODE=YOUR_LOCATION_CODE,DATAFORSEO_LOCATION_NAME=United Arab Emirates,DATAFORSEO_LANGUAGE_CODE=en,DATAFORSEO_SE_DOMAIN=google.ae,DATAFORSEO_DEPTH=50"
```

## Local container test

```bash
docker build -t bayut-competitor-ai .
docker run --rm -p 8080:8080 \
  -e SERPAPI_API_KEY=YOUR_KEY \
  bayut-competitor-ai
```
