# Deploying to Google Cloud Run

This repository includes a Dockerfile that runs the Streamlit app on the
Cloud Run-required port (8080).

Service URL (current):
https://bayut-competitor-ai-798732426681.us-central1.run.app/

## Prerequisites
- gcloud CLI installed and authenticated
- Project set (PROJECT_ID)
- Artifact Registry or Container Registry enabled

## Build and deploy (Dockerfile)
1) Set the project:
   gcloud config set project <PROJECT_ID>

2) Build the container:
   gcloud builds submit --tag gcr.io/<PROJECT_ID>/bayut-competitor-ai

3) Deploy to Cloud Run:
   gcloud run deploy bayut-competitor-ai \
     --image gcr.io/<PROJECT_ID>/bayut-competitor-ai \
     --region us-central1 \
     --platform managed \
     --allow-unauthenticated \
     --port 8080

## Environment variables
Set required secrets via Cloud Run (examples):
gcloud run services update bayut-competitor-ai \
  --region us-central1 \
  --set-env-vars OPENAI_API_KEY=...,SERPAPI_API_KEY=...,DATAFORSEO_LOGIN=...,DATAFORSEO_PASSWORD=...
