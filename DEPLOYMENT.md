# Cloud Run deployment (Cloud Shell)

This repository is a Streamlit app. The included Dockerfile starts Streamlit
on the Cloud Run port (`$PORT`, default 8080).

## Service details

- Service: `bayut-competitor-ai`
- Region: `us-central1`
- Current URL: `https://bayut-competitor-ai-798732426681.us-central1.run.app`

## Prerequisites

- A Google Cloud project with **Cloud Run** and **Cloud Build** APIs enabled
- Access to Cloud Shell in that project

## Deploy from source (recommended)

1. Open Cloud Shell and change into the repo directory:

   ```bash
   cd ~/bayut-competitor-ai
   ```

2. Set your project and region:

   ```bash
   gcloud config set project PROJECT_ID
   gcloud config set run/region us-central1
   ```

3. (Optional) Provide API keys as environment variables:

   - `SERPAPI_API_KEY`
   - `DATAFORSEO_LOGIN` / `DATAFORSEO_PASSWORD`
   - `DATAFORSEO_LOCATION_CODE`
   - `DATAFORSEO_LOCATION_NAME`
   - `DATAFORSEO_LANGUAGE_CODE`
   - `DATAFORSEO_SE_DOMAIN`
   - `DATAFORSEO_DEPTH`

   You can set them during deploy:

   ```bash
   gcloud run deploy bayut-competitor-ai \
     --region us-central1 \
     --source . \
     --allow-unauthenticated \
     --set-env-vars "SERPAPI_API_KEY=YOUR_KEY,DATAFORSEO_LOGIN=YOUR_LOGIN,DATAFORSEO_PASSWORD=YOUR_PASSWORD"
   ```

   If you do not provide these, the app still runs, but some features will be
   disabled.

4. Deploy:

   ```bash
   gcloud run deploy bayut-competitor-ai \
     --region us-central1 \
     --source . \
     --allow-unauthenticated
   ```

5. Retrieve the service URL:

   ```bash
   gcloud run services describe bayut-competitor-ai \
     --region us-central1 \
     --format='value(status.url)'
   ```

## Deploy using an image (alternative)

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/bayut-competitor-ai
gcloud run deploy bayut-competitor-ai \
  --region us-central1 \
  --image gcr.io/PROJECT_ID/bayut-competitor-ai \
  --allow-unauthenticated
```

## Update environment variables later

```bash
gcloud run services update bayut-competitor-ai \
  --region us-central1 \
  --set-env-vars "SERPAPI_API_KEY=NEW_VALUE"
```
