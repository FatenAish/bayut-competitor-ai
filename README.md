# bayut-competitor-ai

Streamlit app for Bayut competitor gap analysis, SEO, and content quality.

## Configuration

The app reads secrets from environment variables (preferred) or `st.secrets`.

Required:
- `DATAFORSEO_LOGIN`
- `DATAFORSEO_PASSWORD`

Optional:
- `SERPAPI_API_KEY`
- `DATAFORSEO_LOCATION_CODE`
- `DATAFORSEO_LOCATION_NAME` (defaults to "United Arab Emirates")
- `DATAFORSEO_LANGUAGE_CODE` (defaults to "en")
- `DATAFORSEO_SE_DOMAIN` (defaults to "google.ae")
- `DATAFORSEO_DEPTH` (defaults to 50)

**Never commit secrets to git.** Use `.env` locally or Google Secret Manager in GCP.

## Local development

1. Copy `.env.example` to `.env` and fill in values.
2. Export the variables or use a dotenv loader.

Example:
```bash
export DATAFORSEO_LOGIN="your-login"
export DATAFORSEO_PASSWORD="your-password"
export SERPAPI_API_KEY="your-serpapi-key"
```

## Google Cloud (Cloud Run + Secret Manager)

Create secrets:
```bash
gcloud secrets create DATAFORSEO_LOGIN --replication-policy="automatic"
printf "%s" "your-login" | gcloud secrets versions add DATAFORSEO_LOGIN --data-file=-

gcloud secrets create DATAFORSEO_PASSWORD --replication-policy="automatic"
printf "%s" "your-password" | gcloud secrets versions add DATAFORSEO_PASSWORD --data-file=-

gcloud secrets create SERPAPI_API_KEY --replication-policy="automatic"
printf "%s" "your-serpapi-key" | gcloud secrets versions add SERPAPI_API_KEY --data-file=-
```

Grant your Cloud Run service account access to the secrets:
```bash
gcloud secrets add-iam-policy-binding DATAFORSEO_LOGIN \
  --member="serviceAccount:YOUR_RUN_SA" \
  --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding DATAFORSEO_PASSWORD \
  --member="serviceAccount:YOUR_RUN_SA" \
  --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding SERPAPI_API_KEY \
  --member="serviceAccount:YOUR_RUN_SA" \
  --role="roles/secretmanager.secretAccessor"
```

Deploy with secrets bound to env vars:
```bash
gcloud run deploy YOUR_SERVICE \
  --image gcr.io/YOUR_PROJECT/YOUR_IMAGE \
  --set-secrets \
DATAFORSEO_LOGIN=DATAFORSEO_LOGIN:latest,\
DATAFORSEO_PASSWORD=DATAFORSEO_PASSWORD:latest,\
SERPAPI_API_KEY=SERPAPI_API_KEY:latest \
  --set-env-vars \
DATAFORSEO_LOCATION_NAME="United Arab Emirates",\
DATAFORSEO_LANGUAGE_CODE=en,\
DATAFORSEO_SE_DOMAIN=google.ae,\
DATAFORSEO_DEPTH=50
```

If you don't use SerpAPI, omit the `SERPAPI_API_KEY` secret.