#!/bin/sh
# Railway cron job: triggers the digest pipeline via the main app's API endpoint.
# Required env vars: CRON_SECRET, RAILWAY_SERVICE_AI_NEWS_AGENT_URL

APP_URL="${RAILWAY_SERVICE_AI_NEWS_AGENT_URL:-ripin.ai}"

echo "Triggering digest at https://${APP_URL}/cron/run-digest ..."
curl -sf -X POST "https://${APP_URL}/cron/run-digest" \
  -H "Authorization: Bearer ${CRON_SECRET}" \
  -H "Content-Type: application/json"
STATUS=$?

if [ $STATUS -eq 0 ]; then
  echo "Digest triggered successfully"
else
  echo "Failed to trigger digest (exit code: $STATUS)"
  exit 1
fi
