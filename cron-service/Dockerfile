FROM curlimages/curl:8.12.1
CMD ["sh", "-c", "curl -sf -X POST https://ripin.ai/cron/run-digest -H \"Authorization: Bearer $CRON_SECRET\" -H \"Content-Type: application/json\" && echo ' OK'"]
