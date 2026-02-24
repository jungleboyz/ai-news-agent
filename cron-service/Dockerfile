FROM curlimages/curl:7.86.0
CMD ["sh", "-c", "curl -sf -X POST https://ripin.ai/cron/run-digest -H \"Authorization: Bearer $CRON_SECRET\" -H \"Content-Type: application/json\" && echo ' OK'"]
