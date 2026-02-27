#!/usr/bin/env python3
"""
Daily Digest Monitor for NEURAL_FEED (ripin.ai)

Checks whether the daily digest pipeline ran successfully, reports status,
and optionally triggers a retry if the digest is missing.

Usage:
    python scripts/monitor_digest.py                # Check only
    python scripts/monitor_digest.py --retry        # Check + auto-retry if missing
    python scripts/monitor_digest.py --verbose      # Detailed output

Environment variables:
    APP_URL         - Base URL (default: https://ripin.ai)
    CRON_SECRET     - Bearer token for /cron/run-digest endpoint
    SITE_USERNAME   - Login username (for authenticated endpoints)
    SITE_PASSWORD   - Login password (for authenticated endpoints)
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta

import requests

# --- Configuration ---
APP_URL = os.getenv("APP_URL", "https://ripin.ai").rstrip("/")
CRON_SECRET = os.getenv("CRON_SECRET", "")
SITE_USERNAME = os.getenv("SITE_USERNAME", "")
SITE_PASSWORD = os.getenv("SITE_PASSWORD", "")

# AEDT = UTC+11
AEDT = timezone(timedelta(hours=11))
AEST = timezone(timedelta(hours=10))

# Expected minimum items in a healthy digest
MIN_EXPECTED_ITEMS = 10


class DigestMonitor:
    """Monitor the daily digest pipeline."""

    def __init__(self, base_url: str, verbose: bool = False):
        self.base_url = base_url
        self.verbose = verbose
        self.session = requests.Session()
        self.session.timeout = 30
        self.issues: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []

    def log(self, msg: str):
        if self.verbose:
            print(f"  [DEBUG] {msg}")

    def _authenticate(self):
        """Authenticate via login form to get session cookie."""
        if not SITE_USERNAME or not SITE_PASSWORD:
            self.log("No credentials provided, skipping auth")
            return False
        try:
            resp = self.session.post(
                f"{self.base_url}/login",
                data={"username": SITE_USERNAME, "password": SITE_PASSWORD},
                allow_redirects=False,
            )
            if resp.status_code in (302, 303):
                self.log("Authenticated successfully")
                return True
            self.log(f"Auth failed with status {resp.status_code}")
            return False
        except Exception as e:
            self.log(f"Auth error: {e}")
            return False

    def check_health(self) -> bool:
        """Step 1: Check app health endpoint."""
        print("\n1. HEALTH CHECK")
        try:
            resp = self.session.get(f"{self.base_url}/health", timeout=10)
            data = resp.json()
            if data.get("status") == "healthy":
                print("   ✅ App is healthy")
                return True
            else:
                print(f"   ❌ Unhealthy response: {data}")
                self.issues.append(f"App health check failed: {data}")
                return False
        except requests.ConnectionError:
            print("   ❌ CRITICAL: Cannot reach the app (connection refused)")
            self.issues.append("App is DOWN — cannot connect")
            return False
        except requests.Timeout:
            print("   ❌ CRITICAL: Health check timed out")
            self.issues.append("App health check timed out")
            return False
        except Exception as e:
            print(f"   ❌ Health check error: {e}")
            self.issues.append(f"Health check error: {e}")
            return False

    def check_scheduler(self) -> dict | None:
        """Step 2: Check scheduler status."""
        print("\n2. SCHEDULER STATUS")
        try:
            resp = self.session.get(f"{self.base_url}/api/admin/scheduler")
            if resp.status_code == 401:
                # Try authenticating first
                if self._authenticate():
                    resp = self.session.get(f"{self.base_url}/api/admin/scheduler")

            if resp.status_code != 200:
                print(f"   ⚠️  Could not fetch scheduler status (HTTP {resp.status_code})")
                self.warnings.append(f"Scheduler endpoint returned {resp.status_code}")
                return None

            data = resp.json()
            enabled = data.get("enabled", False)
            running = data.get("running", False)
            next_run = data.get("next_run")

            if enabled and running:
                print(f"   ✅ Scheduler is enabled and running")
                print(f"      Next run: {next_run or 'unknown'}")
                self.info.append(f"Next scheduled run: {next_run}")
            elif not enabled:
                print("   ❌ Scheduler is DISABLED")
                self.issues.append("Scheduler is disabled — digests will not run automatically")
            elif not running:
                print("   ❌ Scheduler is not running (may need app restart)")
                self.issues.append("Scheduler is enabled but not running")

            return data
        except Exception as e:
            print(f"   ⚠️  Could not check scheduler: {e}")
            self.warnings.append(f"Scheduler check failed: {e}")
            return None

    def check_todays_digest(self) -> dict | None:
        """Step 3: Check if today's digest exists."""
        print("\n3. TODAY'S DIGEST")
        today_aedt = datetime.now(AEDT).strftime("%Y-%m-%d")
        today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        self.log(f"Today (AEDT): {today_aedt}, Today (UTC): {today_utc}")

        try:
            resp = self.session.get(f"{self.base_url}/api/digests", params={"per_page": 3})
            if resp.status_code == 401:
                if self._authenticate():
                    resp = self.session.get(f"{self.base_url}/api/digests", params={"per_page": 3})

            if resp.status_code != 200:
                print(f"   ⚠️  Could not fetch digests (HTTP {resp.status_code})")
                self.warnings.append(f"Digests endpoint returned {resp.status_code}")
                return None

            data = resp.json()
            digests = data.get("digests", [])

            if not digests:
                print("   ❌ No digests found at all")
                self.issues.append("No digests exist in the database")
                return None

            latest = digests[0]
            latest_date = latest.get("date", "")
            item_count = latest.get("item_count", 0)

            # Check if latest digest matches today (AEDT or UTC)
            if latest_date == today_aedt:
                print(f"   ✅ Today's digest exists ({today_aedt})")
                print(f"      Items: {item_count}")
                self.info.append(f"Digest {today_aedt}: {item_count} items")
                return latest
            elif latest_date == today_utc:
                print(f"   ✅ Today's digest exists ({today_utc} UTC)")
                print(f"      Items: {item_count}")
                self.info.append(f"Digest {today_utc}: {item_count} items")
                return latest
            else:
                print(f"   ❌ Today's digest is MISSING")
                print(f"      Expected: {today_aedt}")
                print(f"      Latest:   {latest_date} ({item_count} items)")
                self.issues.append(
                    f"Today's digest ({today_aedt}) is missing. "
                    f"Latest is from {latest_date}."
                )
                # Show last few for context
                if len(digests) > 1:
                    dates = [d.get("date", "?") for d in digests[:3]]
                    print(f"      Recent digests: {', '.join(dates)}")
                return None
        except Exception as e:
            print(f"   ❌ Error checking digests: {e}")
            self.issues.append(f"Failed to check digests: {e}")
            return None

    def check_digest_quality(self, digest: dict) -> None:
        """Step 4: Check digest content quality."""
        print("\n4. DIGEST QUALITY")
        digest_date = digest.get("date", "")
        item_count = digest.get("item_count", 0)

        if item_count == 0:
            print("   ❌ Digest has 0 items — pipeline likely failed mid-run")
            self.issues.append(f"Digest {digest_date} has 0 items")
            return

        if item_count < MIN_EXPECTED_ITEMS:
            print(f"   ⚠️  Low item count: {item_count} (expected {MIN_EXPECTED_ITEMS}+)")
            self.warnings.append(f"Low item count: {item_count}")

        # Fetch full digest details
        try:
            resp = self.session.get(f"{self.base_url}/api/digests/{digest_date}")
            if resp.status_code != 200:
                self.log(f"Could not fetch digest detail: {resp.status_code}")
                return

            detail = resp.json()
            items = detail.get("items", [])
            types = set(item.get("type") for item in items)
            sources_count = detail.get("news_sources_count", 0)
            podcast_count = detail.get("podcast_sources_count", 0)

            print(f"   Sources checked: {sources_count} news, {podcast_count} podcast")
            print(f"   Item types: {', '.join(sorted(types)) or 'none'}")

            if "news" not in types:
                print("   ⚠️  No news items found")
                self.warnings.append("No news items in today's digest")
            if "podcast" not in types and podcast_count > 0:
                print("   ⚠️  No podcast items despite podcast sources being configured")
                self.warnings.append("No podcast items despite sources configured")

            # Check for summaries
            with_summary = sum(1 for i in items if i.get("summary"))
            if items and with_summary < len(items) * 0.5:
                print(f"   ⚠️  Only {with_summary}/{len(items)} items have summaries")
                self.warnings.append(f"Only {with_summary}/{len(items)} items have summaries")
            else:
                print(f"   ✅ {with_summary}/{len(items)} items have summaries")

            if item_count >= MIN_EXPECTED_ITEMS:
                print(f"   ✅ Digest quality looks good")

        except Exception as e:
            self.log(f"Error checking digest quality: {e}")

    def check_timezone(self) -> None:
        """Step 5: Check timezone configuration."""
        print("\n5. TIMEZONE CHECK")
        now_utc = datetime.now(timezone.utc)
        now_aedt = now_utc.astimezone(AEDT)
        now_aest = now_utc.astimezone(AEST)

        # Determine if we're currently in AEDT (first Sun Oct to first Sun Apr)
        month = now_utc.month
        is_dst = 4 <= month <= 10  # Rough check (Australia DST is Oct-Apr)
        is_dst = not is_dst  # Invert: DST is active Oct-Apr, inactive Apr-Oct

        # More accurate: DST in Australia is first Sunday of October to first Sunday of April
        # For simplicity, Oct-Mar is AEDT, Apr-Sep is AEST
        if month >= 10 or month <= 3:
            tz_label = "AEDT (UTC+11)"
            expected_utc_hour = 19  # 6 AM AEDT = 19:00 UTC
        else:
            tz_label = "AEST (UTC+10)"
            expected_utc_hour = 20  # 6 AM AEST = 20:00 UTC

        print(f"   Current timezone: {tz_label}")
        print(f"   Local time: {now_aedt.strftime('%H:%M')} AEDT / {now_aest.strftime('%H:%M')} AEST")
        print(f"   For 6 AM local, SCHEDULER_HOUR should be: {expected_utc_hour} (UTC)")

        # Note: We can't check the actual env var on Railway from here,
        # but we can flag if the default (20) would be wrong
        if month >= 10 or month <= 3:
            print(f"   ⚠️  Currently in AEDT — default SCHEDULER_HOUR=20 runs at 7 AM, not 6 AM")
            print(f"      Set SCHEDULER_HOUR=19 in Railway env vars for 6 AM AEDT")
            self.warnings.append(
                "During AEDT (Oct-Mar), SCHEDULER_HOUR should be 19 for 6 AM local time. "
                "Default of 20 runs at 7 AM AEDT."
            )

    def retry_digest(self) -> bool:
        """Step 6: Trigger a digest retry."""
        print("\n6. AUTO-RETRY")
        if not CRON_SECRET:
            print("   ⚠️  CRON_SECRET not set — cannot trigger retry automatically")
            print("   💡 Set CRON_SECRET env var to enable auto-retry")
            self.warnings.append("Cannot auto-retry: CRON_SECRET not configured")
            return False

        try:
            resp = self.session.post(
                f"{self.base_url}/cron/run-digest",
                headers={
                    "Authorization": f"Bearer {CRON_SECRET}",
                    "Content-Type": "application/json",
                },
                timeout=30,
            )
            data = resp.json()
            if resp.status_code == 200 and data.get("status") == "started":
                print(f"   ✅ Digest retry triggered at {data.get('time', 'unknown')}")
                print("      Note: Pipeline runs in background, check again in ~10 minutes")
                self.info.append("Digest retry triggered successfully")
                return True
            elif data.get("status") == "skipped":
                print(f"   ⚠️  Retry skipped: {data.get('reason', 'unknown')}")
                self.warnings.append(f"Retry skipped: {data.get('reason')}")
                return False
            else:
                print(f"   ❌ Retry failed (HTTP {resp.status_code}): {data}")
                self.issues.append(f"Retry failed: {data}")
                return False
        except Exception as e:
            print(f"   ❌ Retry error: {e}")
            self.issues.append(f"Retry error: {e}")
            return False

    def print_report(self) -> int:
        """Print final summary report."""
        print("\n" + "=" * 60)
        print("📋 DIGEST MONITOR REPORT")
        print("=" * 60)
        now = datetime.now(AEDT)
        print(f"Time: {now.strftime('%Y-%m-%d %H:%M')} AEDT")
        print(f"App:  {self.base_url}")

        if self.issues:
            print(f"\n❌ ISSUES ({len(self.issues)}):")
            for issue in self.issues:
                print(f"   • {issue}")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   • {warning}")

        if self.info:
            print(f"\nℹ️  INFO:")
            for item in self.info:
                print(f"   • {item}")

        if not self.issues and not self.warnings:
            print("\n✅ All checks passed — digest is running normally!")
            return 0
        elif self.issues:
            print(f"\n🔴 {len(self.issues)} issue(s) need attention")
            return 1
        else:
            print(f"\n🟡 {len(self.warnings)} warning(s) to review")
            return 0


def main():
    parser = argparse.ArgumentParser(description="Monitor NEURAL_FEED daily digest")
    parser.add_argument("--retry", action="store_true", help="Auto-retry if digest is missing")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--url", default=APP_URL, help=f"App base URL (default: {APP_URL})")
    args = parser.parse_args()

    print("🔍 NEURAL_FEED Daily Digest Monitor")
    print(f"   Checking {args.url}...")

    monitor = DigestMonitor(args.url, verbose=args.verbose)

    # Run checks
    app_healthy = monitor.check_health()

    if app_healthy:
        monitor.check_scheduler()
        digest = monitor.check_todays_digest()

        if digest:
            monitor.check_digest_quality(digest)
        elif args.retry:
            monitor.retry_digest()

    monitor.check_timezone()

    # Final report
    exit_code = monitor.print_report()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
