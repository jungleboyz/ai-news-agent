"""Feed URL validation and testing service."""
import ipaddress
import socket

import feedparser
import requests
from urllib.parse import urlparse


def validate_url(url: str) -> str | None:
    """Validate a URL is safe to fetch (no SSRF).

    Returns None if safe, or an error string if blocked.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return "Invalid URL"

    if parsed.scheme not in ("http", "https"):
        return f"Scheme '{parsed.scheme}' not allowed (only http/https)"

    hostname = parsed.hostname
    if not hostname:
        return "No hostname in URL"

    # Resolve hostname to check for private IPs
    try:
        addrs = socket.getaddrinfo(hostname, None)
        for family, _, _, _, sockaddr in addrs:
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return f"URL resolves to private/reserved address"
    except socket.gaierror:
        return f"Cannot resolve hostname '{hostname}'"

    # Block cloud metadata endpoints
    if hostname in ("169.254.169.254", "metadata.google.internal"):
        return "Cloud metadata endpoint blocked"

    return None


class FeedValidator:
    """Validates and tests RSS/Atom/YouTube feed URLs."""

    TIMEOUT = 15

    @staticmethod
    def test_feed(url: str) -> dict:
        """Fetch a feed URL and return validation results.

        Returns:
            dict with keys: success, title, item_count, error
        """
        # SSRF check
        url_error = validate_url(url)
        if url_error:
            return {"success": False, "title": None, "item_count": 0, "error": url_error}

        try:
            # Fetch with a reasonable timeout and user-agent
            headers = {"User-Agent": "AI-News-Agent/1.0 (Feed Validator)"}
            resp = requests.get(url, headers=headers, timeout=FeedValidator.TIMEOUT)
            resp.raise_for_status()

            feed = feedparser.parse(resp.content)

            if feed.bozo and not feed.entries:
                error = str(feed.bozo_exception) if feed.bozo_exception else "Invalid feed format"
                return {"success": False, "title": None, "item_count": 0, "error": error}

            title = feed.feed.get("title", "").strip() or None
            item_count = len(feed.entries)

            return {
                "success": True,
                "title": title,
                "item_count": item_count,
                "error": None,
            }

        except requests.exceptions.Timeout:
            return {"success": False, "title": None, "item_count": 0, "error": "Connection timed out"}
        except requests.exceptions.ConnectionError:
            return {"success": False, "title": None, "item_count": 0, "error": "Could not connect to server"}
        except requests.exceptions.HTTPError as e:
            return {"success": False, "title": None, "item_count": 0, "error": f"HTTP {e.response.status_code}"}
        except Exception as e:
            return {"success": False, "title": None, "item_count": 0, "error": str(e)}

    @staticmethod
    def name_from_url(url: str) -> str:
        """Extract a human-readable name from a feed URL."""
        parsed = urlparse(url)
        domain = parsed.hostname or ""

        # YouTube channel feeds
        if "youtube.com" in domain:
            channel_id = parsed.path.split("/")[-1] if "/channel/" in url else None
            params = dict(p.split("=") for p in parsed.query.split("&") if "=" in p)
            channel_id = params.get("channel_id", channel_id)
            return f"YouTube ({channel_id[:12]}...)" if channel_id else "YouTube Channel"

        # Strip common prefixes
        name = domain.replace("www.", "").replace("feeds.", "").replace("rss.", "")

        # Capitalize nicely
        parts = name.split(".")
        if len(parts) >= 2:
            name = parts[-2]  # e.g. "openai" from "openai.com"

        return name.replace("-", " ").replace("_", " ").title()
