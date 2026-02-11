import os
import openai
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Try to import Anthropic
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


def get_anthropic_client():
    """Get Anthropic client if API key is configured."""
    if not ANTHROPIC_AVAILABLE:
        return None
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)

# Load API key from environment variable
def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-openai-api-key":
        return None
    return openai.OpenAI(api_key=api_key)

def fetch_article_text(url):
    try:
        html = requests.get(url, timeout=5).text
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join([p.text for p in paragraphs])
        return text[:3000]  # Trim to avoid token limits
    except Exception as e:
        return None

def generate_fallback_summary(url, title=""):
    """Generate a simple fallback summary when API is unavailable."""
    if title:
        return f"This article about '{title}' may be relevant to your interests. Click the link to read more."
    return "This article may be relevant to your interests. Click the link to read more."

def summarize_article(url, title=""):
    """
    Summarize an article using OpenAI API.
    Returns None if API key is not configured (caller should use fallback).
    Returns error string if API call fails.
    Returns summary string on success.
    """
    client = get_client()
    if client is None:
        # Return None to indicate fallback should be used
        return None
    
    content = fetch_article_text(url)
    if content is None:
        # Failed to fetch article text
        return None

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.5,
            messages=[
                {"role": "system", "content": "You are an AI summarizer for a daily tech news digest."},
                {"role": "user", "content": f"Summarize this article in 2-3 short bullet points:\n\n{content}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Return None on API errors so caller can use fallback
        return None


def summarize_text(text, title=""):
    """
    Summarize text directly using OpenAI API.
    Used for podcast transcripts and other text content.
    Returns None if API key is not configured (caller should use fallback).
    Returns summary string on success.
    """
    client = get_client()
    if client is None:
        # Return None to indicate fallback should be used
        return None

    if not text or not text.strip():
        return None

    # Truncate text to avoid token limits (use first 4000 chars)
    content = text[:4000] if len(text) > 4000 else text

    try:
        prompt = "Summarize this content in 2-3 short bullet points for an AI news digest."
        if title:
            prompt = f"Summarize this podcast segment about '{title}' in 2-3 short bullet points for an AI news digest."

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.5,
            messages=[
                {"role": "system", "content": "You are an AI summarizer for a daily tech news digest."},
                {"role": "user", "content": f"{prompt}\n\n{content}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Return None on API errors so caller can use fallback
        return None


def summarize_podcast(transcript, title="", show_name=""):
    """
    Summarize a podcast transcript with 5 key learnings.
    Uses Anthropic Claude (preferred) or OpenAI as fallback.
    Returns None if no API is available (caller should use fallback).
    Returns formatted summary string on success.
    """
    if not transcript or not transcript.strip():
        return None

    # Use more of the transcript for podcasts
    content = transcript[:12000] if len(transcript) > 12000 else transcript
    context = f"Podcast: {show_name}\nEpisode: {title}\n\n" if show_name else f"Episode: {title}\n\n" if title else ""

    prompt = f"""{context}Transcript:
{content}

---

Provide a summary with exactly 5 key learnings from this podcast. Format as:

**Key Learnings:**

1. **[Topic]:** [Concise insight or takeaway in 1-2 sentences]

2. **[Topic]:** [Concise insight or takeaway in 1-2 sentences]

3. **[Topic]:** [Concise insight or takeaway in 1-2 sentences]

4. **[Topic]:** [Concise insight or takeaway in 1-2 sentences]

5. **[Topic]:** [Concise insight or takeaway in 1-2 sentences]

Focus on actionable insights, surprising facts, or unique perspectives shared in the discussion."""

    # Try Anthropic Claude first (preferred)
    anthropic_client = get_anthropic_client()
    if anthropic_client:
        try:
            response = anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                system="You are an expert podcast summarizer. Extract the most valuable insights and actionable takeaways from podcast transcripts. Focus on unique perspectives, surprising facts, and practical advice that listeners would find most valuable."
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"Warning: Anthropic API failed: {e}")

    # Fall back to OpenAI
    client = get_client()
    if client:
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0.5,
                messages=[
                    {"role": "system", "content": "You are an expert podcast summarizer. Extract the most valuable insights and actionable takeaways from podcast transcripts. Focus on unique perspectives, surprising facts, and practical advice that listeners would find most valuable."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Warning: OpenAI API failed: {e}")

    return None


def generate_fallback_podcast_summary(title="", show_name=""):
    """Generate a simple fallback summary for podcasts when API is unavailable."""
    if show_name and title:
        return f"Listen to '{title}' from {show_name} for insights on this topic."
    elif title:
        return f"Listen to '{title}' for insights on this topic."
    return "Listen to this episode for insights on this topic."
