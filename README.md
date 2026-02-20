# NEURAL_FEED

**AI-Powered News Intelligence Platform**

An intelligent, self-improving news aggregation platform that monitors 400+ sources across RSS feeds, podcasts, and YouTube channels. It uses semantic AI to score, cluster, and summarize content — then delivers personalized daily briefs so you can stay on top of AI/tech news in under 5 minutes.

Live at **[ripin.ai](https://ripin.ai)**

---

## The Story

What started as a simple RSS reader evolved into a full AI news intelligence platform. The core idea: instead of manually scanning dozens of sources every day, let AI do the heavy lifting — fetch everything, score it semantically, cluster related stories, and surface what matters most.

**V1** (Jan 2026) was a basic news agent: scrape RSS feeds, keyword-match, output a markdown digest. It worked, but keyword matching missed nuance and the output was flat.

**V2** (Feb 2026) added podcasts and videos. A transcript extraction pipeline (web scrape, YouTube API, RSS description fallback) let us score and summarize audio/video content alongside articles. Claude-powered summaries replaced raw excerpts.

**V3** (Feb 2026) was the big leap. Semantic embeddings replaced keyword scoring — content is now understood, not just string-matched. Topic clustering groups related stories across sources. A personalization engine learns from your clicks and saves. A chat interface lets you ask questions about recent news using RAG. And a cyberpunk-themed UI ties it all together.

**V3.3** (Feb 2026) made source management fully self-service. All 400+ feeds are now database-driven with a CRUD UI — add, remove, test, and monitor feeds without touching the backend.

---

## What It Does

### Content Aggregation
- **290 news RSS feeds** — major tech outlets, research blogs, Substacks, Reddit communities
- **56 podcast feeds** — AI/ML podcasts with transcript extraction and key learnings
- **87 YouTube channels** — AI research, tutorials, and company channels via RSS
- Automated daily runs via Railway cron

### Semantic Intelligence
- **Vector embeddings** (OpenAI text-embedding-3-small) for all content
- **Semantic scoring** replaces keyword matching — understands meaning, not just words
- **Embedding-based deduplication** catches near-duplicate stories across sources
- **Batch processing** for efficiency with graceful fallback to keyword scoring

### Topic Clustering
- Related articles, podcasts, and videos grouped by theme automatically
- AI-generated cluster names ("OpenAI Model Updates", "RAG Architecture Advances")
- Cross-source synthesis summaries spanning multiple sources on the same topic

### Daily Brief
- AI-written executive summary of the day's key themes
- Top stories curated by relevance score
- Emerging trends and signals section
- Email delivery to subscribers

### Chat (RAG)
- Ask natural language questions about recent news
- Retrieval-augmented generation with source citations
- Time-scoped queries ("What happened with Anthropic this week?")
- Streaming responses

### Source Management
- Database-driven feed management (add/remove/test/toggle from UI)
- Type-separated tabs: News | Podcasts | Videos | Discovery
- Auto-import from config files on first run
- Feed health monitoring — broken feeds flagged after agent runs
- Source quality scoring (0-100) based on match rate, clicks, citations
- Auto-discovery from HackerNews and Reddit with approve/reject workflow

### Personalization
- Cookie-based user profiles (no auth required for preferences)
- Interaction tracking: clicks, saves, skips
- Preference presets ("Deep Tech", "Business News", "Tutorials")
- Adaptive scoring based on learned interests

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.12, FastAPI, SQLAlchemy |
| **Database** | PostgreSQL (production), SQLite (development) |
| **Vector Store** | ChromaDB for semantic embeddings |
| **AI/LLM** | OpenAI (embeddings), Anthropic Claude (summaries, clustering, chat) |
| **Frontend** | Jinja2 templates, Tailwind CSS, HTMX |
| **Deployment** | Railway (auto-deploy on push), Gunicorn + Uvicorn |
| **Scraping** | feedparser, Firecrawl, youtube-transcript-api |
| **Email** | SMTP-based delivery service |
| **Rate Limiting** | slowapi (IP-based) |
| **Clustering** | scikit-learn |

---

## Architecture

```
                    +------------------+
                    |   Railway Cron   |
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |
        +-----v----+  +-----v------+ +----v-------+
        | News Agent|  |Podcast Agent| |Video Agent |
        | (290 RSS) |  | (56 feeds) | | (87 YT)    |
        +-----+----+  +-----+------+ +----+-------+
              |              |              |
              v              v              v
        +--------------------------------------------+
        |          Semantic Scoring Engine            |
        |  (OpenAI embeddings + cosine similarity)   |
        +---------------------+----------------------+
                              |
                              v
        +--------------------------------------------+
        |              PostgreSQL                     |
        |  Digests | Items | Clusters | FeedSources  |
        |  Users | Interactions | SourceQuality      |
        +---------------------+----------------------+
                              |
              +---------------+---------------+
              |               |               |
        +-----v----+   +-----v-----+   +-----v------+
        | Daily Brief|  | Web UI    |   | Chat (RAG) |
        | (email)    |  | (FastAPI) |   | (Claude)   |
        +------------+  +-----------+   +------------+
```

---

## Project Structure

```
ai-news-agent/
  agent.py              # News agent — fetch, score, deduplicate, digest
  podcast_agent.py      # Podcast agent — transcripts, summaries
  video_agent.py        # Video agent — YouTube feeds, transcripts
  summarizer.py         # Claude-powered article/podcast summarization
  sources.txt           # 290 news RSS feed URLs
  podcasts.txt          # 56 podcast RSS feed URLs
  videos.txt            # 87 YouTube channel feed URLs
  config.py             # App settings and feature flags
  services/
    embeddings.py       # OpenAI embedding generation
    vector_store.py     # ChromaDB vector storage
    semantic_scorer.py  # Semantic relevance scoring
    topic_clustering.py # Auto-clustering with AI labels
    chat_rag.py         # RAG-powered chat service
    daily_brief.py      # Executive summary generation
    feed_validator.py   # Feed URL validation/testing
    source_discovery.py # HN/Reddit source discovery
    source_scoring.py   # Source quality evaluation
    personalization.py  # User preference learning
    transcript_service.py # Podcast/video transcript extraction
    email_delivery.py   # Email subscriber delivery
    firecrawl_service.py # Web scraping service
  web/
    app.py              # FastAPI application entry point
    database.py         # SQLAlchemy setup (Postgres/SQLite)
    models.py           # 10 database models
    routes/             # API and page routes
    templates/          # Jinja2 templates (cyberpunk theme)
    static/             # CSS, JS assets
```

---

## Running Locally

```bash
# Clone and setup
git clone https://github.com/jungleboyz/ai-news-agent.git
cd ai-news-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-prod.txt

# Configure environment
cp .env.example .env  # Add your API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY)

# Run the web app (uses SQLite by default)
uvicorn web.app:app --reload --port 8000

# Run the news agent manually
python agent.py
```

---

## The Value

**For individuals:** Stay informed on AI/tech without doom-scrolling. 400+ sources distilled into a 5-minute daily brief with semantic relevance scoring tuned to your interests.

**For teams:** Shared intelligence platform. Everyone sees the same curated feed. Source quality scoring surfaces signal and buries noise. Topic clustering shows the full picture across sources.

**For the curious:** Ask questions about recent news via chat. Get answers with citations. Follow emerging trends before they go mainstream.

---

## License

Private repository. All rights reserved.
