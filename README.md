# NEURAL_FEED

**AI-Powered News Intelligence Platform**

A production-grade, multi-agent AI system that monitors 400+ sources across RSS feeds, podcasts, YouTube channels, and web listings. It uses vector embeddings and cosine similarity to score content against user interests, clusters related stories using KMeans, generates executive summaries via LLM, and delivers personalized daily briefs — replacing hours of manual scanning with a 5-minute AI-curated digest.

Live at **[ripin.ai](https://ripin.ai)**

---

## Executive Summary

NEURAL_FEED is an end-to-end AI news intelligence product built from scratch. It began as a single Python script that parsed RSS feeds with keyword matching, and evolved through deliberate iteration into a multi-agent orchestration system with semantic scoring, RAG-powered chat, automated source discovery, and a production web application serving daily digests.

The project demonstrates practical application of core AI engineering concepts: building and managing autonomous agents, implementing RAG pipelines with vector databases, designing cascading provider architectures for resilience, applying semantic similarity at scale, and securing a production AI application.

**4 autonomous agents** run concurrently to compile each daily digest. **18 service modules** handle everything from embedding generation to topic clustering. **10 database models** persist digests, user interactions, source quality metrics, and discovered sources. The system processes ~1,400 items per run, deduplicates across 5 layers, and produces a scored, clustered, summarized digest — fully automated on a daily cron schedule.

---

## From Single Agent to Multi-Agent System

### V1: The First Agent (Jan 2026)

The project started with a single agent — `agent.py` — that did three things: fetch RSS feeds via `feedparser`, score each item against a hardcoded keyword list, and write a markdown file. The scoring was naive: count how many keywords from a static list appeared in the title, multiply by 2 per match. No understanding of meaning, no deduplication, no summaries. The output was a flat ranked list.

This was the foundation for understanding what an AI agent is at its core: a program that autonomously observes (fetches data), reasons (scores relevance), and acts (produces output) without human intervention in the loop.

### V2: Multi-Agent Architecture (Feb 2026)

V1 only covered news articles. Podcasts and YouTube videos contain some of the most valuable AI commentary, but they require fundamentally different processing — transcript extraction, audio-aware scoring, different summarization prompts. Rather than overloading the news agent with conditional logic, the system was decomposed into specialised agents:

- **Podcast Agent** (`podcast_agent.py`) — parses podcast RSS feeds, extracts transcripts through a 3-tier cascade (web scraping via Firecrawl/BeautifulSoup, YouTube transcript API, RSS description fallback), scores episodes, and generates "5 key learnings" summaries via Claude.
- **Video Agent** (`video_agent.py`) — resolves YouTube `@handle` URLs to channel IDs, parses YouTube RSS feeds, fetches transcripts via the `youtube-transcript-api` library with Webshare proxy support for cloud environments, scores videos against user interests, and summarises.
- **Web Scraper Agent** (`web_scraper_agent.py`) — handles sites that don't publish RSS feeds. Uses Firecrawl to scrape listing pages, extracts article links via regex pattern matching on markdown (`[Title](URL)` patterns), filters to same-domain links, skips navigation/meta URLs, scores, and summarises.

The orchestrator (`agent.py`) evolved from a standalone agent into a dual-role component: it still processes its own 290+ news RSS feeds, but now also launches the other 3 agents concurrently using `ThreadPoolExecutor(max_workers=3)`, collects their results, cross-deduplicates by link, and merges everything into a unified scored digest.

### V3: Semantic Intelligence (Feb 2026)

Keyword matching has a fundamental limitation — it matches strings, not meaning. An article titled "Anthropic raises Series D" would score 2 points for containing "anthropic", but an article titled "New approach to reasoning in language models" would score 0 despite being highly relevant. The system needed to understand semantic meaning.

This led to the integration of vector embeddings and cosine similarity scoring, replacing keyword matching entirely. The semantic scoring pipeline, topic clustering, personalization engine, RAG chat, and source discovery were all added in V3.

---

## How the Daily Digest Pipeline Works

Each daily run is triggered by APScheduler on a UTC cron schedule (configurable via environment variables). The orchestrator agent launches the full pipeline:

### Step 1: Concurrent Data Collection (4 Agents)

```
             +------------------+
             |   APScheduler    |
             +--------+---------+
                      |
       +--------------+--------------+--------------+
       |              |              |              |
  +----v-----+  +-----v------+ +----v-------+ +----v--------+
  |News Agent|  |Podcast Agent| |Video Agent | |Web Scraper  |
  | 290 RSS  |  | 56 feeds   | | 87 YouTube | | 84 listings |
  +----+-----+  +-----+------+ +----+-------+ +----+--------+
       |              |              |              |
       v              v              v              v
  +--------------------------------------------------+
  |         Merge + Cross-Deduplicate by Link         |
  +--------------------------------------------------+
```

The 3 sub-agents run in parallel via `ThreadPoolExecutor` with a 600-second timeout per agent. Each agent independently:

1. **Loads its feed URLs** from PostgreSQL (`FeedSource` table, filtered by `source_type` and `status='active'`), falling back to flat text files (`sources.txt`, `podcasts.txt`, `videos.txt`, `web_sources.txt`) if the database is unavailable.
2. **Fetches feeds in parallel** using `ThreadPoolExecutor(max_workers=10)` via the shared `fetch_feeds_parallel()` service.
3. **Deduplicates against previously seen items** using SHA-256 hashes of `title|link`, stored in JSON cache files and cross-referenced against the database for container-restart resilience.
4. **Scores items** (detailed in the next section).
5. **Generates summaries** via Claude API.
6. **Updates feed source statuses** in the database (sets `last_fetched` to UTC timestamp, clears error state).

The news agent additionally fetches items sequentially (since `feedparser.parse()` is fast), then batch-scores all ~1,400 items at once for embedding efficiency.

### Step 2: Semantic Scoring and Curation

Scoring is the core intelligence of the system. It determines what surfaces in the digest and in what order.

**Semantic Scoring Pipeline:**

1. **Embedding Generation** — Each item's text (`title + description + transcript` where available) is converted to a 1,536-dimensional vector using OpenAI's `text-embedding-3-small` model. The `EmbeddingService` implements a cascading provider architecture: it tries OpenAI first, and if that fails (quota exceeded, rate limited), falls back to Jina AI's `jina-embeddings-v3` (1,024 dimensions). Batch embedding is used for efficiency — up to 2,048 texts per API call.

2. **Interest Embedding** — The `SemanticScorer` maintains a pre-computed "interest vector" by averaging embeddings of 7 user interest descriptions (e.g., "generative AI and large language models", "AI coding assistants like Cursor, Copilot, and Aider", "enterprise AI adoption and automation"). This is computed once and cached.

3. **Cosine Similarity** — Each item's embedding is compared to the interest embedding using cosine similarity, producing a float score between 0.0 and 1.0. This is then scaled to an integer 0-10 for display. A score of 0.3+ is considered relevant.

4. **Quota-Aware Fallback** — If the embedding API returns a 429 (rate limit) or quota error, a global `_quota_exceeded` flag is set across all agents (shared via `service_registry.py`). All subsequent scoring in that run falls back to keyword matching: 40+ AI-related keywords are checked against title (3 points per match) and description/transcript (1 point per match).

**For podcast and video agents**, scoring is done per-item using `score_single_with_embedding()`, which returns the integer score, the raw float score, and the embedding vector — because these agents need the embedding back to store in ChromaDB for later deduplication and search.

**For the news agent**, scoring is done in batch using `score_items_batch()`, which generates all embeddings in a single API call and scores them against the interest vector in a loop. This is significantly more efficient for 1,400+ items.

### Step 3: Multi-Layer Deduplication

Duplicate detection operates at 5 layers to ensure the digest contains only unique content:

1. **ID-based (within agent)** — SHA-256 hash of `title|link` checked against the seen cache before scoring.
2. **Within-batch (within agent)** — Items with duplicate IDs within a single run are dropped.
3. **Embedding similarity (within agent)** — Items whose embedding has >0.95 cosine similarity to an existing item in ChromaDB are flagged as duplicates. This catches paraphrased/syndicated versions of the same story.
4. **Cross-source link matching (orchestrator)** — After all 4 agents return, the orchestrator deduplicates by exact URL match across news, podcasts, videos, and web articles.
5. **Historical database check** — Recent links from the last 14 days of digests are loaded from PostgreSQL and used to prevent items from reappearing across days.

### Step 4: Summarization

Each item is summarised using Anthropic's Claude API (claude-sonnet model):

- **News articles**: Firecrawl pre-fetches article content in batch (up to 50 per run, capped to control API costs). The markdown content is passed to Claude with a prompt asking for a "Why this matters" summary. If the RSS feed already provides a summary >100 characters, Firecrawl pre-fetch is skipped for that item.
- **Podcast episodes**: The transcript (from the 3-tier cascade: web scrape → YouTube → RSS description) is passed to Claude with a prompt requesting "5 key learnings" format.
- **Videos**: Same as podcasts — transcript-based summarisation.
- **Web articles**: Same as news — Firecrawl content + Claude summarisation.

Summaries are cached in JSON files (`out/summaries.json`, `out/podcast_summaries.json`, etc.) so re-runs don't re-summarise already-processed items.

If the Claude API is unavailable, a fallback summary is generated from the RSS description or title.

### Step 5: Digest Assembly and Delivery

The orchestrator merges all items into a single list, sorted by score descending. It then:

1. **Writes markdown and HTML digests** to disk (`out/digest-YYYY-MM-DD.md/html`), with Tailwind CSS styling for the HTML version.
2. **Saves to PostgreSQL** — creates a `Digest` record with metadata (source counts, date) and individual `Item` records for each article/episode/video, including scores, summaries, and cluster assignments.
3. **Runs topic clustering** (see below).
4. **Sends email** to subscribers via SMTP (if configured).

### Step 6: Topic Clustering

After the digest is saved, the `TopicClusterer` groups related items:

1. **Embedding retrieval** — tries to load embeddings from ChromaDB first. If items are missing (e.g., keyword-scored items without embeddings), it generates fresh embeddings via Jina/OpenAI.
2. **Optimal K selection** — iterates K from 2 to 10 (or `item_count / 5`, whichever is smaller), runs KMeans for each, and selects the K with the highest silhouette score.
3. **KMeans clustering** — scikit-learn `KMeans` groups items by embedding proximity. Each item gets a cluster assignment and a confidence score (cosine similarity to cluster centroid).
4. **Label generation** — Claude (haiku model for cost efficiency) generates 3-5 word topic labels from the top 10 item titles in each cluster (e.g., "AI Agents and Automation", "Enterprise LLM Adoption").
5. **Summary generation** — Claude synthesises a 2-3 sentence cross-source summary per cluster.
6. **Persistence** — `TopicCluster` records are saved to PostgreSQL, and each `Item` record is updated with its `cluster_id`, `cluster_label`, and `cluster_confidence`.

---

## RAG-Powered Chat

The chat interface lets users ask natural language questions about recent AI news and receive answers grounded in actual digest content — a practical implementation of Retrieval-Augmented Generation.

### How the RAG Pipeline Works

1. **Query Processing** — The user's question is received via the `/api/chat/stream` endpoint (Server-Sent Events for streaming) or `/api/chat` for single responses.

2. **Keyword Extraction** — Important terms are extracted from the query to improve retrieval precision.

3. **Hybrid Retrieval** — The system searches for relevant context using two strategies:
   - **Vector search (primary)**: The query is embedded and searched against ChromaDB using cosine similarity. Up to 8 relevant items are retrieved across news, podcast, and video collections.
   - **Database keyword search (fallback)**: If the vector store returns insufficient results, the system falls back to PostgreSQL keyword search across recent `Item` records.

4. **Context Formatting** — Retrieved items are formatted with title, source, date, link, and summary — providing the LLM with structured, citable context.

5. **LLM Generation** — The formatted context and user message are sent to Claude Sonnet with a system prompt that instructs it to answer based on the retrieved context, cite sources, and acknowledge when information isn't available.

6. **Conversation Memory** — The service maintains per-user conversation history (up to 50 messages per `conversation_id`) in memory, enabling multi-turn conversations.

7. **Suggested Questions** — The system generates dynamic question suggestions based on recent topic clusters, helping users explore the data.

This implementation demonstrates the core RAG pattern: retrieve relevant documents from a vector store, augment the LLM prompt with that context, and generate grounded responses with citations — avoiding hallucination by constraining the model to retrieved facts.

---

## Personalization Engine

The personalization system learns user preferences without requiring authentication:

### How It Works

1. **User Identification** — Cookie-based UUID user IDs. No login required — preferences are tied to browser sessions.

2. **Interaction Tracking** — Every user action is recorded with weighted scores:
   - Click: 1.0 (mild interest)
   - Save: 3.0 (strong interest)
   - Skip: -0.5 (mild disinterest)
   - Hide: -2.0 (strong disinterest)

3. **Preference Embedding Computation** — The system computes a weighted average of the embeddings of items the user has interacted with: `preference_vector = normalize(sum(embedding_i * weight_i))`. Items the user saved contribute 3x the influence of items they clicked. Hidden items pull the vector away from that content. The result is a single normalised vector representing the user's interests.

4. **Blended Scoring** — When personalisation is active, item scores are a weighted blend: `final_score = 0.6 * semantic_score + 0.4 * preference_score`, where `preference_score` is the cosine similarity between the item's embedding and the user's preference vector.

5. **Presets** — Users can create named preference presets (e.g., "AI Research", "AI in Banking", "Tutorials") with specific interest descriptions. Each preset has its own cached embedding vector computed from the interest descriptions. Activating a preset switches the preference vector used for scoring.

---

## Source Discovery and Enrichment

Rather than manually curating feed lists, the system automatically discovers new AI-relevant sources:

### Discovery Pipeline

1. **HackerNews Scraper** — Uses the official Firebase API (no authentication required) to fetch top and new stories. Each item is scored for AI relevance using a 40+ keyword regex pattern. Items above a minimum score threshold (default 10 HN points) with AI relevance are surfaced as potential sources. The domain is extracted for feed URL discovery.

2. **Reddit Scraper** — Monitors 14 AI-focused subreddits (`r/MachineLearning`, `r/OpenAI`, `r/Anthropic`, `r/LocalLLaMA`, etc.) via Reddit's JSON API. Filters by score and AI keyword relevance. Distinguishes self-posts from external links.

3. **Firecrawl Web Search** — Searches the web using rotating AI-related queries (10 different search terms) to discover sources not captured by HN or Reddit.

### Enrichment Flow

Discovered sources are stored as `DiscoveredSource` records with status `pending`. An admin can review them in the Sources UI and approve/reject. Approved sources are converted to `FeedSource` records and immediately included in the next digest run.

### Source Quality Scoring

Every active source is continuously evaluated on a 0-100 quality score:

| Factor | Weight | What It Measures |
|--------|--------|-----------------|
| Match Rate | 25% | % of items that match AI keywords |
| Average Score | 20% | Mean semantic relevance of matched items |
| Engagement | 25% | User clicks + saves (saves weighted 3x), log-scaled |
| Citations | 15% | Times the domain appears in HN/Reddit, log-scaled |
| Recency | 15% | Days since last seen (decays 2 points/day after 30 days) |

Low-quality sources (score below threshold) are flagged for review. This creates a self-improving source list — the system naturally prunes noisy sources and surfaces high-signal ones.

---

## System Architecture

```
                         +-------------------+
                         |    APScheduler    |
                         |   (UTC cron)      |
                         +---------+---------+
                                   |
          +------------------------+------------------------+
          |                        |                        |
+---------v----------+  +---------v----------+  +----------v---------+
| Podcast Agent      |  | Video Agent        |  | Web Scraper Agent  |
| 56 podcast feeds   |  | 87 YouTube channels|  | 84 listing pages   |
| Transcript cascade |  | Parallel transcripts|  | Firecrawl + regex  |
+---------+----------+  +---------+----------+  +----------+---------+
          |                        |                        |
          +------------------------+------------------------+
                                   |
                    +--------------v--------------+
                    |     News Agent / Orchestrator|
                    |     290 RSS feeds            |
                    |     Batch scoring (1400+)    |
                    |     Cross-source dedup       |
                    +--------------+--------------+
                                   |
          +------------------------+------------------------+
          |                        |                        |
+---------v----------+  +---------v----------+  +----------v---------+
| Scoring Service    |  | Cache Service      |  | Feed Service       |
| Semantic + keyword |  | JSON load/save     |  | Parallel fetch     |
| Batch + single     |  | Set-based seen     |  | Status updates     |
+---------+----------+  +--------------------+  +--------------------+
          |
+---------v----------+
| Service Registry   |         +--------------------+
| EmbeddingService   +-------->| OpenAI API         |
| VectorStore        |         | text-embedding-3-sm|
| SemanticScorer     +-------->| Jina AI (fallback) |
| Quota management   |         +--------------------+
+---------+----------+
          |
+---------v---------------------------------------------------+
|                        PostgreSQL                            |
| Digest | Item | TopicCluster | FeedSource | DiscoveredSource |
| UserProfile | Interaction | PreferencePreset | SourceQuality |
| EmailSubscriber                                              |
+-----+------------------+-------------------+----------------+
      |                  |                   |
+-----v------+    +------v-------+    +------v--------+
| Daily Brief|    | Web UI       |    | Chat (RAG)    |
| Claude gen |    | FastAPI      |    | Vector search |
| DB cached  |    | Tailwind CSS |    | Claude stream |
| Email SMTP |    | HTMX         |    | SSE responses |
+------------+    +--------------+    +---------------+
```

---

## Shared Service Layer

The 4 agents share a common service layer that eliminates code duplication and ensures consistency:

| Service | Purpose |
|---------|---------|
| `service_registry.py` | Singleton management for `EmbeddingService`, `VectorStore`, `SemanticScorer`. Shared quota-exceeded flag prevents repeated API calls after 429 errors. |
| `scoring_service.py` | Unified scoring: `score_semantic()`, `score_keywords()`, `score_items_batch()`, `score_single_with_embedding()`. All agents use the same scoring logic. |
| `cache_service.py` | Generic JSON persistence: `load_json()`, `save_json()`, `load_set()`, `save_set()`. Replaces 12 duplicated load/save functions. |
| `feed_service.py` | `fetch_feeds_parallel()` with ThreadPoolExecutor, `update_feed_statuses()` for DB status updates. |
| `youtube_service.py` | Single YouTube Transcript API instance with Webshare proxy. Configured once, shared by Video Agent and Transcript Service. |
| `embeddings.py` | Cascading embedding provider: OpenAI (1,536d) → Jina (1,024d). Retry logic with exponential backoff. Token counting and text truncation. |
| `vector_store.py` | ChromaDB wrapper with 3 collections (news, podcast, video). Cosine similarity metric. Batch upsert, similarity search, duplicate detection. |
| `transcript_service.py` | 3-tier transcript cascade: Firecrawl web scrape → YouTube API → RSS description. No audio download, no Whisper — pure text extraction. |
| `firecrawl_service.py` | Firecrawl SDK wrapper with timeout protection (30s single, 120s batch), quota tracking, and markdown extraction. |

---

## Security

The application implements defense-in-depth security appropriate for a production web service:

### Authentication and Authorisation
- **Session-based auth** with HMAC-SHA256 signed cookies (30-day expiry)
- **Protected routes**: `/sources`, `/api/feeds/`, `/api/sources/` require authentication
- **Public routes**: digest viewing, chat, search are accessible without login
- **Cron endpoint protection**: `POST /cron/run-digest` requires a `Bearer` token matching the `CRON_SECRET` environment variable

### Rate Limiting
- IP-based rate limiting via SlowAPI (extracts real IP from `X-Forwarded-For`)
- Login: 10 requests/minute
- Chat: 20 requests/minute
- Email subscribe: 10 requests/hour

### Security Headers (applied via middleware)
- `Content-Security-Policy`: restricts script/style/image sources
- `X-Content-Type-Options: nosniff` (prevents MIME sniffing)
- `X-Frame-Options: DENY` (prevents clickjacking)
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy`: disables camera, microphone, geolocation
- `Strict-Transport-Security` (HSTS) in production

### API Key Management
- All secrets stored as environment variables, never in code
- Cascading provider pattern means the system degrades gracefully when keys expire or quotas are exceeded, rather than failing completely

---

## Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| **Runtime** | Python 3.12 | Core language |
| **Web Framework** | FastAPI | Async API + SSE streaming |
| **ORM** | SQLAlchemy 2.0 | Database abstraction |
| **Database** | PostgreSQL (prod), SQLite (dev) | Persistent storage |
| **Vector Store** | ChromaDB | Embedding storage and similarity search |
| **Embeddings** | OpenAI text-embedding-3-small, Jina v3 (fallback) | Text-to-vector conversion |
| **LLM** | Anthropic Claude Sonnet (summaries, chat, clustering, briefs) | Text generation |
| **Frontend** | Jinja2 + Tailwind CSS + HTMX | Server-rendered UI |
| **Deployment** | Railway (auto-deploy on push) | CI/CD and hosting |
| **Process** | Gunicorn + Uvicorn workers | Production ASGI server |
| **Scheduling** | APScheduler | Cron-based digest triggers |
| **Scraping** | feedparser, Firecrawl SDK, youtube-transcript-api | Content extraction |
| **Clustering** | scikit-learn KMeans | Topic grouping |
| **Rate Limiting** | SlowAPI | Request throttling |
| **Email** | SMTP (Gmail) | Digest delivery |
| **Proxy** | Webshare | YouTube API access from cloud |

---

## Database Schema (10 Models)

| Model | Records | Purpose |
|-------|---------|---------|
| `Digest` | 1 per day | Date, source counts, cached brief JSON, file paths |
| `Item` | ~300-400 per digest | Title, link, score, semantic_score, summary, cluster assignment, embedding reference |
| `TopicCluster` | 2-10 per digest | Label, summary, item count, average score |
| `FeedSource` | 400+ | URL, type (news/podcast/video/web), status, last_fetched, error tracking |
| `UserProfile` | Per browser | Cookie-based ID, cached preference embedding |
| `Interaction` | Per click/save/skip | User-item action with weighted scoring |
| `PreferencePreset` | Per user | Named interest lists with cached embeddings |
| `SourceQuality` | Per domain | Multi-factor quality score (0-100) |
| `DiscoveredSource` | From HN/Reddit | Pending/approved/rejected sources |
| `EmailSubscriber` | Opt-in | Email delivery list |

---

## AI Engineering Skills Demonstrated

This project was built as a hands-on learning exercise in applied AI engineering. Key competencies developed:

### Agent Design and Orchestration
- Designed 4 autonomous agents with distinct responsibilities and data processing pipelines
- Implemented concurrent agent execution with timeout management and error isolation
- Built a shared service layer to eliminate duplication and ensure consistency across agents
- Managed agent lifecycle: data collection, scoring, deduplication, summarisation, persistence

### Retrieval-Augmented Generation (RAG)
- Built a complete RAG pipeline: document embedding → vector storage → semantic retrieval → context augmentation → LLM generation
- Implemented hybrid retrieval (vector search + keyword fallback) for robustness
- Designed prompt templates that ground LLM responses in retrieved context with source citations
- Added conversation memory for multi-turn RAG interactions

### Embedding and Vector Operations
- Implemented cascading embedding providers with automatic failover
- Batch embedding for throughput efficiency (1,400+ items per run)
- Cosine similarity scoring against pre-computed interest vectors
- Embedding-based deduplication (>0.95 similarity threshold)
- ChromaDB vector store management across 3 collections

### Machine Learning
- KMeans clustering with automated K selection via silhouette score optimisation
- User preference learning through weighted interaction embedding aggregation
- Multi-factor source quality scoring with normalised component weights

### Production System Design
- Graceful degradation: cascading providers, quota-aware fallback, timeout protection
- 5-layer deduplication strategy (hash, batch, embedding, cross-source, historical)
- Database-backed caching for AI-generated content (briefs, summaries, transcripts)
- Session-based auth, rate limiting, CSP headers, HSTS

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

## License

Private repository. All rights reserved.
