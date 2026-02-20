# AI News Agent V3 - Requirements

## Vision
An intelligent, self-improving AI news platform that learns your preferences, discovers new sources automatically, clusters content by themes, and delivers personalized insights through multiple interfaces.

---

## Core Features

### 1. Semantic Intelligence Engine

| Feature | Description |
|---------|-------------|
| **Embeddings** | Convert all content (titles, summaries, transcripts) to vector embeddings |
| **Vector Database** | Store embeddings in ChromaDB/Pinecone for similarity search |
| **Semantic Scoring** | Replace keyword matching with cosine similarity to user interests |
| **Deduplication** | Detect near-duplicate stories across sources using embedding similarity |

### 2. Topic Clustering

| Feature | Description |
|---------|-------------|
| **Auto-clustering** | Group related articles/videos/podcasts by theme |
| **Cluster Naming** | AI-generated names for each cluster ("OpenAI Drama", "RAG Advances") |
| **Cross-source Synthesis** | Single summary spanning multiple sources on same topic |
| **Trend Detection** | Identify emerging topics gaining momentum |

### 3. Personalized Learning

| Feature | Description |
|---------|-------------|
| **Interaction Tracking** | Log clicks, time spent, saves, skips |
| **Preference Profile** | Build embedding-based user interest model |
| **Adaptive Scoring** | Weight content based on learned preferences |
| **Feedback Loop** | Thumbs up/down to refine recommendations |
| **Preference Presets** | Define named views ("Deep Tech", "Business News", "Tutorials") |

### 4. Source Auto-Discovery

| Feature | Description |
|---------|-------------|
| **Community Monitoring** | Scan HackerNews, Reddit, Twitter for trending AI sources |
| **Citation Analysis** | Find sources frequently referenced by existing content |
| **Quality Scoring** | Evaluate new sources before adding (relevance, freshness, signal-to-noise) |
| **Auto-suggestion** | Weekly recommendations of new sources to add |
| **Source Health** | Monitor feed quality, flag dead/low-value sources |

---

## Delivery Interfaces

### 5. Web Dashboard (Enhanced)

| Feature | Description |
|---------|-------------|
| **Topic View** | Browse by clusters instead of chronological list |
| **Trend Charts** | Visualize topic momentum over time |
| **Source Analytics** | See which sources provide most value |
| **Preference Editor** | Define and manage custom views |
| **Saved Items** | Bookmark for later, create collections |

### 6. Refreshed UI Design

| Feature | Description |
|---------|-------------|
| **Modern AI Aesthetic** | Futuristic, tech-forward visual design |
| **Manga/Anime Theme** | Bold typography, dynamic layouts, anime-inspired accents |
| **Neon Accents** | Cyberpunk color palette (electric blue, hot pink, purple gradients) |
| **Animated Elements** | Subtle glows, hover effects, loading animations |
| **Card-based Layout** | Clean, scannable content cards with visual hierarchy |
| **Dark Mode First** | Deep blacks with vibrant accent colors |
| **Custom Illustrations** | AI/robot mascot character, manga-style icons |
| **Micro-interactions** | Satisfying clicks, swipes, and transitions |

### 7. Daily Brief

| Feature | Description |
|---------|-------------|
| **Executive Summary** | AI-written 3-5 paragraph overview of key themes |
| **Top Stories** | Curated highlights based on preferences |
| **Emerging Signals** | "Things to watch" section for new trends |
| **Customizable Format** | Choose length, depth, focus areas |
| **Multi-channel** | Email, Slack, Discord delivery |

### 8. Chat Interface

| Feature | Description |
|---------|-------------|
| **RAG-powered Q&A** | Ask questions about recent news |
| **Source Citations** | Answers include links to original content |
| **Follow-up** | Conversational context for deep dives |
| **Time Scoping** | "What happened with OpenAI this week?" |
| **Comparison** | "Compare coverage of X across sources" |

---

## Deployment

| Requirement | Description |
|-------------|-------------|
| **Domain** | Live at **ripin.ai** |
| **SSL** | HTTPS with auto-renewing certificates |
| **CDN** | CloudFlare or similar for global performance |
| **Hosting** | Cloud provider (AWS/GCP/Vercel/Railway) |
| **CI/CD** | Auto-deploy on push to main branch |
| **Monitoring** | Uptime monitoring, error tracking, analytics |
| **Backups** | Automated database backups |

---

## Technical Architecture

### Data Layer
- **PostgreSQL** - Structured data (digests, items, users, preferences)
- **ChromaDB/Pinecone** - Vector embeddings for semantic search
- **Redis** - Caching, session management

### AI Layer
- **Embeddings** - OpenAI text-embedding-3-small or local alternative
- **LLM** - Claude for summaries, chat, clustering
- **Local Models** - Optional Ollama for privacy-sensitive operations

### Backend
- **FastAPI** - REST + WebSocket APIs
- **Celery** - Background jobs (scraping, embedding, clustering)
- **Scheduler** - Automated runs (APScheduler or cron)

### Frontend
- **React/Next.js** or **HTMX** (keep it simple)
- **Chat UI** - Streaming responses
- **Charts** - D3.js or Chart.js for trends
- **Styling** - Tailwind CSS with custom manga/cyberpunk theme
- **Animations** - Framer Motion for smooth transitions
- **Icons** - Custom manga-style icon set
- **Fonts** - Mix of clean sans-serif + stylized display fonts

---

## Implementation Phases

### Phase 1: Embeddings & Semantic Search ✅
- [x] Add vector database (ChromaDB)
- [x] Generate embeddings for all content (OpenAI text-embedding-3-small)
- [x] Replace keyword scoring with semantic similarity (with keyword fallback)
- [x] Add semantic search to web UI
- [x] Batch embedding generation for efficiency
- [x] Embedding-based deduplication

### Phase 2: Topic Clustering ✅
- [x] Implement clustering algorithm (scikit-learn)
- [x] AI-generated cluster names (Anthropic Claude)
- [x] Cross-source synthesis summaries
- [x] Topic view in dashboard

### Phase 3: Personalization ✅
- [x] Interaction tracking (click, save, skip, hide)
- [x] User preference model (cookie-based profiles)
- [x] Adaptive scoring (preference embeddings)
- [x] Preference presets UI

### Phase 4: Source Discovery ✅
- [x] Community monitoring scrapers (HackerNews, Reddit)
- [x] Source quality evaluation (quality_score 0-100)
- [x] Auto-suggestion system (discovered sources)
- [x] Source health dashboard

### Phase 4b: Sources Manager ✅ (2026-02-20)
- [x] `FeedSource` model — database-driven feed management
- [x] CRUD API endpoints (list, create, update, delete, test, import)
- [x] Feed validation service (`services/feed_validator.py`)
- [x] Type-separated UI tabs: News (290) | Podcasts (56) | Videos (87) | Discovery
- [x] Auto-import from .txt files on first startup (396 feeds seeded)
- [x] Agent source loading from DB with .txt fallback (backwards compatible)
- [x] Post-run status updates (last_fetched, error flagging)
- [x] Add/remove/toggle/test feeds from UI — no backend access needed

### Phase 5: Daily Brief & Chat ✅
- [x] Executive summary generation (AI-written brief)
- [x] Multi-channel delivery (email via EmailSubscriber)
- [x] RAG pipeline for chat (chat_rag.py)
- [x] Chat UI implementation

### Phase 6: UI Redesign (Cyberpunk Theme) ✅
- [x] Design system and component library (Tailwind + custom CSS)
- [x] Cyberpunk color palette and typography (Orbitron + Rajdhani, neon accents)
- [x] Custom AI mascot (SVG robot in nav)
- [x] Animated micro-interactions (hover glows, pulse effects)
- [x] Responsive mobile-first layouts
- [x] Dark mode first (cyber-dark backgrounds)
- [x] HTMX for partial updates
- [x] Toast notification system
- [x] Keyboard shortcuts (j/k navigation, / search)

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Relevance | >80% of top 10 items rated useful |
| Noise Reduction | 50% fewer irrelevant items vs V2 |
| Discovery | 5+ quality sources auto-discovered per month |
| Engagement | Daily active usage |
| Time Saved | <5 min to get full AI news picture |

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| V1 | 2026-01 | Core news agent, web interface, SQLite |
| V2 | 2026-02 | Podcasts, videos, transcription, Claude summaries |
| V3 | 2026-02 | Semantic AI, clustering, personalization, chat, cyberpunk UI |
| V3.1 | 2026-02-09 | UX polish: active nav, toast notifications, keyboard shortcuts |
| V3.2 | 2026-02-14 | Video agent, YouTube feeds, transcript extraction |
| V3.3 | 2026-02-20 | Sources Manager: DB-driven CRUD, type-separated tabs, auto-import |
