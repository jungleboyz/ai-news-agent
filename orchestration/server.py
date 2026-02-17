"""MCP Server for AI News Agent orchestration.

This server exposes the AI News Agent functionality through the Model Context Protocol,
allowing AI assistants to interact with news, podcasts, and video agents.
"""

import os
import sys
import json
import asyncio
from datetime import date, datetime
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    Resource,
    ResourceTemplate,
    Prompt,
    PromptMessage,
    PromptArgument,
    GetPromptResult,
)

# Import agent modules
from agent import run_agent, USER_INTERESTS
from podcast_agent import run_podcast_agent
from video_agent import run_video_agent
from services.vector_store import VectorStore
from services.semantic_scorer import SemanticScorer
from web.database import SessionLocal, init_db
from web.models import Digest, Item

# Initialize MCP server
mcp_server = Server("ai-news-agent")

# Initialize database
init_db()


# =============================================================================
# TOOLS - Actions the AI can perform
# =============================================================================

@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools."""
    return [
        Tool(
            name="run_news_agent",
            description="Run the news agent to fetch and process RSS feeds. Returns top AI-relevant news articles with summaries.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="run_podcast_agent",
            description="Run the podcast agent to fetch and process podcast episodes. Returns top AI-relevant episodes with transcripts and summaries.",
            inputSchema={
                "type": "object",
                "properties": {
                    "skip_transcription": {
                        "type": "boolean",
                        "description": "Skip audio transcription (faster but less accurate scoring)",
                        "default": False,
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="run_video_agent",
            description="Run the video agent to fetch and process YouTube videos. Returns top AI-relevant videos with transcripts and summaries.",
            inputSchema={
                "type": "object",
                "properties": {
                    "max_videos": {
                        "type": "integer",
                        "description": "Maximum number of videos to return",
                        "default": 10,
                    },
                    "days_back": {
                        "type": "integer",
                        "description": "How many days back to look for videos",
                        "default": 7,
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="run_full_digest",
            description="Run all agents (news, podcast, video) and generate a complete daily digest.",
            inputSchema={
                "type": "object",
                "properties": {
                    "skip_transcription": {
                        "type": "boolean",
                        "description": "Skip podcast/video transcription for faster processing",
                        "default": False,
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="semantic_search",
            description="Search across all content using semantic similarity. Returns items most relevant to your query.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text",
                    },
                    "item_type": {
                        "type": "string",
                        "enum": ["news", "podcast", "video"],
                        "description": "Filter by content type (optional)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="find_similar",
            description="Find content similar to a specific item by its ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "item_id": {
                        "type": "string",
                        "description": "ID of the item to find similar content for",
                    },
                    "item_type": {
                        "type": "string",
                        "enum": ["news", "podcast", "video"],
                        "description": "Type of the source item",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return",
                        "default": 5,
                    },
                },
                "required": ["item_id", "item_type"],
            },
        ),
        Tool(
            name="get_digest",
            description="Get a specific digest by date.",
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format (defaults to today)",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="list_digests",
            description="List recent digests.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of digests to return",
                        "default": 10,
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="get_vector_stats",
            description="Get statistics about the vector store (embedding counts per collection).",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Execute a tool and return results."""
    try:
        if name == "run_news_agent":
            result = await asyncio.to_thread(run_agent)
            return [TextContent(type="text", text=f"News digest generated: {result}")]

        elif name == "run_podcast_agent":
            skip = arguments.get("skip_transcription", False)
            episodes = await asyncio.to_thread(run_podcast_agent, skip)
            if episodes:
                summary = "\n".join([
                    f"- [{ep['score']}] {ep['title']} ({ep.get('show_name', 'Unknown')})"
                    for ep in episodes
                ])
                return [TextContent(type="text", text=f"Found {len(episodes)} podcast episodes:\n{summary}")]
            return [TextContent(type="text", text="No new podcast episodes found.")]

        elif name == "run_video_agent":
            max_videos = arguments.get("max_videos", 10)
            days_back = arguments.get("days_back", 7)
            videos = await asyncio.to_thread(run_video_agent, max_videos, days_back)
            if videos:
                summary = "\n".join([
                    f"- [{v['score']}] {v['title']} ({v.get('channel', 'Unknown')})"
                    for v in videos
                ])
                return [TextContent(type="text", text=f"Found {len(videos)} AI-relevant videos:\n{summary}")]
            return [TextContent(type="text", text="No new AI-relevant videos found.")]

        elif name == "run_full_digest":
            skip = arguments.get("skip_transcription", False)
            os.environ["SKIP_PODCAST_TRANSCRIPTION"] = "1" if skip else ""
            result = await asyncio.to_thread(run_agent)
            return [TextContent(type="text", text=f"Full digest generated: {result}")]

        elif name == "semantic_search":
            query = arguments.get("query", "")
            item_type = arguments.get("item_type")
            limit = arguments.get("limit", 10)

            vector_store = VectorStore()
            results = vector_store.search(query, item_type=item_type, limit=limit)

            if results:
                output = f"Found {len(results)} results for '{query}':\n\n"
                for r in results:
                    meta = r.get("metadata", {})
                    output += f"- **{meta.get('title', 'Untitled')}**\n"
                    output += f"  Type: {meta.get('item_type', 'unknown')} | Similarity: {r['similarity']:.3f}\n"
                    if meta.get("link"):
                        output += f"  Link: {meta['link']}\n"
                    output += "\n"
                return [TextContent(type="text", text=output)]
            return [TextContent(type="text", text=f"No results found for '{query}'.")]

        elif name == "find_similar":
            item_id = arguments.get("item_id", "")
            item_type = arguments.get("item_type", "news")
            limit = arguments.get("limit", 5)

            vector_store = VectorStore()
            item = vector_store.get_item(item_id, item_type)

            if not item or not item.get("embedding"):
                return [TextContent(type="text", text=f"Item {item_id} not found or has no embedding.")]

            results = vector_store.find_similar(
                embedding=item["embedding"],
                item_type=item_type,
                threshold=0.5,
                exclude_ids=[item_id],
            )[:limit]

            if results:
                output = f"Items similar to '{item.get('metadata', {}).get('title', item_id)}':\n\n"
                for r in results:
                    meta = r.get("metadata", {})
                    output += f"- **{meta.get('title', 'Untitled')}** (similarity: {r['similarity']:.3f})\n"
                return [TextContent(type="text", text=output)]
            return [TextContent(type="text", text="No similar items found.")]

        elif name == "get_digest":
            date_str = arguments.get("date", date.today().isoformat())
            try:
                digest_date = date.fromisoformat(date_str)
            except ValueError:
                return [TextContent(type="text", text=f"Invalid date format: {date_str}. Use YYYY-MM-DD.")]

            db = SessionLocal()
            try:
                digest = db.query(Digest).filter(Digest.date == digest_date).first()
                if not digest:
                    return [TextContent(type="text", text=f"No digest found for {date_str}.")]

                items = db.query(Item).filter(Item.digest_id == digest.id).order_by(Item.position).all()

                output = f"# Digest for {date_str}\n\n"
                output += f"Sources: {digest.news_sources_count} news, {digest.podcast_sources_count} podcasts\n"
                output += f"Items considered: {digest.total_items_considered}\n\n"

                for item in items:
                    icon = {"news": "📰", "podcast": "🎙️", "video": "📺"}.get(item.type, "📄")
                    output += f"## {icon} {item.title}\n"
                    output += f"- Type: {item.type} | Score: {item.score}"
                    if item.semantic_score:
                        output += f" | Semantic: {item.semantic_score:.3f}"
                    output += f"\n- Link: {item.link}\n"
                    if item.summary:
                        output += f"\n{item.summary}\n"
                    output += "\n---\n\n"

                return [TextContent(type="text", text=output)]
            finally:
                db.close()

        elif name == "list_digests":
            limit = arguments.get("limit", 10)

            db = SessionLocal()
            try:
                digests = db.query(Digest).order_by(Digest.date.desc()).limit(limit).all()
                if not digests:
                    return [TextContent(type="text", text="No digests found.")]

                output = "# Recent Digests\n\n"
                for d in digests:
                    item_count = db.query(Item).filter(Item.digest_id == d.id).count()
                    output += f"- **{d.date}**: {item_count} items ({d.news_sources_count} news sources, {d.podcast_sources_count} podcast sources)\n"

                return [TextContent(type="text", text=output)]
            finally:
                db.close()

        elif name == "get_vector_stats":
            vector_store = VectorStore()
            stats = {}
            for item_type in ["news", "podcast", "video"]:
                try:
                    stats[item_type] = vector_store.get_collection_count(item_type)
                except Exception:
                    stats[item_type] = 0

            output = "# Vector Store Statistics\n\n"
            output += f"- News embeddings: {stats['news']}\n"
            output += f"- Podcast embeddings: {stats['podcast']}\n"
            output += f"- Video embeddings: {stats['video']}\n"
            output += f"- **Total**: {sum(stats.values())}\n"

            return [TextContent(type="text", text=output)]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        return [TextContent(type="text", text=f"Error executing {name}: {str(e)}")]


# =============================================================================
# RESOURCES - Data the AI can read
# =============================================================================

@mcp_server.list_resources()
async def list_resources() -> list[Resource]:
    """List available resources."""
    resources = [
        Resource(
            uri="config://interests",
            name="User Interests",
            description="Current list of topics/keywords used for scoring",
            mimeType="application/json",
        ),
        Resource(
            uri="config://sources",
            name="RSS Sources",
            description="List of RSS feed URLs for news",
            mimeType="text/plain",
        ),
        Resource(
            uri="config://podcasts",
            name="Podcast Feeds",
            description="List of podcast RSS feed URLs",
            mimeType="text/plain",
        ),
        Resource(
            uri="config://videos",
            name="Video Channels",
            description="List of YouTube channel URLs",
            mimeType="text/plain",
        ),
    ]

    # Add recent digests as resources
    db = SessionLocal()
    try:
        digests = db.query(Digest).order_by(Digest.date.desc()).limit(5).all()
        for d in digests:
            resources.append(Resource(
                uri=f"digest://{d.date}",
                name=f"Digest {d.date}",
                description=f"AI News Digest for {d.date}",
                mimeType="text/markdown",
            ))
    finally:
        db.close()

    return resources


@mcp_server.list_resource_templates()
async def list_resource_templates() -> list[ResourceTemplate]:
    """List resource templates for dynamic resources."""
    return [
        ResourceTemplate(
            uriTemplate="digest://{date}",
            name="Digest by Date",
            description="Get a digest for a specific date (YYYY-MM-DD)",
            mimeType="text/markdown",
        ),
        ResourceTemplate(
            uriTemplate="item://{item_type}/{item_id}",
            name="Item Details",
            description="Get details for a specific item",
            mimeType="application/json",
        ),
    ]


@mcp_server.read_resource()
async def read_resource(uri: str) -> str:
    """Read a resource by URI."""
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if uri == "config://interests":
        return json.dumps(USER_INTERESTS, indent=2)

    elif uri == "config://sources":
        sources_path = os.path.join(project_dir, "sources.txt")
        if os.path.exists(sources_path):
            with open(sources_path, "r") as f:
                return f.read()
        return "# No sources configured"

    elif uri == "config://podcasts":
        podcasts_path = os.path.join(project_dir, "podcasts.txt")
        if os.path.exists(podcasts_path):
            with open(podcasts_path, "r") as f:
                return f.read()
        return "# No podcasts configured"

    elif uri == "config://videos":
        videos_path = os.path.join(project_dir, "videos.txt")
        if os.path.exists(videos_path):
            with open(videos_path, "r") as f:
                return f.read()
        return "# No video channels configured"

    elif uri.startswith("digest://"):
        date_str = uri.replace("digest://", "")
        try:
            digest_date = date.fromisoformat(date_str)
        except ValueError:
            return f"Invalid date format: {date_str}"

        db = SessionLocal()
        try:
            digest = db.query(Digest).filter(Digest.date == digest_date).first()
            if not digest:
                return f"No digest found for {date_str}"

            # Read markdown file if available
            if digest.md_path and os.path.exists(digest.md_path):
                with open(digest.md_path, "r") as f:
                    return f.read()

            # Generate markdown from database
            items = db.query(Item).filter(Item.digest_id == digest.id).order_by(Item.position).all()
            output = f"# AI News Digest — {date_str}\n\n"
            for item in items:
                output += f"## {item.title}\n"
                output += f"- Type: {item.type}\n"
                output += f"- Link: {item.link}\n"
                if item.summary:
                    output += f"\n{item.summary}\n"
                output += "\n---\n\n"
            return output
        finally:
            db.close()

    elif uri.startswith("item://"):
        parts = uri.replace("item://", "").split("/")
        if len(parts) != 2:
            return json.dumps({"error": "Invalid item URI format"})

        item_type, item_id = parts
        vector_store = VectorStore()
        item = vector_store.get_item(item_id, item_type)

        if item:
            # Don't include the embedding in the response (too large)
            result = {
                "id": item["id"],
                "text": item["text"],
                "metadata": item["metadata"],
            }
            return json.dumps(result, indent=2)
        return json.dumps({"error": f"Item {item_id} not found"})

    return f"Unknown resource: {uri}"


# =============================================================================
# PROMPTS - Pre-built interaction templates
# =============================================================================

@mcp_server.list_prompts()
async def list_prompts() -> list[Prompt]:
    """List available prompts."""
    return [
        Prompt(
            name="daily_brief",
            description="Generate a daily briefing of the most important AI news",
            arguments=[
                PromptArgument(
                    name="focus_area",
                    description="Optional focus area (e.g., 'LLMs', 'AI agents', 'enterprise AI')",
                    required=False,
                ),
            ],
        ),
        Prompt(
            name="topic_deep_dive",
            description="Do a deep dive on a specific AI topic across all content",
            arguments=[
                PromptArgument(
                    name="topic",
                    description="The topic to explore (e.g., 'RAG', 'multimodal AI', 'AI safety')",
                    required=True,
                ),
            ],
        ),
        Prompt(
            name="weekly_summary",
            description="Generate a summary of the week's AI news highlights",
            arguments=[],
        ),
        Prompt(
            name="competitor_watch",
            description="Track news about specific AI companies",
            arguments=[
                PromptArgument(
                    name="companies",
                    description="Comma-separated list of companies to track",
                    required=True,
                ),
            ],
        ),
        Prompt(
            name="trend_analysis",
            description="Analyze emerging trends in AI based on recent content",
            arguments=[],
        ),
    ]


@mcp_server.get_prompt()
async def get_prompt(name: str, arguments: dict[str, str] | None = None) -> GetPromptResult:
    """Get a prompt by name with arguments."""
    arguments = arguments or {}

    if name == "daily_brief":
        focus = arguments.get("focus_area", "")
        focus_instruction = f" with a focus on {focus}" if focus else ""

        return GetPromptResult(
            description="Daily AI news briefing",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=f"""Please generate a daily briefing of the most important AI news{focus_instruction}.

Steps:
1. First, use the `get_digest` tool to fetch today's digest
2. Analyze the items and identify the 3-5 most significant stories
3. For each story, explain:
   - What happened
   - Why it matters
   - Potential implications

Format the output as an executive brief that can be read in 2-3 minutes."""
                    ),
                ),
            ],
        )

    elif name == "topic_deep_dive":
        topic = arguments.get("topic", "AI")

        return GetPromptResult(
            description=f"Deep dive on {topic}",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=f"""Do a comprehensive deep dive on "{topic}" across all available content.

Steps:
1. Use `semantic_search` to find all content related to "{topic}"
2. Analyze the results to identify:
   - Key developments and announcements
   - Major players and their positions
   - Technical details and innovations
   - Market implications
3. Use `find_similar` on the most relevant items to discover related content

Provide a comprehensive analysis with:
- Executive summary
- Key findings (5-7 bullet points)
- Detailed analysis
- Outlook and predictions"""
                    ),
                ),
            ],
        )

    elif name == "weekly_summary":
        return GetPromptResult(
            description="Weekly AI news summary",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text="""Generate a comprehensive summary of this week's AI news.

Steps:
1. Use `list_digests` to see available digests
2. Read the last 7 days of digests using `get_digest`
3. Identify recurring themes and the biggest stories
4. Synthesize into a coherent weekly narrative

Format:
- Top 5 stories of the week
- Emerging themes
- Notable company moves
- Technical breakthroughs
- What to watch next week"""
                    ),
                ),
            ],
        )

    elif name == "competitor_watch":
        companies = arguments.get("companies", "OpenAI, Anthropic, Google")

        return GetPromptResult(
            description=f"Competitor tracking for {companies}",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=f"""Track and analyze news about these AI companies: {companies}

Steps:
1. For each company, use `semantic_search` with the company name
2. Compile all relevant news, announcements, and mentions
3. Analyze competitive positioning

Provide:
- Summary per company
- Recent moves and announcements
- Competitive analysis
- Market positioning insights"""
                    ),
                ),
            ],
        )

    elif name == "trend_analysis":
        return GetPromptResult(
            description="AI trend analysis",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text="""Analyze emerging trends in AI based on recent content.

Steps:
1. Use `get_vector_stats` to understand the data available
2. Use `semantic_search` for key trend indicators:
   - "breakthrough" or "innovation"
   - "funding" or "investment"
   - "partnership" or "collaboration"
   - "regulation" or "policy"
3. Identify patterns across multiple stories

Provide:
- Top 5 emerging trends
- Evidence supporting each trend
- Predictions for the next 3-6 months
- Potential risks and opportunities"""
                    ),
                ),
            ],
        )

    return GetPromptResult(
        description="Unknown prompt",
        messages=[
            PromptMessage(
                role="user",
                content=TextContent(type="text", text=f"Unknown prompt: {name}"),
            ),
        ],
    )


# =============================================================================
# SERVER ENTRY POINT
# =============================================================================

async def run_server():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(
            read_stream,
            write_stream,
            mcp_server.create_initialization_options(),
        )


def main():
    """Main entry point."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
