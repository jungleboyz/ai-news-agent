"""Prompt templates for AI News Agent MCP server."""

from typing import Optional, List
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """A prompt template with metadata."""
    name: str
    description: str
    template: str
    arguments: List[dict]
    category: str = "general"


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

PROMPT_TEMPLATES = {
    # --- Daily Operations ---
    "daily_brief": PromptTemplate(
        name="daily_brief",
        description="Generate a daily briefing of the most important AI news",
        category="daily",
        arguments=[
            {
                "name": "focus_area",
                "description": "Optional focus area (e.g., 'LLMs', 'AI agents', 'enterprise AI')",
                "required": False,
            },
        ],
        template="""Please generate a daily briefing of the most important AI news{focus_instruction}.

Steps:
1. First, use the `get_digest` tool to fetch today's digest
2. Analyze the items and identify the 3-5 most significant stories
3. For each story, explain:
   - What happened
   - Why it matters
   - Potential implications

Format the output as an executive brief that can be read in 2-3 minutes.

Key considerations:
- Prioritize breaking news and major announcements
- Highlight any stories affecting enterprise AI adoption
- Note any significant funding or acquisition news
- Identify emerging technical trends""",
    ),

    "morning_scan": PromptTemplate(
        name="morning_scan",
        description="Quick morning scan of overnight AI developments",
        category="daily",
        arguments=[],
        template="""Perform a quick morning scan of overnight AI developments.

Steps:
1. Run `run_full_digest` with skip_transcription=true for speed
2. Identify any breaking news or major announcements
3. Flag items that need immediate attention

Provide a 1-minute summary with:
- Top 3 headlines
- Any urgent items requiring attention
- Quick market sentiment indicator""",
    ),

    # --- Research & Analysis ---
    "topic_deep_dive": PromptTemplate(
        name="topic_deep_dive",
        description="Do a deep dive on a specific AI topic across all content",
        category="research",
        arguments=[
            {
                "name": "topic",
                "description": "The topic to explore (e.g., 'RAG', 'multimodal AI', 'AI safety')",
                "required": True,
            },
        ],
        template="""Do a comprehensive deep dive on "{topic}" across all available content.

Steps:
1. Use `semantic_search` to find all content related to "{topic}"
2. Analyze the results to identify:
   - Key developments and announcements
   - Major players and their positions
   - Technical details and innovations
   - Market implications
3. Use `find_similar` on the most relevant items to discover related content
4. Check `get_vector_stats` to understand data coverage

Provide a comprehensive analysis with:
- Executive summary (2-3 sentences)
- Key findings (5-7 bullet points)
- Detailed analysis by sub-topic
- Key players and their positions
- Technical innovations
- Market implications
- Outlook and predictions
- Recommended follow-up research""",
    ),

    "trend_analysis": PromptTemplate(
        name="trend_analysis",
        description="Analyze emerging trends in AI based on recent content",
        category="research",
        arguments=[
            {
                "name": "timeframe",
                "description": "Timeframe for analysis (e.g., 'week', 'month')",
                "required": False,
            },
        ],
        template="""Analyze emerging trends in AI based on recent content.

Steps:
1. Use `get_vector_stats` to understand the data available
2. Use `semantic_search` for key trend indicators:
   - "breakthrough" or "innovation"
   - "funding" or "investment"
   - "partnership" or "collaboration"
   - "regulation" or "policy"
   - "open source" or "release"
3. Use `list_digests` to see data coverage
4. Identify patterns across multiple stories

Provide:
- Top 5 emerging trends with evidence
- Trend momentum indicators (accelerating/steady/declining)
- Cross-cutting themes
- Predictions for the next 3-6 months
- Potential risks and opportunities
- Suggested areas for deeper research""",
    ),

    # --- Competitive Intelligence ---
    "competitor_watch": PromptTemplate(
        name="competitor_watch",
        description="Track news about specific AI companies",
        category="competitive",
        arguments=[
            {
                "name": "companies",
                "description": "Comma-separated list of companies to track",
                "required": True,
            },
        ],
        template="""Track and analyze news about these AI companies: {companies}

Steps:
1. For each company, use `semantic_search` with the company name
2. Compile all relevant news, announcements, and mentions
3. Cross-reference between companies for competitive moves

Provide per company:
- Recent announcements and releases
- Product/feature updates
- Partnerships and collaborations
- Hiring and team changes
- Funding/financial news

Overall competitive analysis:
- Market positioning comparison
- Competitive moves and responses
- Emerging battlegrounds
- Strategic implications""",
    ),

    "market_landscape": PromptTemplate(
        name="market_landscape",
        description="Map the competitive landscape for an AI market segment",
        category="competitive",
        arguments=[
            {
                "name": "segment",
                "description": "Market segment (e.g., 'AI coding assistants', 'enterprise LLMs')",
                "required": True,
            },
        ],
        template="""Map the competitive landscape for the {segment} market segment.

Steps:
1. Use `semantic_search` to find content related to "{segment}"
2. Identify all players mentioned in the results
3. Categorize by: leaders, challengers, niche players, emerging entrants

Provide:
- Market overview and size indicators
- Player mapping with positioning
- Recent competitive moves
- Differentiation strategies
- Entry barriers and moats
- Predicted market evolution""",
    ),

    # --- Summaries ---
    "weekly_summary": PromptTemplate(
        name="weekly_summary",
        description="Generate a summary of the week's AI news highlights",
        category="summary",
        arguments=[],
        template="""Generate a comprehensive summary of this week's AI news.

Steps:
1. Use `list_digests` to see available digests
2. Read the last 7 days of digests using `get_digest`
3. Identify recurring themes and the biggest stories
4. Synthesize into a coherent weekly narrative

Format:
- Week at a glance (3 sentences)
- Top 5 stories of the week with context
- Emerging themes
- Notable company moves
- Technical breakthroughs
- What to watch next week
- Key questions going forward""",
    ),

    "executive_summary": PromptTemplate(
        name="executive_summary",
        description="Generate an executive summary for leadership",
        category="summary",
        arguments=[
            {
                "name": "audience",
                "description": "Target audience (e.g., 'C-suite', 'board', 'investors')",
                "required": False,
            },
        ],
        template="""Generate an executive summary suitable for {audience}.

Steps:
1. Get today's digest with `get_digest`
2. Filter for high-impact items
3. Focus on business implications over technical details

Format for executives:
- One-paragraph overview
- 3 critical developments (bullet points)
- Strategic implications
- Recommended actions
- Key metrics/numbers to note

Keep it concise - readable in under 2 minutes.""",
    ),

    # --- Technical ---
    "technical_analysis": PromptTemplate(
        name="technical_analysis",
        description="Analyze technical developments in a specific AI area",
        category="technical",
        arguments=[
            {
                "name": "technology",
                "description": "Technology area (e.g., 'transformers', 'RLHF', 'inference optimization')",
                "required": True,
            },
        ],
        template="""Analyze technical developments in {technology}.

Steps:
1. Use `semantic_search` with technical terms related to "{technology}"
2. Identify papers, releases, and technical announcements
3. Look for benchmarks and performance claims

Provide:
- Current state of the art
- Recent breakthroughs and improvements
- Key research directions
- Open problems and challenges
- Practical implications for implementation
- Resources for learning more""",
    ),

    # --- Custom Operations ---
    "source_health_check": PromptTemplate(
        name="source_health_check",
        description="Check the health and quality of configured sources",
        category="operations",
        arguments=[],
        template="""Check the health and quality of all configured sources.

Steps:
1. Read sources with the resources: config://sources, config://podcasts, config://videos
2. Check `get_vector_stats` for embedding coverage
3. Use `list_digests` to see recent activity

Evaluate:
- Source availability and freshness
- Content quality indicators
- Coverage gaps
- Recommendations for new sources
- Sources to consider removing""",
    ),

    "content_gap_analysis": PromptTemplate(
        name="content_gap_analysis",
        description="Identify gaps in content coverage",
        category="operations",
        arguments=[
            {
                "name": "topics",
                "description": "Comma-separated topics to check coverage for",
                "required": True,
            },
        ],
        template="""Analyze content coverage gaps for: {topics}

Steps:
1. For each topic, use `semantic_search` to find related content
2. Assess coverage depth and recency
3. Identify missing perspectives or sources

Provide:
- Coverage assessment per topic (good/moderate/poor)
- Missing sub-topics or angles
- Recommended sources to add
- Suggested search terms for manual research""",
    ),
}


def get_prompt_template(name: str, arguments: Optional[dict] = None) -> Optional[dict]:
    """Get a prompt template by name with arguments substituted.

    Args:
        name: Template name.
        arguments: Arguments to substitute.

    Returns:
        Dict with description and formatted message, or None if not found.
    """
    template = PROMPT_TEMPLATES.get(name)
    if not template:
        return None

    arguments = arguments or {}

    # Build the formatted template
    formatted = template.template

    # Handle special argument substitutions
    if name == "daily_brief":
        focus = arguments.get("focus_area", "")
        focus_instruction = f" with a focus on {focus}" if focus else ""
        formatted = formatted.replace("{focus_instruction}", focus_instruction)

    elif name == "executive_summary":
        audience = arguments.get("audience", "leadership")
        formatted = formatted.replace("{audience}", audience)

    else:
        # Generic substitution
        for key, value in arguments.items():
            formatted = formatted.replace(f"{{{key}}}", str(value))

    return {
        "name": template.name,
        "description": template.description,
        "category": template.category,
        "message": formatted,
    }


def list_prompt_names() -> List[dict]:
    """List all available prompt templates.

    Returns:
        List of prompt metadata dicts.
    """
    return [
        {
            "name": t.name,
            "description": t.description,
            "category": t.category,
            "arguments": t.arguments,
        }
        for t in PROMPT_TEMPLATES.values()
    ]


def get_prompts_by_category(category: str) -> List[dict]:
    """Get prompts filtered by category.

    Args:
        category: Category to filter by.

    Returns:
        List of matching prompt metadata.
    """
    return [
        {
            "name": t.name,
            "description": t.description,
            "arguments": t.arguments,
        }
        for t in PROMPT_TEMPLATES.values()
        if t.category == category
    ]
