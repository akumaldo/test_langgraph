"""
Project 6 — Pydantic models for structured output.

These models define the DATA that flows between our three crews:

  Research Crew → CompetitorProfile (per competitor)
  Analysis Crew → CompetitiveAnalysis (comparison + insights)
  Report Crew   → IntelligenceReport (executive briefing)

Same idea as P3's models (Outline, ResearchNotes, ArticleDraft, etc.),
but here the models flow BETWEEN crews (via the Flow), not just between
tasks within a single crew.

Why Pydantic?
  - Ideally, each crew's final task would use `output_pydantic=<Model>` to
    force the LLM to return JSON matching the schema.
  - The framework would validate the output automatically.
  - We'd get typed, predictable data — not freeform text.

HOWEVER: output_pydantic is currently DISABLED because CrewAI uses the
Instructor library internally, and Instructor's tool-call approach doesn't
work with Ollama/qwen (the model returns empty tool_calls). With an
OpenAI/Anthropic API, you'd re-enable output_pydantic on the last task of
each crew. See research_crew.py for the full explanation.

Currently, crews return freeform text (result.raw) and the flow passes
it as string context to the next crew. The models are kept here as
REFERENCE for the expected data shape and for future re-enablement.

You already know this from P3. The new thing in P6 is that these models
are the CONTRACT between independent crews. The Research Crew doesn't know
anything about the Analysis Crew — they're connected only by the shape of
the data they produce and consume.
"""

from datetime import datetime

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# RESEARCH CREW OUTPUT MODELS
#
# The Research Crew scrapes websites and organizes raw content into
# structured competitor profiles. These models define what that output
# looks like.
# ---------------------------------------------------------------------------


class ScrapedPage(BaseModel):
    """A single scraped web page.

    The WebScraperTool produces this (as text), but we don't use it as a
    task's output_pydantic — it's an intermediate representation. The Scraper
    agent's task returns text, and the Gatherer agent reads it to build
    CompetitorProfiles.
    """
    url: str = Field(description="The URL that was scraped")
    title: str = Field(description="The page title")
    raw_content: str = Field(description="Cleaned text content from the page")
    scraped_at: str = Field(description="When the page was scraped (ISO format)")


class CompetitorProfile(BaseModel):
    """A structured profile for one competitor.

    This is the Research Crew's FINAL output (from the Gatherer agent).
    The Gatherer reads raw scraped content and organizes it into these fields.

    Compare to P3's ResearchNotes — same idea (structured research output),
    but here it represents a company profile instead of article notes.
    """
    name: str = Field(description="The competitor's company name")
    description: str = Field(description="Brief description of what the company does")
    products: list[str] = Field(description="Key products or services offered")
    pricing_info: str = Field(description="Pricing details if found, or 'Not available'")
    strengths: list[str] = Field(description="Identified competitive strengths")
    weaknesses: list[str] = Field(description="Identified competitive weaknesses")


# ---------------------------------------------------------------------------
# ANALYSIS CREW OUTPUT MODEL
#
# The Analysis Crew takes competitor profiles and produces a comprehensive
# competitive analysis — comparisons, market positioning, strategic insights.
# ---------------------------------------------------------------------------


class CompetitiveAnalysis(BaseModel):
    """The Analysis Crew's output: a comprehensive competitive comparison.

    Three agents contribute to this (Analyst, Comparator, Strategist),
    coordinated by a hierarchical manager. But the crew's output is a
    single unified model — the manager ensures all pieces fit together.

    Compare to P3 where the hierarchical manager coordinated 5 agents
    producing different output types. Here, the manager coordinates 3
    agents producing parts of ONE analysis.
    """
    target_company: str = Field(description="The company being analyzed")
    competitors: list[CompetitorProfile] = Field(
        description="Enriched competitor profiles with analysis added"
    )
    feature_comparison: str = Field(
        description="Side-by-side comparison of products, features, and pricing"
    )
    market_positioning: str = Field(
        description="Analysis of each competitor's market position and target audience"
    )
    key_insights: list[str] = Field(
        description="Strategic insights and actionable observations"
    )


# ---------------------------------------------------------------------------
# REPORT CREW OUTPUT MODELS
#
# The Report Crew takes the analysis and produces a polished executive
# briefing — both as structured data and as a markdown file.
# ---------------------------------------------------------------------------


class CompetitorSummary(BaseModel):
    """A brief summary of one competitor for the final report.

    This is a simplified view of a competitor, designed for executive
    consumption. The full CompetitorProfile has raw details; this has
    the distilled takeaways.
    """
    name: str = Field(description="The competitor's company name")
    overview: str = Field(description="One-paragraph overview of the competitor")
    key_strengths: list[str] = Field(description="Top 2-3 competitive strengths")
    key_weaknesses: list[str] = Field(description="Top 2-3 competitive weaknesses")
    competitive_position: str = Field(
        description="How this competitor compares to the target company"
    )


class IntelligenceReport(BaseModel):
    """The Report Crew's FINAL output: a polished intelligence briefing.

    This is the end product of the entire pipeline:
      Research → Analysis → Report

    Compare to P3's PublishedArticle — same idea (final deliverable),
    but more structured. P3 had just title + content. Here we have
    distinct sections that can be rendered as console output AND as a
    markdown file.
    """
    title: str = Field(description="Report title")
    executive_summary: str = Field(
        description="2-3 paragraph executive summary of key findings"
    )
    detailed_analysis: str = Field(
        description="Full analysis narrative with comparisons and market context"
    )
    competitor_summaries: list[CompetitorSummary] = Field(
        description="Per-competitor breakdown for quick reference"
    )
    recommendations: list[str] = Field(
        description="Actionable strategic recommendations based on the analysis"
    )
    generated_at: str = Field(
        description="When this report was generated (ISO format)"
    )
