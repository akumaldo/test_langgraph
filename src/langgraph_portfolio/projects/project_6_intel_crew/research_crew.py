"""
Project 6 — Research Crew (Sequential Process).

The Research Crew is the FIRST stage of our intelligence pipeline.
It takes PRE-SCRAPED web content and organizes it into structured
competitor profiles.

ARCHITECTURE NOTE — Why we pre-scrape in the Flow, not in the Crew:

  We originally wanted the Scraper agent to use WebScraperTool
  autonomously (like LangGraph's bind_tools from P2). But two issues
  with CrewAI + Ollama made this unreliable:

  1. CrewAI uses Instructor internally to parse ALL agent responses
     (not just output_pydantic). Instructor doesn't handle the
     combination of tools + response_model with Ollama well.

  2. When the Scraper agent scrapes multiple pages, the accumulated
     context grows huge (40k+ tokens). The Gatherer then fails because
     the model can't produce structured output with that much context.

  The fix: separate I/O from reasoning.
    - The Flow (flow.py) calls WebScraperTool directly = deterministic I/O
      with controlled content size
    - The Crew takes truncated scraped text as input = pure LLM reasoning
      with predictable context size

  This is actually a BETTER pattern for production:
    - Deterministic work (HTTP requests) shouldn't depend on LLM decisions
    - You control content size (truncate before sending to the LLM)
    - Failures are easier to debug (network vs LLM errors are separate)
    - The LLM focuses on what it's good at: understanding and structuring

  The WebScraperTool class still exists in tools.py — you learned the
  CrewAI BaseTool pattern. We just call it from Python instead of through
  an agent. This is a common real-world pattern.

This crew uses Process.sequential — tasks run in a FIXED order:
  Task 1 (Analyzer) → Task 2 (Profiler)

Compare to P3 where we used Process.hierarchical for everything.
Why sequential here? Because the order is obvious and fixed.
"""

from crewai import Agent, Crew, Process, Task

# DISABLED: output_pydantic import — see note on profile_task below.
# from .models import CompetitorProfile


# ---------------------------------------------------------------------------
# LLM CONFIGURATION
# ---------------------------------------------------------------------------

LLM_MODEL = "ollama/qwen3.5:35b"


# ---------------------------------------------------------------------------
# AGENTS
#
# Two agents, neither with tools. The scraped content is passed in as text
# via the task description. The agents focus on REASONING — analyzing and
# structuring the raw data.
#
# Compare to P3's agents: same pattern (role, goal, backstory, no tools).
# ---------------------------------------------------------------------------

analyzer = Agent(
    role="Web Content Analyzer",
    goal=(
        "Analyze raw scraped web content and extract key information about "
        "companies — what they do, their products, services, pricing, and "
        "competitive position."
    ),
    backstory=(
        "You are an expert at reading messy web content and finding the "
        "signal in the noise. You can take a wall of scraped text and "
        "identify the important facts: what the company does, what it "
        "sells, how it positions itself. You ignore navigation elements, "
        "ads, and boilerplate — focusing only on substantive content."
    ),
    llm=LLM_MODEL,
    verbose=True,
)

gatherer = Agent(
    role="Intelligence Gatherer",
    goal=(
        "Organize analyzed content into structured competitor profiles "
        "with key details: what the company does, their products, pricing, "
        "strengths, and weaknesses."
    ),
    backstory=(
        "You are a business intelligence analyst. You take analyzed web "
        "content and build structured competitor profiles. You're skilled at "
        "reading between the lines — identifying strengths and weaknesses "
        "even when they're not explicitly stated. When information is "
        "missing, you note 'Not available' rather than guessing."
    ),
    llm=LLM_MODEL,
    verbose=True,
)


# ---------------------------------------------------------------------------
# BUILDING THE CREW
# ---------------------------------------------------------------------------


def build_research_crew(
    target_company: str,
    competitors: list[str],
    scraped_content: str,
) -> Crew:
    """Build and return the Research Crew.

    Args:
        target_company: The company being analyzed.
        competitors: List of competitor names.
        scraped_content: Pre-scraped web content (from WebScraperTool,
            called in the Flow). Already truncated to a manageable size.
    """

    analyze_task = Task(
        description=(
            f"Analyze the following scraped web content about {target_company} "
            f"and its competitors: {', '.join(competitors)}.\n\n"
            f"SCRAPED WEB CONTENT:\n"
            f"{scraped_content}\n\n"
            f"For each company mentioned, extract:\n"
            f"  - What the company does\n"
            f"  - Key products or services\n"
            f"  - Pricing information (if available)\n"
            f"  - Notable features or capabilities\n"
            f"  - Any competitive advantages or disadvantages mentioned\n\n"
            f"Organize your findings by company name."
        ),
        expected_output=(
            "Analyzed content organized by company, with key facts about "
            "each company's products, services, pricing, and market position."
        ),
        agent=analyzer,
    )

    profile_task = Task(
        description=(
            f"Using the analyzed content, create a structured competitor "
            f"profile for EACH of these competitors: "
            f"{', '.join(competitors)}.\n\n"
            f"For each competitor, provide:\n"
            f"  - name: the company name\n"
            f"  - description: brief description of what the company does\n"
            f"  - products: list of key products or services\n"
            f"  - pricing_info: pricing details, or 'Not available' if not found\n"
            f"  - strengths: list of competitive strengths\n"
            f"  - weaknesses: list of competitive weaknesses\n\n"
            f"Also include a profile for the target company ({target_company}) "
            f"so we can compare."
        ),
        expected_output=(
            "A structured list of competitor profiles, each with name, "
            "description, products, pricing, strengths, and weaknesses."
        ),
        agent=gatherer,
        # DISABLED: output_pydantic=CompetitorProfile
        #
        # CrewAI uses the Instructor library to enforce structured output.
        # Instructor works by sending a tool-call schema to the LLM and
        # expecting it to "call" the tool with structured JSON. But Ollama/qwen
        # returns tool_calls=None (empty), so Instructor crashes with:
        #   "Instructor does not support multiple tool calls, use List[Model]"
        #
        # Workaround: let the agent output free text. The flow (flow.py) reads
        # result.raw as a string and passes it to the next crew as context.
        # We lose automatic Pydantic validation, but the pipeline still works.
        #
        # With an OpenAI/Anthropic API, you'd re-enable this:
        #   output_pydantic=CompetitorProfile,
    )

    crew = Crew(
        agents=[analyzer, gatherer],
        tasks=[analyze_task, profile_task],
        process=Process.sequential,
        # DISABLED: memory=True requires an embedding model for storing/retrieving
        # agent memories (short-term, long-term, entity). By default CrewAI uses
        # OpenAI's embedding API, which needs OPENAI_API_KEY. To enable with Ollama:
        #   memory=True,
        #   embedder={"provider": "ollama", "config": {"model": "nomic-embed-text"}},
        memory=False,
        verbose=True,
    )

    return crew
