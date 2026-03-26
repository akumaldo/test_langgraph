"""
Project 6 — Analysis Crew (Hierarchical Process).

The Analysis Crew is the SECOND stage of our intelligence pipeline.
It takes the competitor profiles from the Research Crew and produces
a comprehensive competitive analysis.

This crew uses Process.hierarchical — a MANAGER LLM coordinates the agents.
You already used this in P3 for the writing team. The difference here is
that you can COMPARE it to the Research Crew's sequential process and
understand when each makes sense.

Why hierarchical for analysis?
  Unlike research (scrape → organize, fixed order), analysis is ADAPTIVE.
  The manager might:
    - Ask the Analyst to dig deeper after the Comparator finds a pricing gap
    - Have the Strategist revise recommendations after new insights emerge
    - Reorder tasks based on what the data reveals

  This is like LangGraph's conditional edges — the next step depends on
  what happened before. But instead of YOU writing the routing logic,
  a manager LLM makes those decisions.

  Trade-off:
    Sequential:    faster (no manager calls), predictable, less flexible
    Hierarchical:  slower (extra LLM calls for manager), adaptive, smarter

  Rule of thumb:
    If you can draw the flow as a straight line → sequential
    If "it depends" → hierarchical
"""

from crewai import Agent, Crew, Process, Task

# DISABLED: output_pydantic import — Instructor + Ollama incompatibility.
# See research_crew.py for the full explanation.
# from .models import CompetitiveAnalysis


# ---------------------------------------------------------------------------
# LLM CONFIGURATION
# ---------------------------------------------------------------------------

LLM_MODEL = "ollama/qwen3.5:35b"


# ---------------------------------------------------------------------------
# AGENTS
#
# Three agents with complementary roles. None have tools — they work purely
# with the text data from the Research Crew.
#
# In hierarchical mode, the manager agent (auto-created by CrewAI) reads
# all task descriptions, looks at available agents, and decides:
#   1. Which agent handles which task
#   2. In what order
#   3. Whether to send work back for revision
#
# The agent= field on each Task is a SUGGESTION in hierarchical mode.
# The manager CAN reassign tasks if it thinks another agent is better
# suited. This is fundamentally different from sequential mode where
# the agent assignment is FIXED.
#
# Compare to P3:
#   In P3, we had 5 agents in one hierarchical crew. The manager coordinated
#   planner → researcher → writer → editor → publisher.
#   Here, we have 3 agents doing different TYPES of analysis on the SAME data.
#   The manager decides how to interleave their work.
# ---------------------------------------------------------------------------

analyst = Agent(
    role="Market Analyst",
    goal=(
        "Analyze each competitor's market position, target audience, "
        "and business model. Identify where each company sits in the "
        "competitive landscape."
    ),
    backstory=(
        "You are a senior market analyst with deep experience in "
        "competitive intelligence. You look beyond surface-level features "
        "and understand the strategic positioning of companies — their "
        "target markets, value propositions, and growth trajectories. "
        "You are data-driven and objective in your assessments."
    ),
    llm=LLM_MODEL,
    verbose=True,
)

comparator = Agent(
    role="Feature Comparator",
    goal=(
        "Build detailed side-by-side comparisons of products, features, "
        "and pricing across all competitors. Make differences clear and "
        "easy to understand."
    ),
    backstory=(
        "You are a product comparison specialist. You excel at creating "
        "clear, structured comparisons that highlight differences and "
        "similarities. You organize information into easy-to-scan formats "
        "and always note when data is missing or uncertain."
    ),
    llm=LLM_MODEL,
    verbose=True,
)

strategist = Agent(
    role="Strategic Advisor",
    goal=(
        "Identify strategic opportunities, threats, and actionable "
        "recommendations based on the competitive analysis. Your insights "
        "should help the target company make better decisions."
    ),
    backstory=(
        "You are a strategy consultant. You synthesize market analysis "
        "and product comparisons into actionable strategic recommendations. "
        "You think about competitive advantages, market gaps, and future "
        "trends. Your recommendations are specific and prioritized, not "
        "generic platitudes."
    ),
    llm=LLM_MODEL,
    verbose=True,
)


# ---------------------------------------------------------------------------
# BUILDING THE CREW
#
# The factory function receives the Research Crew's output as a string.
# This string contains the competitor profiles — the Analysis Crew reads
# it as context in the task descriptions.
#
# Data flow:
#   Research Crew → crew.kickoff() returns CrewOutput
#   → CrewOutput.raw is a text representation of the results
#   → we pass that text into the Analysis Crew's task descriptions
#   → the Analysis LLM reads it as context
#
# This is the "contract" between crews: text in, structured output out.
# The Flow (flow.py) handles passing data between crews.
# ---------------------------------------------------------------------------


def build_analysis_crew(
    target_company: str,
    competitor_profiles_text: str,
) -> Crew:
    """Build and return the Analysis Crew.

    Args:
        target_company: The company being analyzed.
        competitor_profiles_text: Raw text output from the Research Crew
            containing structured competitor profiles.
    """

    # Shared context that all tasks reference
    context_block = (
        f"TARGET COMPANY: {target_company}\n\n"
        f"COMPETITOR PROFILES (from research phase):\n"
        f"{competitor_profiles_text}"
    )

    market_task = Task(
        description=(
            f"Analyze the market positioning of {target_company} and its "
            f"competitors based on the research data below.\n\n"
            f"{context_block}\n\n"
            f"For each company, assess:\n"
            f"  - Target audience and market segment\n"
            f"  - Core value proposition\n"
            f"  - Business model and revenue approach\n"
            f"  - Market position (leader, challenger, niche, etc.)\n\n"
            f"Be specific and evidence-based — reference details from the "
            f"research data, not general knowledge."
        ),
        expected_output=(
            "A detailed market positioning analysis for each company, "
            "covering target audience, value proposition, business model, "
            "and competitive position."
        ),
        agent=analyst,  # suggestion — the manager may reassign
    )

    comparison_task = Task(
        description=(
            f"Create a detailed side-by-side comparison of {target_company} "
            f"and its competitors.\n\n"
            f"{context_block}\n\n"
            f"Compare across these dimensions:\n"
            f"  - Products and services offered\n"
            f"  - Key features and capabilities\n"
            f"  - Pricing (if available)\n"
            f"  - Strengths vs weaknesses\n\n"
            f"Present the comparison in a clear, structured format. "
            f"Highlight where each company has an advantage."
        ),
        expected_output=(
            "A structured comparison of all companies across products, "
            "features, pricing, and strengths/weaknesses."
        ),
        agent=comparator,
    )

    strategy_task = Task(
        description=(
            f"Based on the market analysis and feature comparison, provide "
            f"strategic recommendations for {target_company}.\n\n"
            f"{context_block}\n\n"
            f"Identify:\n"
            f"  - Key competitive advantages {target_company} should leverage\n"
            f"  - Threats from competitors that need to be addressed\n"
            f"  - Market gaps or opportunities\n"
            f"  - 3-5 specific, actionable recommendations\n\n"
            f"Each recommendation should be concrete, not generic. "
            f"For example, 'Invest in X because competitor Y is weak there' "
            f"is better than 'Innovate more'."
        ),
        expected_output=(
            "Strategic insights and 3-5 actionable recommendations for "
            "the target company based on competitive analysis."
        ),
        agent=strategist,
        # DISABLED: output_pydantic=CompetitiveAnalysis
        #
        # Even in hierarchical mode, the crew's output is determined by the
        # last task in the list (the manager ensures all tasks complete).
        #
        # Instructor + Ollama incompatibility prevents output_pydantic from
        # working. See research_crew.py for the full explanation.
        # With OpenAI/Anthropic, you'd re-enable:
        #   output_pydantic=CompetitiveAnalysis,
    )

    # ---------------------------------------------------------------------------
    # CREW ASSEMBLY — HIERARCHICAL MODE
    #
    # The key difference from the Research Crew:
    #
    #   Research:  Crew(process=Process.sequential)
    #             → no manager, tasks run in fixed order
    #
    #   Analysis: Crew(process=Process.hierarchical, manager_llm=LLM_MODEL)
    #             → a manager LLM coordinates the agents
    #
    # The manager_llm parameter is REQUIRED for hierarchical mode.
    # This creates an invisible "manager agent" that:
    #   1. Reads all task descriptions
    #   2. Decides which agent handles which task (can override agent= hints)
    #   3. Reviews intermediate results
    #   4. Can send work back for revision if quality is insufficient
    #   5. Ensures all tasks complete before returning the final result
    #
    # In P3, the manager coordinated 5 agents through a pipeline.
    # Here, it coordinates 3 analysts working on the SAME data from
    # different angles. The manager might interleave their work:
    #   "Analyst, assess the market → Comparator, now compare features
    #    → wait, Analyst, revisit Company X → Strategist, synthesize."
    #
    # This adaptive coordination is why hierarchical is valuable for
    # analysis work — the insights from one agent inform the others.
    # ---------------------------------------------------------------------------

    crew = Crew(
        agents=[analyst, comparator, strategist],
        tasks=[market_task, comparison_task, strategy_task],
        process=Process.hierarchical,
        manager_llm=LLM_MODEL,  # ← REQUIRED for hierarchical mode
        # DISABLED: memory=True requires an embedding model (defaults to OpenAI).
        # To enable with Ollama:
        #   memory=True,
        #   embedder={"provider": "ollama", "config": {"model": "nomic-embed-text"}},
        memory=False,
        verbose=True,
    )

    return crew
