"""
Project 6 — Intelligence Flow (CrewAI Flows).

This is the CORE of P6 — the Flow that chains our three crews together:

  Research Crew → Analysis Crew → Report Crew → Save to file

In P5 (LangGraph), you learned SUB-GRAPHS — graphs inside graphs. You
composed a support bot from billing, technical, and returns sub-graphs:

    main_graph.add_node("billing", billing_subgraph)
    main_graph.add_edge("router", "billing")

CrewAI Flows are the SAME idea — composing independent units into a
larger pipeline — but with a completely different syntax:

    @start()
    def research_phase(self): ...

    @listen(research_phase)
    def analysis_phase(self): ...

Instead of adding nodes and edges to a graph, you write methods and
decorate them. The decorators ARE the edges:
    @start()              = the START edge in LangGraph
    @listen(prev_method)  = add_edge("prev", "this") in LangGraph

The result is the same — a pipeline with defined execution order —
but the code looks very different.

Key comparison:
    LangGraph sub-graphs:
      - Explicit: you see every node and edge
      - Flexible: conditional edges, cycles, complex routing
      - Verbose: more code for the same pipeline

    CrewAI Flows:
      - Declarative: decorators describe the pipeline
      - Simpler: less code for linear/branching pipelines
      - Limited: harder to do complex routing (but supports @router)

Flows also support features we won't use in P6 but are good to know:
    @router()  — conditional routing (like LangGraph conditional_edge)
    Parallel @listen — multiple methods listening to the same source
    Flow.kickoff() — starts the pipeline (like graph.invoke())
"""

import os
from datetime import datetime

from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel, Field

from .analysis_crew import build_analysis_crew
# DISABLED: not used since output_pydantic was removed from crews.
# Models are kept in models.py for reference and future re-enablement.
# from .models import IntelligenceReport
from .report_crew import build_report_crew
from .research_crew import build_research_crew
from .tools import WebScraperTool


# ---------------------------------------------------------------------------
# FLOW STATE
#
# In LangGraph, state is a TypedDict that all nodes read from and write to:
#
#   class SupportState(TypedDict):
#       messages: Annotated[list, add_messages]
#       department: str
#       ...
#
# In CrewAI Flows, state is a Pydantic BaseModel (or a plain dict) that
# persists across all flow methods:
#
#   class MyFlowState(BaseModel):
#       result: str = ""
#
# You access it via self.state inside any flow method.
#
# Key difference:
#   LangGraph: nodes return update dicts, the framework MERGES them into state
#   CrewAI:    you mutate self.state directly (self.state.field = value)
#
# The FlowState below carries data between our three crews. Each phase
# writes its output to self.state, and the next phase reads from it.
#
# We store crew outputs as STRINGS (result.raw), not Pydantic objects.
# Why? Because the next crew reads them as TEXT in task descriptions.
# The LLM doesn't need deserialized objects — it reads text naturally.
# ---------------------------------------------------------------------------


class IntelFlowState(BaseModel):
    """State that flows between all phases of the intelligence pipeline.

    Compare to LangGraph's TypedDict:
      - Both define the shape of shared state
      - LangGraph merges return dicts; Flows use direct mutation
      - Both persist across the entire pipeline execution
    """
    target_company: str = Field(default="", description="The company being analyzed")
    competitors: list[str] = Field(default_factory=list, description="List of competitor names")
    urls: list[str] = Field(default_factory=list, description="Optional URLs to scrape")

    # Crew outputs stored as text — each phase writes here, next phase reads
    competitor_profiles: str = Field(default="", description="Research Crew output")
    analysis: str = Field(default="", description="Analysis Crew output")
    report: str = Field(default="", description="Report Crew output")


# ---------------------------------------------------------------------------
# THE FLOW
#
# A Flow is a class with methods decorated by @start and @listen.
# Think of it as a graph defined by decorators:
#
#   @start()           → this method runs first (the entry point)
#   @listen(method)    → this method runs after 'method' completes
#
# The execution order is:
#   research_phase → analysis_phase → report_phase → save_report
#
# This is equivalent to a LangGraph with four nodes in sequence:
#   graph.add_edge(START, "research")
#   graph.add_edge("research", "analysis")
#   graph.add_edge("analysis", "report")
#   graph.add_edge("report", "save")
#
# But written with ~50% less code.
#
# How data flows between methods:
#   1. Each @listen method receives the RETURN VALUE of the previous method
#      as its second argument (after self)
#   2. We ALSO store outputs in self.state for redundancy and debugging
#   3. The Flow calls methods automatically — you never call them yourself
# ---------------------------------------------------------------------------


class IntelligenceFlow(Flow[IntelFlowState]):
    """The main intelligence pipeline.

    Chains three crews together:
      Research → Analysis → Report → Save

    Usage:
        flow = IntelligenceFlow()
        flow.kickoff(inputs={
            "target_company": "Anthropic",
            "competitors": ["OpenAI", "Google DeepMind"],
            "urls": []  # optional
        })
    """

    # ----- PHASE 1: RESEARCH -----

    @start()
    def research_phase(self):
        """Scrape websites, then run the Research Crew to organize the data.

        @start() marks this as the entry point — the first method to run.
        Compare to LangGraph: graph.add_edge(START, "research")

        This method does TWO things:
          1. Scrape web pages using WebScraperTool (deterministic Python)
          2. Pass scraped content to the Research Crew (LLM reasoning)

        Why scrape here instead of in the Crew?
          When agents use tools autonomously, the scraped content
          accumulates in the agent's context (40k+ tokens for 3-4
          Wikipedia pages). The next agent then fails because the
          context is too large for structured output generation.

          By scraping in Python, we control the content size — each
          page is truncated to ~2000 chars, keeping total context
          manageable for the LLM.

        This is also a good Flow pattern: not every step needs to be
        a Crew. You can mix plain Python with Crew executions, just
        like LangGraph nodes can do anything (not just LLM calls).
        """
        print("\n" + "=" * 60)
        print("  PHASE 1: RESEARCH")
        print("  Scraping websites and organizing competitor data...")
        print("=" * 60 + "\n")

        # --- Step 1: Scrape web pages (deterministic Python) ---
        scraped_content = self._scrape_companies()

        # --- Step 2: Run Research Crew on scraped content (LLM reasoning) ---
        crew = build_research_crew(
            target_company=self.state.target_company,
            competitors=self.state.competitors,
            scraped_content=scraped_content,
        )

        result = crew.kickoff()

        # Store the raw text output in state for the next phase.
        # result.raw is the text representation of the crew's output.
        # In sequential mode, this is the last task's output as text.
        self.state.competitor_profiles = result.raw

        print("\n  ✓ Research phase complete.\n")

        # Return the output — @listen methods receive this as a parameter
        return result.raw

    def _scrape_companies(self) -> str:
        """Scrape web pages for all companies using WebScraperTool.

        We call the tool directly from Python — same BaseTool class from
        tools.py, just invoked as regular code instead of through an agent.

        Content size control: we use a smaller max_content_length (2000 chars)
        to keep total context manageable. With 4 companies, that's ~8000 chars
        of scraped content — well within the LLM's comfort zone.
        """
        tool = WebScraperTool(max_content_length=2000)
        all_companies = [self.state.target_company] + self.state.competitors
        results = []

        if self.state.urls:
            print(f"  Scraping {len(self.state.urls)} provided URLs...")
            for url in self.state.urls:
                print(f"    → {url}")
                content = tool._run(url)
                results.append(content)
        else:
            print(f"  No URLs provided — scraping Wikipedia for {len(all_companies)} companies...")
            for company in all_companies:
                wiki_name = company.replace(" ", "_")
                url = f"https://en.wikipedia.org/wiki/{wiki_name}"
                print(f"    → {company}: {url}")
                content = tool._run(url)
                results.append(f"=== {company} ===\n{content}")

        scraped_text = "\n\n---\n\n".join(results)
        print(f"\n  Scraped {len(results)} pages ({len(scraped_text)} chars total).\n")
        return scraped_text

    # ----- PHASE 2: ANALYSIS -----

    @listen(research_phase)
    def analysis_phase(self, research_output: str):
        """Run the Analysis Crew on the research results.

        @listen(research_phase) means: run this AFTER research_phase completes.
        Compare to LangGraph: graph.add_edge("research", "analysis")

        The research_output parameter is the RETURN VALUE of research_phase.
        This is how data flows between methods in a Flow — through return
        values AND through self.state (we use both for clarity).
        """
        print("\n" + "=" * 60)
        print("  PHASE 2: ANALYSIS")
        print("  Analyzing competitive landscape...")
        print("=" * 60 + "\n")

        crew = build_analysis_crew(
            target_company=self.state.target_company,
            competitor_profiles_text=research_output,
        )

        result = crew.kickoff()
        self.state.analysis = result.raw

        print("\n  ✓ Analysis phase complete.\n")

        return result.raw

    # ----- PHASE 3: REPORT -----

    @listen(analysis_phase)
    def report_phase(self, analysis_output: str):
        """Run the Report Crew to produce the final intelligence briefing.

        @listen(analysis_phase) → runs after analysis.
        Compare to LangGraph: graph.add_edge("analysis", "report")
        """
        print("\n" + "=" * 60)
        print("  PHASE 3: REPORT")
        print("  Generating intelligence report...")
        print("=" * 60 + "\n")

        crew = build_report_crew(
            target_company=self.state.target_company,
            analysis_text=analysis_output,
        )

        result = crew.kickoff()
        self.state.report = result.raw

        print("\n  ✓ Report phase complete.\n")

        return result.raw

    # ----- PHASE 4: SAVE (not a crew — plain Python) -----

    @listen(report_phase)
    def save_report(self, report_output: str):
        """Save the report to console and markdown file.

        This is NOT a crew — it's a plain Python method. This shows an
        important Flow feature: not every step needs to be a crew execution.
        You can mix crew steps with regular Python logic.

        Compare to LangGraph where every node is a function that takes state
        and returns an update dict. Same flexibility — nodes can do anything,
        not just LLM calls.

        @listen(report_phase) → runs after the Report Crew finishes.
        """
        print("\n" + "=" * 60)
        print("  INTELLIGENCE REPORT")
        print("=" * 60)
        print(report_output)
        print("=" * 60 + "\n")

        # Save to markdown file
        self._save_markdown(report_output)

        return report_output

    def _save_markdown(self, report_text: str):
        """Save the report as a markdown file.

        Not a flow step — just a helper method. Only methods with @start
        or @listen decorators are flow steps.
        """
        # Create reports directory if it doesn't exist
        reports_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "reports")
        reports_dir = os.path.normpath(reports_dir)
        os.makedirs(reports_dir, exist_ok=True)

        # Build filename from company names
        competitors_str = "_vs_".join(self.state.competitors[:3])  # limit length
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.state.target_company}_vs_{competitors_str}_{timestamp}.md"

        # Clean filename (replace spaces and special chars)
        filename = filename.replace(" ", "_").replace("/", "_")

        filepath = os.path.join(reports_dir, filename)

        with open(filepath, "w") as f:
            f.write(f"# Competitive Intelligence Report\n\n")
            f.write(f"**Target:** {self.state.target_company}\n")
            f.write(f"**Competitors:** {', '.join(self.state.competitors)}\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
            f.write("---\n\n")
            f.write(report_text)

        print(f"  📄 Report saved to: {filepath}\n")
