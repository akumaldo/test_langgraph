"""
Project 6 — CLI Entry Point.

This is the user-facing interface. It collects inputs (target company,
competitors, optional URLs) and kicks off the IntelligenceFlow.

Compare to previous projects:
  P1-P3: simple input("Enter topic: ") → single function call
  P4:    input("Enter question: ") → graph.invoke()
  P5:    input("Enter message: ") → graph.stream() (streaming)
  P6:    multi-field input → flow.kickoff()

The kickoff() method is CrewAI Flows' equivalent of LangGraph's invoke().
It starts the pipeline, passing the inputs dict into the FlowState.

Behind the scenes:
  1. Flow creates an IntelFlowState from the inputs dict
  2. @start() method runs (research_phase)
  3. @listen() methods run in order (analysis → report → save)
  4. The final report is printed and saved to a markdown file

The entire pipeline runs synchronously — kickoff() blocks until all
phases complete. This can take several minutes with 7 agents and a
local LLM, so we print progress messages at each phase.
"""

from .flow import IntelligenceFlow


def main():
    """Run the Competitive Intelligence System."""
    print("\n" + "=" * 60)
    print("  🔍 COMPETITIVE INTELLIGENCE SYSTEM")
    print("  Project 6 — CrewAI Flows + Multi-Crew Pipeline")
    print("=" * 60 + "\n")

    # --- Collect inputs ---

    target_company = input("  Enter target company: ").strip()
    if not target_company:
        print("  ❌ Target company is required.")
        return

    competitors_input = input("  Enter competitors (comma-separated): ").strip()
    if not competitors_input:
        print("  ❌ At least one competitor is required.")
        return

    competitors = [c.strip() for c in competitors_input.split(",") if c.strip()]

    urls_input = input(
        "  Enter URLs to research (comma-separated, or press Enter to skip): "
    ).strip()
    urls = [u.strip() for u in urls_input.split(",") if u.strip()] if urls_input else []

    # --- Confirm and run ---

    print(f"\n  Target:      {target_company}")
    print(f"  Competitors: {', '.join(competitors)}")
    print(f"  URLs:        {len(urls)} provided" if urls else "  URLs:        none (agents will find their own)")
    print(f"\n  Starting intelligence pipeline...")
    print(f"  This will run 3 crews with 7 agents — it may take a few minutes.\n")

    # --- Kick off the flow ---
    # flow.kickoff() is like graph.invoke() in LangGraph.
    # The inputs dict populates the FlowState fields.

    flow = IntelligenceFlow()
    flow.kickoff(inputs={
        "target_company": target_company,
        "competitors": competitors,
        "urls": urls,
    })

    print("\n  ✅ Intelligence pipeline complete.\n")


if __name__ == "__main__":
    main()
