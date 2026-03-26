"""
Project 7 — CLI Entry Point.

This is the user-facing interface. It collects the debate topic and
runs the debate arena.

Compare entry points across projects:
  P1-P3: input("Enter topic: ") → graph.invoke() or crew.kickoff()
  P4:    input("Enter question: ") → graph.invoke()
  P5:    input("Enter message: ") → graph.stream()
  P6:    multi-field input → flow.kickoff()
  P7:    input("Enter topic: ") → run_debate()

P7's entry point is simpler because AG2's API is simpler — you don't
need to build a graph or define tasks. You create agents, put them in
a GroupChat, and start the conversation.
"""

from .debate import run_debate


def main():
    """Run the Multi-Agent Debate Arena."""
    print("\n" + "=" * 60)
    print("  Multi-Agent Debate Arena")
    print("  Project 7 — AG2 GroupChat + Speaker Selection")
    print("=" * 60 + "\n")

    topic = input("  Enter a debate topic: ").strip()
    if not topic:
        print("  A topic is required.")
        return

    print(f"\n  Starting debate on: '{topic}'")
    print("  4 agents: Moderator, Debater_Pro, Debater_Con, Judge")
    print("  The debate will run for several rounds, then the Judge scores.\n")

    # --- Run the debate ---
    result = run_debate(topic)

    # --- Display results ---
    print("\n" + "=" * 60)
    print("  DEBATE RESULTS")
    print("=" * 60)
    print(f"\n  Topic:   {result.topic}")
    print(f"  Rounds:  {result.total_rounds}")
    print(f"  Winner:  {result.winner}")

    if result.scores:
        print("\n  Scores:")
        for score in result.scores:
            print(f"    {score.name}: {score.score}/10 — {score.reasoning}")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
