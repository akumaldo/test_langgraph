# LangGraph vs CrewAI

Project 3 is intentionally duplicated so the same writing-team idea can be implemented two ways.

## LangGraph

- Best when workflow control matters
- Makes routing, retries, and checkpoints explicit
- Fits complex branching or human-in-the-loop logic
- Easier to inspect when debugging state transitions

## CrewAI

- Best when you want higher-level agent abstractions
- Reduces boilerplate for role/task-oriented collaboration
- Faster to express a straightforward team workflow
- Trades some explicit control for convenience

## Recommendation

Use LangGraph for the projects that depend on stateful orchestration, branching, and persistence. Use CrewAI where the goal is to compare agent collaboration ergonomics rather than to own every transition explicitly.

