"""
Project 3 — Writing Team using real CrewAI.

This is the SAME writing team as langgraph_writing_team.py, but built with
CrewAI instead of LangGraph. Same 5 roles, same goal — but a completely
different way of thinking about orchestration.

We'll highlight the differences at every step so you can compare frameworks.

Key difference up front:
  LangGraph = you BUILD the graph (nodes, edges, conditions)
  CrewAI    = you DESCRIBE the team (agents, tasks, process) and let the
              framework figure out the flow

Think of it like this:
  LangGraph is like drawing a flowchart and then executing it.
  CrewAI is like hiring a team, giving them job descriptions, and letting
  a manager coordinate who does what.
"""

from pydantic import BaseModel, Field

from crewai import Agent, Crew, Process, Task


# ---------------------------------------------------------------------------
# 0. STRUCTURED OUTPUT MODELS
#
# Pydantic models for structured LLM output are NOT framework-specific.
# Both LangGraph and CrewAI support them:
#
#   LangGraph (via LangChain):
#     llm_structured = llm.with_structured_output(Outline)
#     result = llm_structured.invoke([system, human])  # → Outline object
#
#   CrewAI:
#     Task(..., output_pydantic=Outline)  # CrewAI handles the rest
#
# The difference is WHERE you configure it:
#   LangGraph: on the LLM call, inside each node function (you control it)
#   CrewAI:    on the Task definition (the framework controls it)
#
# In our LangGraph version, we used plain strings and manual parsing
# (e.g., parsing "APPROVED" from the editor's first word). That was a
# teaching choice, not a framework limitation. We could have used
# with_structured_output() there too.
#
# Why Pydantic models?
#
#   Without them, each task returns a raw string — just whatever the LLM
#   felt like saying. With a Pydantic model, the framework instructs the
#   LLM to return JSON that matches the schema. This gives us:
#     1. Typed, predictable output (not freeform text)
#     2. Validation (Pydantic checks the structure)
#     3. Easy access to fields (result.sections, not "parse the string")
#
# How it works under the hood:
#   The framework takes your Pydantic model, converts it to a JSON schema,
#   and adds it to the LLM prompt: "You MUST return valid JSON matching
#   this schema: {...}". The LLM's response is then parsed and validated
#   by Pydantic. If it doesn't match, the framework retries.
#
# Note: the Field(description=...) matters! These descriptions are
# included in the prompt sent to the LLM, so the agent knows what each
# field means. Think of it as labeling each box on the form.
#
# The REAL difference between frameworks here is about STATE:
#
#   LangGraph: all agents share ONE state (WritingState TypedDict).
#     Every agent can read/write any field. It's a shared desk.
#     Structured output is per-LLM-call, then YOU merge it into state.
#
#   CrewAI: each task gets its OWN output model. It's like each agent
#     fills out a DIFFERENT form, and the manager passes the completed
#     form to the next agent as context. You don't manage the state.
# ---------------------------------------------------------------------------


class Section(BaseModel):
    """A single section of the article outline."""
    title: str = Field(description="The section heading")
    description: str = Field(description="One sentence explaining what this section covers")


class Outline(BaseModel):
    """The planner's output: a structured article outline."""
    sections: list[Section] = Field(description="3-4 sections that make up the article")


class SectionNotes(BaseModel):
    """Research notes for one section."""
    section_title: str = Field(description="Which section these notes are for")
    bullet_points: list[str] = Field(description="2-3 key facts or points for this section")


class ResearchNotes(BaseModel):
    """The researcher's output: organized notes for each section."""
    notes: list[SectionNotes] = Field(description="Research notes grouped by section")


class DraftSection(BaseModel):
    """A single section of the written article."""
    heading: str = Field(description="The section heading")
    body: str = Field(description="The full written content for this section")


class ArticleDraft(BaseModel):
    """The writer's output: a complete article draft."""
    title: str = Field(description="The article title")
    sections: list[DraftSection] = Field(description="The written sections of the article")


class EditorialReview(BaseModel):
    """The editor's output: a quality assessment with a clear decision.

    This replaces the LangGraph pattern where we parsed 'APPROVED' or
    'REVISION_NEEDED' from the first word of the editor's response.
    With Pydantic, we get a proper boolean field — no string parsing needed.
    """
    approved: bool = Field(description="True if the article is ready to publish, False if it needs revisions")
    feedback: str = Field(description="Specific feedback: praise if approved, revision instructions if not")


class PublishedArticle(BaseModel):
    """The publisher's output: the final, formatted article."""
    title: str = Field(description="The final article title")
    content: str = Field(description="The complete, formatted article ready for publication")

# ---------------------------------------------------------------------------
# 1. THE LLM
#
# In LangGraph, we created one ChatOllama instance and called it directly
# in each node function:
#
#     llm = ChatOllama(model="qwen3.5:35b", temperature=0)
#     response = llm.invoke([system, human])
#
# In CrewAI, you don't call the LLM yourself. Instead, you tell each Agent
# which LLM to use, and CrewAI handles the calling internally.
#
# CrewAI uses LiteLLM under the hood, which means model names follow a
# provider/model format. For Ollama, the prefix is "ollama/".
#
# You CAN also pass a LangChain LLM object — CrewAI supports both styles.
# We'll use the string format since it's CrewAI's native approach.
# ---------------------------------------------------------------------------

LLM_MODEL = "ollama/qwen3.5:35b"


# ---------------------------------------------------------------------------
# 2. AGENTS — THE TEAM MEMBERS
#
# In LangGraph, "agents" were just functions. Each function had a system
# prompt baked into it:
#
#     def planner(state: WritingState) -> dict:
#         system = SystemMessage(content="You are an article planner...")
#         human = HumanMessage(content=f"Create an outline for: {state['topic']}")
#         response = llm.invoke([system, human])
#         return {"outline": response.content}
#
# In CrewAI, agents are OBJECTS with identity. You define:
#   - role: their job title (like the system prompt's "You are a...")
#   - goal: what they're trying to achieve
#   - backstory: extra context about how they should behave
#
# CrewAI builds the system prompt FOR you from these fields. You never
# write SystemMessage/HumanMessage — the framework handles that.
#
# Analogy:
#   LangGraph agent = a function with a prompt baked in (you control everything)
#   CrewAI agent    = a persona with a job description (the framework prompts it)
#
# Notice we DON'T define what each agent DOES here — that comes in the Tasks.
# Agents are WHO, Tasks are WHAT.
# ---------------------------------------------------------------------------

planner = Agent(
    role="Article Planner",
    goal="Create clear, structured outlines that guide the writing process",
    backstory=(
        "You are a senior content strategist. Given a topic, you break it "
        "down into 3-4 logical sections, each with a clear purpose. Your "
        "outlines are known for being focused and actionable."
    ),
    llm=LLM_MODEL,
    verbose=True,   # prints what the agent is thinking (great for learning)
)

researcher = Agent(
    role="Research Analyst",
    goal="Gather accurate, relevant information for each section of the outline",
    backstory=(
        "You are a thorough researcher. Given an outline, you find key facts, "
        "important comparisons, and supporting evidence for each section. "
        "You present your findings as concise bullet points — 2-3 per section."
    ),
    llm=LLM_MODEL,
    verbose=True,
)

writer = Agent(
    role="Article Writer",
    goal="Write clear, well-structured articles based on outlines and research",
    backstory=(
        "You are a skilled technical writer. You take outlines and research "
        "notes and turn them into polished, readable articles. When you "
        "receive revision feedback, you incorporate it thoughtfully."
    ),
    llm=LLM_MODEL,
    verbose=True,
)

editor = Agent(
    role="Senior Editor",
    goal="Ensure articles are high quality, well-structured, and ready to publish",
    backstory=(
        "You are an experienced editor. You review drafts for clarity, "
        "structure, accuracy, and completeness. You give specific, actionable "
        "feedback. You are thorough but fair — you approve work that meets "
        "your standards and request revisions when needed."
    ),
    llm=LLM_MODEL,
    verbose=True,
)

publisher = Agent(
    role="Content Publisher",
    goal="Format and finalize articles for publication",
    backstory=(
        "You handle final formatting. You take editor-approved articles, "
        "add proper titles, clean up structure, and produce the final "
        "publishable version. You don't change content — just polish the form."
    ),
    llm=LLM_MODEL,
    verbose=True,
)


# ---------------------------------------------------------------------------
# 3. TASKS — THE WORK TO DO
#
# This is where the big conceptual shift happens.
#
# In LangGraph, each node function did TWO things:
#   1. Defined WHAT to do (via the prompt)
#   2. Was wired into the graph at a specific POSITION (via edges)
#
# In CrewAI, these are SEPARATED:
#   - Agents define WHO (role, personality)
#   - Tasks define WHAT (description, expected output)
#   - The Process defines HOW tasks flow (sequential or hierarchical)
#
# Each Task has:
#   - description: what needs to be done (like the HumanMessage content)
#   - expected_output: what the result should look like
#   - agent: who is assigned to this task (optional in hierarchical mode!)
#
# IMPORTANT: In hierarchical mode, agent assignment is a SUGGESTION.
# The manager agent can reassign tasks if it thinks someone else should
# handle it. This is fundamentally different from LangGraph where the
# graph edges are FIXED — the planner ALWAYS runs before the researcher.
#
# We use {topic} as a placeholder — CrewAI will substitute it when we
# pass inputs to the crew's kickoff() method.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Adding output_pydantic to each Task.
#
# This is the key addition. Before, each task returned a raw string —
# whatever the LLM felt like writing. Now we tell CrewAI: "the output
# of this task MUST match this Pydantic model."
#
# What CrewAI does with it:
#   1. Converts the model to a JSON schema
#   2. Appends to the prompt: "Return valid JSON matching: {schema}"
#   3. Parses the LLM's response through Pydantic
#   4. If validation fails, retries automatically
#
# The result: task.output.pydantic gives you a typed Python object,
# not a string. You can do result.sections[0].title instead of
# parsing text.
#
# Compare to LangGraph:
#   LangGraph: each node returns a dict → merged into shared state
#              {"outline": response.content, "messages": [...]}
#   CrewAI:    each task returns a Pydantic object → passed as context
#              to the next task by the manager
#
# Same idea (structured intermediate results), different mechanism.
# ---------------------------------------------------------------------------

plan_task = Task(
    description=(
        "Create a structured outline for an article about: {topic}\n\n"
        "The outline should have 3-4 sections, each with a title and a "
        "one-sentence description of what it will cover."
    ),
    expected_output="A structured outline with 3-4 sections, each having a title and description.",
    agent=planner,
    output_pydantic=Outline,  # ← forces the planner to return an Outline object
)

research_task = Task(
    description=(
        "Research each section from the outline. For each section, provide "
        "2-3 bullet points with key facts, important comparisons, or "
        "supporting evidence."
    ),
    expected_output="Research notes with 2-3 bullet points per section from the outline.",
    agent=researcher,
    output_pydantic=ResearchNotes,  # ← forces structured notes per section
)

write_task = Task(
    description=(
        "Write a complete article using the outline and research notes. "
        "Follow the outline structure, incorporate the research, and "
        "produce a clear, well-written article."
    ),
    expected_output="A complete, well-structured article based on the outline and research.",
    agent=writer,
    output_pydantic=ArticleDraft,  # ← forces a title + list of sections
)

edit_task = Task(
    description=(
        "Review the article draft. Check that it:\n"
        "1. Follows the outline structure\n"
        "2. Is clear and well-written\n"
        "3. Covers the key points from the research\n\n"
        "If it needs improvements, provide specific feedback. "
        "If it's ready, approve it for publication."
    ),
    expected_output=(
        "A JSON object with 'approved' (true/false) and 'feedback' "
        "(specific praise or revision instructions)."
    ),
    agent=editor,
    # In our LangGraph version, we parsed "APPROVED" / "REVISION_NEEDED"
    # from the first word of the editor's response (string parsing).
    # With Pydantic output, we get a proper boolean field instead.
    #
    # Note: this is NOT a CrewAI-only feature. LangGraph can do the same
    # thing via LangChain's with_structured_output():
    #
    #     llm_structured = llm.with_structured_output(EditorialReview)
    #     result = llm_structured.invoke([system, human])
    #     # result is an EditorialReview object, not a string
    #
    # We just didn't use it in our LangGraph version because we were
    # learning concepts step by step. Structured output is a LangChain
    # feature available to BOTH frameworks.
    output_pydantic=EditorialReview,
)

publish_task = Task(
    description=(
        "Take the editor-approved article and produce a final, publishable "
        "version. Add a title if missing, clean up formatting, but do not "
        "change the content."
    ),
    expected_output="The final, polished article with title and formatted content.",
    agent=publisher,
    output_pydantic=PublishedArticle,  # ← the final typed output
)


# ---------------------------------------------------------------------------
# 4. THE CREW — ORCHESTRATION
#
# This is the equivalent of LangGraph's build_writing_team() function,
# but the difference is HUGE.
#
# LangGraph (explicit control flow):
#
#     graph = StateGraph(WritingState)
#     graph.add_node("planner", planner)
#     graph.add_node("writer", writer)
#     graph.add_node("editor", editor)
#     graph.add_edge("planner", "researcher")
#     graph.add_edge("researcher", "writer")
#     graph.add_edge("writer", "editor")
#     graph.add_conditional_edges("editor", after_editor)  # YOUR logic
#     graph.add_edge("publisher", END)
#
# CrewAI (delegated control flow):
#
#     crew = Crew(agents=[...], tasks=[...], process=Process.hierarchical)
#
# That's it. No edges. No routing function. No conditional logic.
#
# PROCESS TYPES:
#
# Process.sequential:
#   Tasks run in order: task1 → task2 → task3 → ...
#   Similar to LangGraph's linear edges, but WITHOUT the ability to loop.
#   The output of each task is automatically passed as context to the next.
#   This is simple but rigid — no editor loop possible.
#
# Process.hierarchical:
#   A MANAGER AGENT is created automatically by CrewAI. This manager:
#   1. Reads the list of tasks and agents
#   2. Decides which agent should work on which task
#   3. Reviews intermediate results
#   4. Can send work BACK to an agent for revision
#   5. Decides when the overall goal is met
#
#   This is the closest analogy to our LangGraph editor loop, but the
#   control logic lives in the MANAGER'S reasoning, not in your code.
#
# THE BIG TRADE-OFF:
#
#   LangGraph: You write after_editor() with explicit conditions.
#              MAX_REVISIONS is a number YOU set. The routing is deterministic.
#              You know EXACTLY when the loop will stop.
#
#   CrewAI:    The manager decides when quality is good enough.
#              The revision behavior emerges from the manager's judgment.
#              You DON'T have a hard MAX_REVISIONS (though you can set
#              max_iter on the crew to limit total iterations).
#
#   Neither is "better" — they're different philosophies:
#     LangGraph = maximum control, you design every path
#     CrewAI    = maximum delegation, the AI manages itself
# ---------------------------------------------------------------------------


def build_writing_crew() -> Crew:
    """Build and return the writing team crew.

    Compare this to build_writing_team() in langgraph_writing_team.py.
    There we had 15+ lines of graph wiring. Here it's just a constructor.
    """
    return Crew(
        agents=[planner, researcher, writer, editor, publisher],
        tasks=[plan_task, research_task, write_task, edit_task, publish_task],
        process=Process.hierarchical,

        # The manager LLM — this is the "invisible 6th agent" that
        # coordinates the team. In LangGraph, YOU were the manager
        # (by writing the graph). Here, an LLM is the manager.
        manager_llm=LLM_MODEL,

        # verbose=True shows the manager's delegation decisions.
        # This is like watching the graph execution trace in LangGraph,
        # but instead of "node X → node Y", you see the manager thinking
        # "I'll assign this to the writer" or "the editor wants revisions,
        # sending back to writer".
        verbose=True,
    )


# ---------------------------------------------------------------------------
# 5. RUN IT
#
# In LangGraph, we ran the graph with:
#
#     app = build_writing_team()
#     result = app.invoke({
#         "messages": [], "topic": "...", "outline": "", ...
#     })
#
# We had to initialize EVERY state field — LangGraph needs the full TypedDict.
#
# In CrewAI, we just pass the dynamic inputs. The framework handles
# intermediate state internally — you never see "outline", "draft", etc.
# as separate fields. Each task's output is passed as context to subsequent
# tasks automatically.
#
# This is simpler, but it also means you have LESS VISIBILITY into
# intermediate results. In LangGraph, you could inspect state["outline"]
# after running. In CrewAI, you get the final output.
#
# To see intermediate results, you rely on verbose=True (console output)
# or task callbacks.
# ---------------------------------------------------------------------------

def run_writing_crew(topic: str) -> PublishedArticle:
    """Run the writing crew and return the final article as a typed object.

    Compare the return types:
      LangGraph: returns the FULL state dict (you can inspect every field)
                 result["outline"], result["draft"], result["final_article"]
      CrewAI:    returns a CrewOutput; with output_pydantic, the last task's
                 output is a typed Pydantic object you can access by field

    Before structured output:
      result.raw → a raw string (whatever the LLM said)

    After structured output:
      result.pydantic → a PublishedArticle(title=..., content=...)
      You can do result.pydantic.title and result.pydantic.content

    We still don't get the INTERMEDIATE results (outline, research, etc.)
    as easily as LangGraph's state dict. But the FINAL output is now typed
    and predictable.
    """
    crew = build_writing_crew()

    # kickoff() is CrewAI's equivalent of app.invoke()
    # The inputs dict fills in {topic} placeholders in task descriptions.
    result = crew.kickoff(inputs={"topic": topic})

    # With output_pydantic on the last task, result.pydantic gives us
    # a PublishedArticle object — not a raw string.
    return result.pydantic


# ---------------------------------------------------------------------------
# SUMMARY: FRAMEWORK COMPARISON
#
# | Aspect                | LangGraph              | CrewAI                    |
# |-----------------------|------------------------|---------------------------|
# | Flow definition       | Explicit graph edges   | Process type (seq/hier)   |
# | State                 | TypedDict you manage   | Internal, hidden          |
# | Revision loop         | Conditional edge +     | Manager agent decides     |
# |                       | routing function       |                           |
# | Loop safety           | MAX_REVISIONS (code)   | max_iter (config)         |
# | Intermediate results  | Inspect state fields   | verbose logs              |
# | Agent identity        | Just a function        | Object with role/goal     |
# | Control philosophy    | You are the manager    | AI is the manager         |
# | Predictability        | Deterministic routing  | LLM-driven routing        |
# | Flexibility           | Maximum (any graph)    | Constrained to patterns   |
# | Boilerplate           | More (edges, state)    | Less (declarative)        |
# | Output structure      | Dict fields (manual)   | Pydantic models (typed)   |
#
# When to use which?
#   LangGraph: when you need precise control, deterministic behavior,
#              or complex custom flows (branches, parallel, sub-graphs).
#   CrewAI:    when you want to spin up a team quickly, the task flow
#              is straightforward, and you're OK with the AI managing itself.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    article = run_writing_crew(
        topic="LangGraph versus CrewAI for building AI agent teams"
    )

    # Now we have a typed object, not a raw string!
    print("\n=== Final Article (CrewAI) ===\n")
    print(f"Title: {article.title}\n")
    print(article.content)
