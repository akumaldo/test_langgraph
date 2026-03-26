"""
Project 4 — Autonomous Data Analyst using real LangGraph.

Building on everything from Projects 1-3:
  P1: state + nodes + conditional edges (fork shape)
  P2: tools + agent loop (roundabout shape)
  P3: multi-agent pipeline + quality-control loop
  P4: structured output + code execution + error-recovery loop + checkpointing

New things in this project:
  1. with_structured_output() — LLM returns Pydantic objects, not strings
  2. Code execution — the agent writes AND runs Python code on real data
  3. Error recovery — when code fails, the agent fixes it and retries
  4. Checkpointing — LangGraph saves state after every node automatically
  5. User input loop — the human asks questions, the agent analyzes
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, TypedDict

import pandas as pd
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, add_messages

from .models import AnalysisPlan, AnalysisReport, ChartPlan, ChartSpec, CodeFix
from .tools import execute_python, save_chart


# ---------------------------------------------------------------------------
# 1. STATE
#
# Compare to previous projects:
#
# P1 ChatbotState:   messages, intent, confidence
# P2 ResearchState:  messages, documents, citations
# P3 WritingState:   messages, topic, outline, draft, editor_feedback, ...
# P4 DataAnalystState: messages, question, csv data, plan, step tracking,
#                      errors, retries, charts, report, progress
#
# This is our MOST COMPLEX state. Why so many fields?
#
# Because data analysis is a multi-phase process:
#   1. Load data (csv_path, dataframe_info)
#   2. Plan (plan — the AnalysisPlan object, serialized)
#   3. Execute steps one by one (current_step_index, step_results)
#   4. Handle errors (last_error, retries)
#   5. Generate charts (charts — list of file paths)
#   6. Write report (report)
#
# Each phase reads from and writes to different fields. The state is
# the "shared desk" that tracks the entire analysis lifecycle.
#
# New concept: current_step_index
#   In P3, all agents ran once (except the writer in the revision loop).
#   Here, execute_step runs MULTIPLE TIMES — once per step in the plan.
#   We need an index to know which step we're on. This is like a for-loop
#   implemented as a graph cycle.
# ---------------------------------------------------------------------------

class DataAnalystState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str                    # what the user wants to analyze
    csv_path: str                    # path to the CSV file
    dataframe_info: str              # column names, dtypes, sample rows (for LLM context)
    plan: str                        # serialized AnalysisPlan JSON (from structured output)
    plan_steps_code: list[str]       # the code for each step (extracted from plan)
    plan_steps_desc: list[str]       # description for each step
    current_step_index: int          # which step we're executing now
    step_results: list[str]          # output from each executed step
    last_error: str                  # last execution error (empty if no error)
    retries: int                     # how many times we've retried the current step
    charts: list[str]                # file paths of generated charts
    report: str                      # the final markdown report
    progress: float                  # 0.0 to 1.0


# ---------------------------------------------------------------------------
# 2. LLM SETUP
#
# Same model as previous projects, but now we'll use it in TWO ways:
#
#   llm.invoke([...])                          → raw text response (like before)
#   llm.with_structured_output(Model).invoke() → Pydantic object response (NEW)
#
# with_structured_output() wraps the LLM so that:
#   1. It adds JSON schema instructions to the prompt
#   2. It parses the response through Pydantic
#   3. It returns a typed object, not a string
#
# We create the structured variants once and reuse them.
# ---------------------------------------------------------------------------

llm = ChatOllama(model="qwen3.5:35b", temperature=0)

# These are "specialized" versions of the same LLM.
# Same model, but each one is configured to return a specific Pydantic type.
llm_planner = llm.with_structured_output(AnalysisPlan)
llm_fixer = llm.with_structured_output(CodeFix)
llm_charter = llm.with_structured_output(ChartPlan)
llm_reporter = llm.with_structured_output(AnalysisReport)


# ---------------------------------------------------------------------------
# 3. NODE FUNCTIONS
#
# The flow:
#   load_data → plan_analysis → execute_step ←→ fix_and_retry
#                                    ↓ (when all steps done)
#                              generate_charts → write_report → END
# ---------------------------------------------------------------------------

# --- Limits ---
MAX_RETRIES = 2          # max retries per step (like P3's MAX_REVISIONS)
OUTPUT_DIR = Path("output/project_4")


def load_data(state: DataAnalystState) -> dict:
    """Node 1: Load the CSV and extract info for the LLM.

    The LLM can't "see" the CSV directly. We need to describe it in text:
    column names, data types, a few sample rows, and basic stats.
    This goes into dataframe_info, which every subsequent node uses as
    context so the LLM knows what data it's working with.
    """
    df = pd.read_csv(state["csv_path"])

    # Build a text description of the dataset for the LLM
    info_parts = [
        f"Dataset: {state['csv_path']}",
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
        f"\nColumns and types:\n{df.dtypes.to_string()}",
        f"\nFirst 3 rows:\n{df.head(3).to_string()}",
        f"\nBasic statistics:\n{df.describe().to_string()}",
    ]
    dataframe_info = "\n".join(info_parts)

    return {
        "dataframe_info": dataframe_info,
        "progress": 0.1,
        "messages": [AIMessage(content=f"[Data Loader] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")],
    }


def plan_analysis(state: DataAnalystState) -> dict:
    """Node 2: LLM creates a structured analysis plan.

    THIS is where with_structured_output() shines.

    Without it (like P1-P3):
        response = llm.invoke([system, human])
        text = response.content  # "Step 1: do X. Step 2: do Y. ..."
        # Now you have to PARSE this text to extract steps and code.
        # What if the LLM formats it differently? What if it forgets a step?

    With it:
        plan = llm_planner.invoke([system, human])
        # plan is an AnalysisPlan object with .steps, .reasoning
        # plan.steps[0].code is guaranteed to be a string
        # Pydantic validated the entire structure

    The LLM gets the dataset info (columns, types, sample rows) so it
    can write accurate pandas code that references real column names.
    """
    system = SystemMessage(content=(
        "You are a data analyst. Given a question about a dataset, create "
        "a step-by-step analysis plan with executable Python/pandas code.\n\n"
        "IMPORTANT:\n"
        "- The DataFrame is available as 'df'\n"
        "- Store each step's result in a variable called 'result'\n"
        "- Use print() to display findings\n"
        "- Use only columns that exist in the dataset\n"
        "- Keep each step focused on ONE thing\n"
        "- 3-5 steps total"
    ))

    human = HumanMessage(content=(
        f"QUESTION: {state['question']}\n\n"
        f"DATASET INFO:\n{state['dataframe_info']}"
    ))

    # This returns an AnalysisPlan object, NOT a string!
    plan: AnalysisPlan = llm_planner.invoke([system, human])

    # Extract the code and descriptions into separate lists for easy access
    # during the execute loop. We also serialize the full plan to JSON
    # for the state (TypedDict fields must be JSON-serializable).
    steps_code = [step.code for step in plan.steps]
    steps_desc = [step.description for step in plan.steps]

    return {
        "plan": plan.model_dump_json(),
        "plan_steps_code": steps_code,
        "plan_steps_desc": steps_desc,
        "current_step_index": 0,
        "step_results": [],
        "progress": 0.2,
        "messages": [AIMessage(content=(
            f"[Planner] Analysis plan created with {len(plan.steps)} steps:\n"
            + "\n".join(f"  {i+1}. {s.description}" for i, s in enumerate(plan.steps))
        ))],
    }


def execute_step(state: DataAnalystState) -> dict:
    """Node 3: Execute the current analysis step's code.

    This node runs MULTIPLE TIMES — once per step in the plan.
    It's like a for-loop implemented as a graph cycle:

        execute_step → (check: more steps?) → YES → execute_step
                                             → NO  → generate_charts

    The current_step_index tells us which step to run. After execution,
    we either store the result (success) or store the error (failure).
    The routing function after this node decides what happens next.
    """
    idx = state["current_step_index"]
    code = state["plan_steps_code"][idx]
    desc = state["plan_steps_desc"][idx]
    total = len(state["plan_steps_code"])

    # Load the DataFrame fresh for each execution
    df = pd.read_csv(state["csv_path"])

    # Run the code using our execute_python tool
    result = execute_python(code, df)

    if result["success"]:
        output = result["output"] or str(result["result"]) or "Step completed (no output)"
        return {
            "step_results": state["step_results"] + [f"Step {idx+1} ({desc}):\n{output}"],
            "current_step_index": idx + 1,
            "last_error": "",
            "retries": 0,  # reset retries for the next step
            "progress": 0.2 + (0.5 * (idx + 1) / total),
            "messages": [AIMessage(content=f"[Executor] Step {idx+1}/{total} completed: {desc}")],
        }
    else:
        return {
            "last_error": f"Step {idx+1} ({desc}) failed:\nCode:\n{code}\nError: {result['error']}",
            "progress": 0.2 + (0.5 * idx / total),
            "messages": [AIMessage(content=f"[Executor] Step {idx+1}/{total} FAILED: {result['error']}")],
        }


def fix_and_retry(state: DataAnalystState) -> dict:
    """Node 4: LLM fixes broken code and prepares for retry.

    This is the ERROR-RECOVERY loop — compare to P3's QUALITY-CONTROL loop:

    P3 (Writing Team):
        editor reads draft → "not good enough" → writer revises
        Decision based on: LLM judgment (subjective)

    P4 (Data Analyst):
        code runs → exception → fixer reads error → produces fixed code
        Decision based on: execution error (objective)

    The fixer sees:
      - The original code that failed
      - The error message
      - The dataset info (so it can check column names)

    It produces a CodeFix with the corrected code, which replaces the
    current step's code in plan_steps_code.
    """
    system = SystemMessage(content=(
        "You are a Python debugging expert. A pandas code snippet failed. "
        "Analyze the error and provide corrected code.\n\n"
        "IMPORTANT:\n"
        "- The DataFrame is available as 'df'\n"
        "- Store the result in a variable called 'result'\n"
        "- Use print() to display findings\n"
        "- Only use columns that exist in the dataset"
    ))

    human = HumanMessage(content=(
        f"FAILED CODE AND ERROR:\n{state['last_error']}\n\n"
        f"DATASET INFO:\n{state['dataframe_info']}"
    ))

    fix: CodeFix = llm_fixer.invoke([system, human])

    # Replace the failed step's code with the fixed version
    idx = state["current_step_index"]
    updated_code = list(state["plan_steps_code"])
    updated_code[idx] = fix.fixed_code

    return {
        "plan_steps_code": updated_code,
        "last_error": "",
        "retries": state["retries"] + 1,
        "messages": [AIMessage(content=(
            f"[Fixer] Fixed step {idx+1}: {fix.error_analysis}\n"
            f"Retry {state['retries']+1}/{MAX_RETRIES}"
        ))],
    }


def generate_charts(state: DataAnalystState) -> dict:
    """Node 5: LLM decides what charts to create, then generates them.

    The LLM sees:
      - The original question
      - The step results (what the analysis found)
      - The dataset info

    It produces a ChartPlan with 1-3 chart specs. Each spec has matplotlib
    code that we execute with save_chart().

    If a chart fails, we skip it (charts are nice-to-have, not critical).
    """
    # Make sure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    system = SystemMessage(content=(
        "You are a data visualization expert. Based on the analysis results, "
        "create 1-3 matplotlib charts that best illustrate the findings.\n\n"
        "IMPORTANT:\n"
        "- The DataFrame is available as 'df'\n"
        "- matplotlib.pyplot is available as 'plt'\n"
        "- Save each chart with: plt.savefig(chart_path)\n"
        "- Always call plt.close() at the end\n"
        "- Use clear titles and labels\n"
        "- Keep it simple and readable"
    ))

    results_text = "\n\n".join(state["step_results"])
    human = HumanMessage(content=(
        f"QUESTION: {state['question']}\n\n"
        f"ANALYSIS RESULTS:\n{results_text}\n\n"
        f"DATASET INFO:\n{state['dataframe_info']}"
    ))

    chart_plan: ChartPlan = llm_charter.invoke([system, human])

    # Execute each chart spec
    df = pd.read_csv(state["csv_path"])
    chart_paths: list[str] = []

    for i, spec in enumerate(chart_plan.charts):
        chart_path = OUTPUT_DIR / f"chart_{i+1}_{spec.chart_type}.png"
        result = save_chart(spec.code, df, chart_path)

        if result["success"]:
            chart_paths.append(result["chart_path"])
        else:
            # Chart failed — log it but keep going
            print(f"  Warning: Chart '{spec.title}' failed: {result['error']}")

    return {
        "charts": chart_paths,
        "progress": 0.85,
        "messages": [AIMessage(content=f"[Charts] Generated {len(chart_paths)} chart(s): {chart_paths}")],
    }


def write_report(state: DataAnalystState) -> dict:
    """Node 6: LLM writes a structured analysis report.

    The final node. It sees everything: question, plan, results, charts.
    It produces an AnalysisReport with title, summary, sections, and
    chart references.

    We format it as markdown so it's readable in the terminal and
    could be saved to a .md file.
    """
    system = SystemMessage(content=(
        "You are a data analyst writing a report. Summarize the analysis "
        "findings in a clear, structured report.\n\n"
        "Include:\n"
        "- A concise title\n"
        "- An executive summary (2-3 sentences)\n"
        "- One section per analysis step with findings\n"
        "- References to any charts generated"
    ))

    results_text = "\n\n".join(state["step_results"])
    charts_text = "\n".join(state["charts"]) if state["charts"] else "No charts generated"

    human = HumanMessage(content=(
        f"QUESTION: {state['question']}\n\n"
        f"ANALYSIS RESULTS:\n{results_text}\n\n"
        f"CHARTS GENERATED:\n{charts_text}"
    ))

    report: AnalysisReport = llm_reporter.invoke([system, human])

    # Format as markdown
    md_parts = [
        f"# {report.title}",
        f"\n**Summary:** {report.summary}\n",
    ]
    for section in report.sections:
        md_parts.append(f"## {section.heading}\n\n{section.content}\n")

    if report.chart_references:
        md_parts.append("## Charts\n")
        for path in report.chart_references:
            md_parts.append(f"- {path}")

    report_md = "\n".join(md_parts)

    return {
        "report": report_md,
        "progress": 1.0,
        "messages": [AIMessage(content="[Reporter] Analysis report complete.")],
    }


# ---------------------------------------------------------------------------
# 4. ROUTING LOGIC
#
# Two decision points (compare to previous projects):
#
# P1: confidence >= 0.7? → respond or clarify (ONE fork)
# P2: tool calls? → tools or END (ONE loop)
# P3: editor approved? → publisher or writer (ONE loop)
# P4: TWO routing decisions:
#     1. after_execute_step: error? more steps? done?
#     2. after_fix_and_retry: retries left? give up?
#
# This is our most complex routing. But each individual decision is simple.
# ---------------------------------------------------------------------------

def after_execute_step(state: DataAnalystState) -> str:
    """Three-way routing after executing a step.

    1. Error occurred → fix_and_retry (try to recover)
    2. No error, more steps → execute_step (continue the loop)
    3. No error, all steps done → generate_charts (move on)
    """
    if state["last_error"]:
        return "fix_and_retry"

    if state["current_step_index"] < len(state["plan_steps_code"]):
        return "execute_step"

    return "generate_charts"


def after_fix(state: DataAnalystState) -> str:
    """Two-way routing after fixing code.

    1. Retries left → execute_step (try the fixed code)
    2. Max retries hit → skip this step, move on

    Compare to P3's after_editor():
      P3: approved → publisher, rejected + under limit → writer, over limit → publisher
      P4: retries < max → execute_step, retries >= max → skip to next step or charts
    """
    if state["retries"] < MAX_RETRIES:
        return "execute_step"

    # Give up on this step — skip it and move to the next one
    # (or to charts if this was the last step)
    return "skip_step"


def skip_step(state: DataAnalystState) -> dict:
    """Skip a step that failed too many times and move on."""
    idx = state["current_step_index"]
    desc = state["plan_steps_desc"][idx]
    return {
        "step_results": state["step_results"] + [f"Step {idx+1} ({desc}): SKIPPED (failed after {MAX_RETRIES} retries)"],
        "current_step_index": idx + 1,
        "last_error": "",
        "retries": 0,
        "messages": [AIMessage(content=f"[Executor] Step {idx+1} skipped after {MAX_RETRIES} failed retries.")],
    }


# ---------------------------------------------------------------------------
# 5. GRAPH ASSEMBLY + CHECKPOINTING
#
# Compare graph assembly across all projects:
#
# P1: 3 nodes, 1 conditional edge, 2 regular edges (simple fork)
# P2: 2 nodes, 1 conditional edge, 1 regular edge (agent loop)
# P3: 5 nodes, 1 conditional edge, 5 regular edges (pipeline + loop)
# P4: 7 nodes, 2 conditional edges, 4 regular edges (pipeline + step loop + error recovery)
#
# New: MemorySaver checkpointer
#   In the scaffold, we manually saved/loaded state to JSON files:
#     save_checkpoint(state, path)  # 3 lines of JSON writing
#     load_checkpoint(path)         # 3 lines of JSON reading
#
#   With LangGraph's MemorySaver:
#     checkpointer = MemorySaver()  # that's it
#     graph.compile(checkpointer=checkpointer)
#
#   LangGraph snapshots the state AFTER EVERY NODE automatically.
#   If the process crashes mid-analysis, you can resume from the
#   last successful node. We don't use resume in this project,
#   but the infrastructure is there.
# ---------------------------------------------------------------------------

def build_data_analyst(checkpointer=None):
    """Build and compile the data analyst graph.

    Args:
        checkpointer: optional LangGraph checkpointer (e.g., MemorySaver).
    """
    graph = StateGraph(DataAnalystState)

    # --- Add all nodes ---
    graph.add_node("load_data", load_data)
    graph.add_node("plan_analysis", plan_analysis)
    graph.add_node("execute_step", execute_step)
    graph.add_node("fix_and_retry", fix_and_retry)
    graph.add_node("skip_step", skip_step)
    graph.add_node("generate_charts", generate_charts)
    graph.add_node("write_report", write_report)

    # --- Entry point ---
    graph.set_entry_point("load_data")

    # --- Pipeline edges (always go to the next node) ---
    graph.add_edge("load_data", "plan_analysis")
    graph.add_edge("plan_analysis", "execute_step")
    graph.add_edge("generate_charts", "write_report")
    graph.add_edge("write_report", END)

    # --- Conditional edges (the two decision points) ---

    # After executing a step: error? more steps? done?
    graph.add_conditional_edges("execute_step", after_execute_step)

    # After fixing code: retry or skip?
    graph.add_conditional_edges("fix_and_retry", after_fix)

    # After skipping: check if more steps (reuse same logic as after_execute_step)
    graph.add_conditional_edges("skip_step", after_execute_step)

    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# 6. RUN IT — Interactive loop with user input
#
# The graph handles ONE question at a time. The loop OUTSIDE the graph
# handles the conversation: ask → analyze → show results → ask again.
#
# This is the simplest form of human-in-the-loop:
#   Human controls WHAT to analyze (types the question)
#   Agent controls HOW to analyze (plans, executes, charts, reports)
#
# Each question is a fresh graph invocation — no state carries over
# between questions (the checkpointer stores per-run state, not
# cross-run state).
# ---------------------------------------------------------------------------

def save_report(result: dict) -> Path:
    """Save the analysis report as a markdown file with embedded charts.

    The report is saved to output/project_4/report_<timestamp>.md.
    Charts are referenced with relative paths (since they're in the same dir)
    using markdown image syntax: ![title](chart_file.png)
    """
    import datetime

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = OUTPUT_DIR / f"report_{timestamp}.md"

    # Start with the report content
    content = result["report"]

    # Replace chart absolute paths with relative markdown images
    # so the .md file renders the charts when opened in a viewer
    if result["charts"]:
        content += "\n\n---\n\n## Charts\n"
        for chart_path in result["charts"]:
            chart_name = Path(chart_path).name
            # Use relative path since charts are in the same output dir
            content += f"\n![{chart_name}]({chart_name})\n"

    report_path.write_text(content)
    return report_path


def run_analysis(question: str, csv_path: str) -> dict:
    """Run the data analyst graph for a single question.

    Returns the final state dict so the caller can extract the report,
    charts, and full message trace.
    """
    checkpointer = MemorySaver()
    app = build_data_analyst(checkpointer=checkpointer)

    initial_state = {
        "messages": [HumanMessage(content=question)],
        "question": question,
        "csv_path": csv_path,
        "dataframe_info": "",
        "plan": "",
        "plan_steps_code": [],
        "plan_steps_desc": [],
        "current_step_index": 0,
        "step_results": [],
        "last_error": "",
        "retries": 0,
        "charts": [],
        "report": "",
        "progress": 0.0,
    }

    # thread_id is required by MemorySaver to namespace checkpoints
    config = {"configurable": {"thread_id": "analysis-1"}}

    result = app.invoke(initial_state, config=config)
    return result


def interactive_loop(csv_path: str) -> None:
    """Run the data analyst in interactive mode.

    The user types questions, the agent analyzes, repeat.
    Type 'quit' or 'exit' to stop.
    """
    print(f"\n{'='*60}")
    print("  Autonomous Data Analyst — Project 4")
    print(f"  Dataset: {csv_path}")
    print(f"{'='*60}")
    print("\nAsk me anything about the dataset.")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ").strip()

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break

        print(f"\nAnalyzing: {question}\n")

        try:
            result = run_analysis(question, csv_path)

            # Show the execution trace
            print("\n--- Execution Trace ---")
            for msg in result["messages"]:
                if isinstance(msg, AIMessage):
                    print(f"  {msg.content}")
            print()

            # Show the report
            if result["report"]:
                print("--- Report ---")
                print(result["report"])

            # Show chart paths
            if result["charts"]:
                print("\n--- Charts ---")
                for path in result["charts"]:
                    print(f"  Saved: {path}")

            # Save the report as a .md file with embedded chart references
            report_path = save_report(result)
            print(f"\n--- Report saved to: {report_path} ---")

            print()

        except Exception as e:
            print(f"\nError during analysis: {e}\n")


def single_question(csv_path: str, question: str) -> None:
    """Run a single analysis question (non-interactive mode).

    Useful when running from environments that don't support input()
    (like Claude Code's ! command).
    """
    print(f"\nDataset: {csv_path}")
    print(f"Question: {question}\n")

    result = run_analysis(question, csv_path)

    # Show the execution trace
    print("\n--- Execution Trace ---")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"  {msg.content}")
    print()

    # Show the report
    if result["report"]:
        print("--- Report ---")
        print(result["report"])

    # Show chart paths
    if result["charts"]:
        print("\n--- Charts ---")
        for path in result["charts"]:
            print(f"  Saved: {path}")

    # Save the report
    report_path = save_report(result)
    print(f"\n--- Report saved to: {report_path} ---")


if __name__ == "__main__":
    import sys

    csv = "src/langgraph_portfolio/projects/project_4_data_analyst/student-mat.csv"

    # Support: python ... --question "What affects grades?"
    if "--question" in sys.argv:
        idx = sys.argv.index("--question")
        question = sys.argv[idx + 1]
        single_question(csv, question)
    else:
        interactive_loop(csv)
