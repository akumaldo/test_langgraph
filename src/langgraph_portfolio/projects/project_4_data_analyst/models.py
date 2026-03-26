"""
Project 4 — Pydantic models for structured LLM output.

In Projects 1-3, the LLM returned freeform text and we had to parse it
ourselves (e.g., checking if the editor's response started with "APPROVED").

Here we use Pydantic models with LangChain's with_structured_output().
Instead of:
    response = llm.invoke([system, human])
    text = response.content  # raw string, hope it's in the right format

We do:
    structured_llm = llm.with_structured_output(AnalysisPlan)
    plan = structured_llm.invoke([system, human])  # → AnalysisPlan object
    plan.steps[0].code  # typed access, no parsing

How with_structured_output() works under the hood:
  1. Takes your Pydantic model and converts it to a JSON schema
  2. Adds instructions to the LLM prompt: "Return valid JSON matching: {schema}"
  3. Parses the LLM's JSON response through Pydantic
  4. If validation fails, raises an error (the caller can retry)

The Field(description=...) values are included in the schema sent to the LLM.
They act as instructions: "this field should contain...". Good descriptions
= better LLM output.
"""

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. ANALYSIS PLAN
#
# The LLM reads the user's question + CSV column info, then produces a
# step-by-step plan. Each step has a description AND executable code.
#
# Why include code in the plan?
#   Because the LLM can see ALL the column names and data types upfront.
#   If we asked it to write code later (without the plan), it might
#   forget which columns exist or use wrong names. Planning + coding
#   together gives better results.
# ---------------------------------------------------------------------------

class AnalysisStep(BaseModel):
    """A single step in the analysis plan."""
    description: str = Field(
        description="What this step does in plain English, e.g. 'Calculate average grade by study time'"
    )
    code: str = Field(
        description=(
            "Python/pandas code to execute this step. "
            "The DataFrame is available as 'df'. "
            "Store the result in a variable called 'result'. "
            "Use print() to show output."
        )
    )


class AnalysisPlan(BaseModel):
    """The planner's output: a structured analysis plan.

    The LLM fills this out after seeing the question and the dataset info.
    Each step is a concrete, executable action — not vague instructions.
    """
    reasoning: str = Field(
        description="Brief explanation of the analysis approach and why these steps answer the question"
    )
    steps: list[AnalysisStep] = Field(
        description="3-5 concrete analysis steps, each with description and executable pandas code"
    )


# ---------------------------------------------------------------------------
# 2. CODE FIX
#
# When code execution fails (wrong column name, syntax error, etc.),
# the LLM sees the error and produces a fix.
#
# This is the error-recovery loop:
#   execute_step → error → LLM produces CodeFix → execute fixed code
#
# Compare to P3's editor loop:
#   P3: editor JUDGES quality → sends back to writer (subjective)
#   P4: system DETECTS error → sends to fixer (mechanical)
#
# The explanation field is important for learning — it tells you WHY
# the code failed, not just what the fix is.
# ---------------------------------------------------------------------------

class CodeFix(BaseModel):
    """The fixer's output: corrected code after an execution error."""
    error_analysis: str = Field(
        description="What went wrong and why (e.g., 'Column name was misspelled')"
    )
    fixed_code: str = Field(
        description=(
            "The corrected Python/pandas code. "
            "The DataFrame is available as 'df'. "
            "Store the result in a variable called 'result'. "
            "Use print() to show output."
        )
    )


# ---------------------------------------------------------------------------
# 3. CHART SPECIFICATION
#
# After analysis is done, the LLM decides what charts would best
# visualize the findings. It returns a typed spec — not raw matplotlib
# code — so we can validate the chart type, columns, etc.
#
# The code field contains the actual matplotlib code to run.
# We keep it separate from the metadata (title, chart_type) so we
# can log/display the intent even if the code fails.
# ---------------------------------------------------------------------------

class ChartSpec(BaseModel):
    """Specification for a single chart to generate."""
    title: str = Field(description="Chart title, e.g. 'Average Grade by Study Time'")
    chart_type: str = Field(description="One of: line, bar, scatter, histogram, box, heatmap")
    description: str = Field(description="What this chart shows and why it's relevant")
    code: str = Field(
        description=(
            "Complete matplotlib code to generate and save this chart. "
            "The DataFrame is available as 'df'. "
            "Use plt from matplotlib.pyplot (already imported). "
            "End with plt.savefig(chart_path) where chart_path is provided. "
            "Always call plt.close() at the end to free memory."
        )
    )


class ChartPlan(BaseModel):
    """The chart planner's output: what visualizations to create."""
    charts: list[ChartSpec] = Field(
        description="1-3 charts that best visualize the analysis findings"
    )


# ---------------------------------------------------------------------------
# 4. ANALYSIS REPORT
#
# The final output: a structured report summarizing everything.
# This replaces P3's publisher agent — but instead of formatting an
# article, we're summarizing data analysis with references to charts.
# ---------------------------------------------------------------------------

class ReportSection(BaseModel):
    """A section of the analysis report."""
    heading: str = Field(description="Section heading")
    content: str = Field(description="The analysis findings for this section")


class AnalysisReport(BaseModel):
    """The reporter's output: a complete analysis report."""
    title: str = Field(description="Report title")
    summary: str = Field(description="2-3 sentence executive summary of key findings")
    sections: list[ReportSection] = Field(description="Detailed findings, one section per analysis step")
    chart_references: list[str] = Field(
        description="List of chart file paths generated during analysis"
    )
