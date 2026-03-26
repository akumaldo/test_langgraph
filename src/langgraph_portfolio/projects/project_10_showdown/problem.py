"""
PROJECT 10 — FRAMEWORK SHOWDOWN: SHARED PROBLEM DEFINITION
===========================================================

This file is the FOUNDATION that all 5 framework implementations share.
It defines:
  1. The knowledge base (same documents for every framework)
  2. The search function (same retrieval logic for every framework)
  3. The output schema (same target format for every framework)
  4. Result persistence (save/load so we can compare later)

WHY A SHARED PROBLEM FILE?
  The whole point of P10 is comparing frameworks on EQUAL FOOTING.
  If each framework had its own documents or search logic, we wouldn't
  know whether differences in output came from the framework or the data.
  By sharing everything except the orchestration, the framework IS the
  only variable.
"""

import json
import time
from pathlib import Path
from pydantic import BaseModel


# ── 1. RESEARCH QUESTION ────────────────────────────────────────────────
# One question, asked to all 5 frameworks. We keep it factual and broad
# enough that the KB documents can all contribute useful information.

RESEARCH_QUESTION = "What are the main uses for AI orchestration in industry today?"


# ── 2. KNOWLEDGE BASE ───────────────────────────────────────────────────
# 4 documents covering different industries. Each is ~400-600 words.
# These are hardcoded (not fetched from the web) so results are
# reproducible across runs — no network dependency, no content drift.

KNOWLEDGE_BASE = [
    {
        "id": "doc1",
        "title": "AI Orchestration in Financial Services",
        "source": "Industry Report 2024",
        "content": (
            "Financial institutions are among the earliest adopters of AI orchestration, "
            "using multi-agent systems to coordinate complex workflows that span fraud detection, "
            "algorithmic trading, and regulatory compliance. "
            "\n\n"
            "In fraud detection, orchestrated AI pipelines combine transaction monitoring agents "
            "with pattern recognition models. One agent scans real-time transactions for anomalies, "
            "a second agent cross-references against known fraud patterns, and a third agent decides "
            "whether to flag, block, or escalate the transaction. This pipeline approach reduces "
            "false positives by 40% compared to single-model systems because each agent specializes "
            "in one aspect of the decision. "
            "\n\n"
            "Algorithmic trading firms use AI orchestration to coordinate market analysis agents. "
            "A sentiment analysis agent processes news feeds and social media, a technical analysis "
            "agent evaluates price charts and indicators, and a risk management agent ensures "
            "positions stay within predefined limits. The orchestration layer decides which signals "
            "to act on and in what order, enabling sub-second decision-making that would be "
            "impossible with a single monolithic model. "
            "\n\n"
            "Regulatory compliance is another major use case. Banks deploy orchestrated agents to "
            "monitor transactions for anti-money-laundering (AML) violations, generate suspicious "
            "activity reports (SARs), and ensure adherence to Know Your Customer (KYC) requirements. "
            "The orchestration ensures that each compliance check happens in the correct sequence "
            "and that no step is skipped, which is critical for audit trails."
        ),
    },
    {
        "id": "doc2",
        "title": "AI Orchestration in Healthcare",
        "source": "Tech Review 2024",
        "content": (
            "Healthcare organizations are adopting AI orchestration to improve diagnostics, "
            "accelerate drug discovery, and optimize patient flow through hospitals. "
            "\n\n"
            "In diagnostics, orchestrated AI systems coordinate multiple specialized models. "
            "A radiology agent analyzes medical images (X-rays, MRIs, CT scans), a lab results "
            "agent interprets blood work and pathology reports, and a clinical history agent "
            "reviews the patient's medical records. An orchestration layer synthesizes these "
            "inputs into a preliminary diagnosis that assists physicians. Studies show this "
            "multi-agent approach catches 15% more early-stage conditions than single-model "
            "systems because no single model can be expert in all modalities. "
            "\n\n"
            "Drug discovery pipelines use AI orchestration to coordinate molecule screening, "
            "toxicity prediction, and clinical trial design. One agent generates candidate "
            "molecules, another predicts their binding affinity to target proteins, and a third "
            "evaluates potential side effects. This orchestrated pipeline can screen thousands "
            "of candidates in days rather than months, dramatically reducing the time from "
            "discovery to clinical trials. "
            "\n\n"
            "Hospital operations benefit from orchestrated AI for patient flow optimization. "
            "Admission prediction agents forecast incoming patient volumes, bed management agents "
            "optimize room assignments, and discharge planning agents coordinate follow-up care. "
            "The orchestration layer balances these competing priorities to minimize wait times "
            "and maximize resource utilization, with some hospitals reporting 20% improvements "
            "in bed turnover rates."
        ),
    },
    {
        "id": "doc3",
        "title": "AI Orchestration in Manufacturing",
        "source": "Industry Analysis 2024",
        "content": (
            "Manufacturing has embraced AI orchestration for predictive maintenance, supply chain "
            "optimization, and quality assurance, transforming traditional factory operations into "
            "intelligent, self-adjusting systems. "
            "\n\n"
            "Predictive maintenance uses orchestrated AI agents to monitor equipment health. "
            "Sensor data agents collect vibration, temperature, and pressure readings from "
            "factory equipment. Analysis agents detect patterns that indicate impending failures. "
            "Scheduling agents coordinate maintenance windows to minimize production disruption. "
            "This orchestrated approach reduces unplanned downtime by up to 50% and extends "
            "equipment lifespan by 20-30%, according to industry benchmarks. "
            "\n\n"
            "Supply chain orchestration coordinates procurement, inventory, and logistics agents. "
            "A demand forecasting agent predicts future material needs based on orders and market "
            "trends. A supplier management agent evaluates vendor reliability and pricing. A "
            "logistics agent optimizes shipping routes and warehouse allocation. The orchestration "
            "layer ensures these agents collaborate rather than compete — for example, preventing "
            "the procurement agent from ordering materials that the logistics agent cannot store. "
            "\n\n"
            "Quality assurance is increasingly orchestrated through AI. Visual inspection agents "
            "use computer vision to detect defects on production lines. Statistical process control "
            "agents monitor production parameters. Root cause analysis agents trace defects back "
            "to specific machines or materials. When orchestrated together, these agents can detect "
            "and respond to quality issues in real time, reducing defect rates by 30-40% compared "
            "to manual inspection processes."
        ),
    },
    {
        "id": "doc4",
        "title": "AI Orchestration in Customer Service",
        "source": "Business Tech Report 2024",
        "content": (
            "Customer service is being transformed by AI orchestration, moving beyond simple "
            "chatbots to coordinated multi-agent systems that handle complex customer interactions "
            "across channels. "
            "\n\n"
            "Modern customer service orchestration typically involves three layers of agents. "
            "First-line agents handle routine inquiries — checking order status, answering FAQs, "
            "and processing simple requests. Escalation agents detect when a conversation requires "
            "specialized knowledge (billing disputes, technical issues, returns) and route to the "
            "appropriate specialist agent. Supervisor agents monitor overall service quality and "
            "intervene when customer sentiment turns negative. "
            "\n\n"
            "Ticket routing has evolved from keyword matching to AI-orchestrated classification. "
            "A natural language understanding agent categorizes the customer's issue. A priority "
            "agent assesses urgency based on customer tier, issue severity, and SLA requirements. "
            "An assignment agent matches the ticket to the best available human or AI agent based "
            "on expertise and current workload. Companies using this orchestrated approach report "
            "35% faster resolution times and 25% higher customer satisfaction scores. "
            "\n\n"
            "Sentiment analysis orchestration monitors customer interactions across all channels — "
            "email, chat, phone, social media. Individual channel agents process interactions in "
            "their native format, while a central orchestration agent aggregates sentiment signals "
            "to detect emerging issues before they become widespread complaints. This proactive "
            "approach allows companies to address problems before they escalate, reducing customer "
            "churn by an estimated 15-20%."
        ),
    },
]


# ── 3. SEARCH FUNCTION ──────────────────────────────────────────────────
# Simple keyword search — no embeddings, no vector store.
#
# WHY KEYWORD SEARCH?
#   We want the FRAMEWORK to be the only variable. If we used embeddings,
#   each framework would need its own embedding setup (LangChain vs
#   LlamaIndex vs manual), introducing a second variable. Keyword search
#   is framework-agnostic — it's just Python.
#
# MATCHING LOGIC:
#   - Split query into words (case-insensitive)
#   - A document matches if its content contains at least one query word
#   - If no documents match, return ALL documents (so the agent always
#     has context to work with)
#   - Returns list of dicts with id, title, source, content

def search_knowledge_base(query: str) -> list[dict]:
    """Search the knowledge base for documents matching the query.

    Returns documents whose content contains at least one query term
    (case-insensitive). If no terms match, returns all documents.
    """
    query_terms = query.lower().split()

    if not query_terms:
        return KNOWLEDGE_BASE

    matches = []
    for doc in KNOWLEDGE_BASE:
        content_lower = doc["content"].lower()
        if any(term in content_lower for term in query_terms):
            matches.append(doc)

    # Fallback: if nothing matched, return everything
    return matches if matches else KNOWLEDGE_BASE


# ── 4. OUTPUT SCHEMA ─────────────────────────────────────────────────────
# The target output format for all frameworks.
#
# STRUCTURED OUTPUT SUPPORT BY FRAMEWORK:
#   - LangGraph: native via with_structured_output(ResearchReport)
#   - BeeAI: prompt-based JSON, parsed afterward
#   - CrewAI: plain text (output_pydantic broken with Ollama)
#   - AG2: plain text (conversational, no structured output)
#   - LlamaIndex: prompt-based JSON, parsed afterward
#
# For frameworks that can't produce Pydantic natively, we use
# parse_report() to attempt JSON parsing from the LLM's text output.

class ResearchReport(BaseModel):
    """Target output schema for the research assistant."""
    summary: str          # 2-3 sentence overview of findings
    findings: list[str]   # key findings as bullet points
    citations: list[str]  # document titles/sources used
    framework: str        # which framework produced this


def parse_report(text: str, framework: str) -> dict:
    """Try to parse a ResearchReport from LLM text output.

    Attempts JSON parsing first. If that fails, wraps the raw text
    as-is. Either way, returns a dict suitable for save_result().

    Returns:
        dict with keys: summary, findings, citations, framework,
              structured_output_success (bool), raw_text (if parsing failed)
    """
    # Try to extract JSON from the text (LLM might wrap it in markdown code blocks)
    json_text = text.strip()
    if "```json" in json_text:
        json_text = json_text.split("```json")[1].split("```")[0].strip()
    elif "```" in json_text:
        json_text = json_text.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(json_text)
        report = ResearchReport(
            summary=data.get("summary", ""),
            findings=data.get("findings", []),
            citations=data.get("citations", []),
            framework=framework,
        )
        return {
            **report.model_dump(),
            "structured_output_success": True,
        }
    except (json.JSONDecodeError, Exception):
        return {
            "summary": "",
            "findings": [],
            "citations": [],
            "framework": framework,
            "structured_output_success": False,
            "raw_text": text,
        }


# ── 5. RESULT PERSISTENCE ───────────────────────────────────────────────
# Each framework saves its result to results/{framework}.json.
# compare.py loads all results to generate the comparison report.

RESULTS_DIR = Path(__file__).parent / "results"


def save_result(framework: str, output: dict, elapsed: float) -> Path:
    """Save a framework's result to JSON.

    Args:
        framework: e.g. "langgraph", "crewai", "ag2", "beeai", "llamaindex"
        output: dict with summary, findings, citations, etc.
        elapsed: execution time in seconds
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    result = {
        "framework": framework,
        "elapsed_seconds": round(elapsed, 2),
        "question": RESEARCH_QUESTION,
        **output,
    }
    path = RESULTS_DIR / f"{framework}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return path


def load_results() -> dict[str, dict]:
    """Load all saved results from the results directory.

    Returns:
        dict mapping framework name -> result dict
    """
    results = {}
    if RESULTS_DIR.exists():
        for path in RESULTS_DIR.glob("*.json"):
            data = json.loads(path.read_text())
            results[data["framework"]] = data
    return results
