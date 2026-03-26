# Project 10 — Framework Showdown: Comparison Report

**Question**: What are the main uses for AI orchestration in industry today?

**Frameworks tested**: ag2, beeai, crewai, langgraph, llamaindex

## Automated Metrics

| Framework | Time (s) | Lines of Code | Output Complete | Structured Output |
|-----------|----------|---------------|-----------------|-------------------|
| langgraph | 27.07 | 103 | Yes | Yes |
| crewai | 129.53 | 99 | Yes | Yes |
| ag2 | 132.91 | 88 | Yes | Yes |
| beeai | 46.41 | 88 | Partial | No |
| llamaindex | 177.71 | 78 | Yes | Yes |

## Output Summaries

### langgraph

**Summary**: AI orchestration in industry primarily enables multi-agent systems to coordinate complex workflows across financial services, healthcare, manufacturing, and customer service sectors, improving efficiency and accuracy through specialized agent collaboration.

**Findings**:
- Financial institutions use AI orchestration for fraud detection, algorithmic trading, and regulatory compliance through multi-agent systems
- Healthcare organizations leverage AI orchestration for diagnostics, drug discovery, and hospital operations optimization
- Manufacturing embraces AI orchestration for predictive maintenance, supply chain coordination, and quality assurance
- Customer service orchestration moves beyond simple chatbots to coordinated multi-agent systems handling complex interactions
- Orchestration reduces false positives by 40% in fraud detection, cuts drug discovery time from months to days, and improves bed turnover rates by 20%
- Multi-agent systems catch 15% more early-stage conditions in healthcare diagnostics than single-model systems
- Predictive maintenance reduces unplanned downtime by up to 50% through coordinated sensor data and scheduling agents

**Citations**:
- Industry Report 2024 - Financial Services
- Tech Review 2024 - Healthcare
- Industry Analysis 2024 - Manufacturing
- Business Tech Report 2024 - Customer Service

### crewai

**Summary**: AI orchestration demonstrates a value proposition centered on specialization, coordination, and efficiency across four key industries. Adoption prioritizes operational cost reduction and risk mitigation through specialized multi-agent architectures.

**Findings**:
- Reduction of false positives by 40% in financial fraud detection using specialized multi-agent pipelines
- Increased diagnostic accuracy in healthcare catching 15% more early-stage conditions than single models
- Achievement of sub-second decision-making speed in financial algorithmic trading
- Reduction of unplanned downtime by 50% in manufacturing through predictive maintenance
- Increase in equipment lifespan by 20-30% via orchestration of visual inspection agents
- Improvement in bed turnover rates in healthcare by 20% via operational optimization
- 35% faster resolution times in customer service ticket routing
- 15-20% reduction in customer churn through proactive sentiment analysis

**Citations**:
- Analysis of AI Orchestration Use Cases in Industry - Source
- Industry Reports and Technical Reviews - Source

### ag2

**Summary**: AI orchestration is primarily used in financial services for fraud detection, algorithmic trading, and regulatory compliance through multi-agent systems. It is also widely adopted in healthcare to improve diagnostics, accelerate drug discovery, and optimize patient flow.

**Findings**:
- Financial institutions utilize multi-agent systems to coordinate complex workflows in fraud detection and algorithmic trading.
- Healthcare organizations leverage AI orchestration to enhance diagnostics, accelerate drug discovery, and optimize patient flow.

**Citations**:
- AI Orchestration in Financial Services - Industry Report 2024
- AI Orchestration in Healthcare - Tech Review 2024

### beeai

**Raw output** (structured parsing failed):

```

```

### llamaindex

**Summary**: The provided documents demonstrate a clear trend toward moving beyond single-model AI solutions toward multi-agent orchestration. Industries utilize a layered architecture where specialized agents perform specific tasks to maximize efficiency and scalability.

**Findings**:
- Specialization is key as no single AI model handles all tasks efficiently.
- Orchestration enables real-time collaboration for complex workflows across industries.
- Systems optimize resource allocation in manufacturing, healthcare, and customer service sectors.
- The framework ensures auditability and traceability through a central orchestration layer.
- Agents prioritize collaboration over competition for better end-to-end operational stability.

**Citations**:
- Document 1 - Source
- Document 2 - Source
- Document 3 - Source
- Document 4 - Source

## Qualitative Notes

*Fill in these sections after running all frameworks.*

### langgraph

- **Control level**: <!-- how much did you design the flow vs framework decided? -->
- **Debuggability**: <!-- how easy was it to trace what happened? -->
- **Boilerplate**: <!-- how much setup code vs actual logic? -->
- **Constraints**: <!-- what didn't work or required workarounds? -->

### crewai

- **Control level**: <!-- how much did you design the flow vs framework decided? -->
- **Debuggability**: <!-- how easy was it to trace what happened? -->
- **Boilerplate**: <!-- how much setup code vs actual logic? -->
- **Constraints**: <!-- what didn't work or required workarounds? -->

### ag2

- **Control level**: <!-- how much did you design the flow vs framework decided? -->
- **Debuggability**: <!-- how easy was it to trace what happened? -->
- **Boilerplate**: <!-- how much setup code vs actual logic? -->
- **Constraints**: <!-- what didn't work or required workarounds? -->

### beeai

- **Control level**: <!-- how much did you design the flow vs framework decided? -->
- **Debuggability**: <!-- how easy was it to trace what happened? -->
- **Boilerplate**: <!-- how much setup code vs actual logic? -->
- **Constraints**: <!-- what didn't work or required workarounds? -->

### llamaindex

- **Control level**: <!-- how much did you design the flow vs framework decided? -->
- **Debuggability**: <!-- how easy was it to trace what happened? -->
- **Boilerplate**: <!-- how much setup code vs actual logic? -->
- **Constraints**: <!-- what didn't work or required workarounds? -->

## Known Limitations

- **Model**: Using `qwen3.5:2b` for all frameworks. Production would use larger models or API-based LLMs. Chosen for fair comparison and to avoid timeout issues experienced in earlier projects.
- **CrewAI tool use**: Broken with Ollama due to Instructor incompatibility. Tools called from Python instead of by agents. Would work fine with OpenAI API.
- **CrewAI structured output**: `output_pydantic` disabled for same reason. Returns plain text.
- **BeeAI/CrewAI dep conflict**: Can't coexist in same Poetry venv (json-repair version clash). Must install separately via pip.
- **Knowledge base**: Static keyword search, not vector/semantic retrieval. Chosen for simplicity and equal footing across frameworks.
- **Single question**: One research question for speed. More questions would give stronger comparison signal.
- **Structured output**: Only native in LangGraph (`with_structured_output`). Others rely on prompt-based JSON — a real framework difference worth noting.
- **Local LLM context limits**: Prompts kept under ~15k tokens to avoid empty responses from Ollama.
