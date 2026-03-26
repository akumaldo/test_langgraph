FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    APP_MODULE=langgraph_portfolio.projects.project_1_chatbot.main:main

WORKDIR /app

COPY pyproject.toml README.md sitecustomize.py ./
COPY src ./src
COPY project_1_chatbot ./project_1_chatbot
COPY project_2_research_agent ./project_2_research_agent
COPY project_3_writing_team_langgraph ./project_3_writing_team_langgraph
COPY project_3_writing_team_crewai ./project_3_writing_team_crewai
COPY project_4_data_analyst ./project_4_data_analyst
COPY project_5_capstone ./project_5_capstone
COPY scripts ./scripts
COPY docs ./docs

CMD ["python", "scripts/run_project.py"]
