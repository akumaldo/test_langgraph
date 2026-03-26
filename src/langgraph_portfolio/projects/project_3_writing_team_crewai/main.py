from __future__ import annotations


def main(topic: str | None = None) -> int:
    """Run the CrewAI writing team.

    By default runs the real CrewAI version (crewai_writing_team.py).
    """
    from .crewai_writing_team import run_writing_crew

    article = run_writing_crew(topic or "LangGraph versus CrewAI for portfolio agents")

    # article is a PublishedArticle Pydantic object — typed, not a raw string
    print("\n=== Final Article (CrewAI) ===\n")
    print(f"Title: {article.title}\n")
    print(article.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
