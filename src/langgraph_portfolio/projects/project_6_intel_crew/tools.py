"""
Project 6 — Custom CrewAI tools.

In P2 (LangGraph), you built tools using LangChain's @tool decorator:

    @tool
    def search_knowledge_base(query: str) -> str:
        ...

Then you bound them to the LLM and used a ToolNode to execute them:

    llm_with_tools = llm.bind_tools([search_knowledge_base])
    tool_node = ToolNode([search_knowledge_base])

In CrewAI, tools work differently. Instead of decorating functions, you
create a CLASS that extends BaseTool:

    class MyTool(BaseTool):
        name: str = "My Tool"
        description: str = "What this tool does"

        def _run(self, argument: str) -> str:
            # Do the work and return a string
            ...

Then you assign tools to agents (not to the LLM directly):

    agent = Agent(
        role="Researcher",
        tools=[MyTool()],   # ← agent can use this tool
        ...
    )

Key differences:
  - LangGraph: YOU decide when tools run (via ToolNode in the graph)
  - CrewAI: the AGENT decides when to use its tools (autonomous)
  - LangGraph tools are functions; CrewAI tools are classes
  - Both return strings — the LLM reads the result and decides what to do

Why classes? CrewAI tools can have configuration (attributes on the class),
validation, and more complex initialization. The class pattern also makes
it clear that a tool is a reusable, self-contained unit.
"""

import requests
from bs4 import BeautifulSoup
from crewai.tools import BaseTool


# ---------------------------------------------------------------------------
# WEB SCRAPER TOOL
#
# This is the first REAL tool in the portfolio that does something external —
# it fetches a live web page. In P2, the search_knowledge_base tool queried
# a local in-memory document store. Here, we're hitting the actual internet.
#
# The tool returns a STRING (not a Pydantic model). Why? Because tools in
# CrewAI communicate with agents via text. The agent reads the tool's output
# as context and decides what to do with it. The structured output (Pydantic)
# happens at the TASK level, not the tool level.
# ---------------------------------------------------------------------------


class WebScraperTool(BaseTool):
    """Scrapes a website URL and returns clean text content.

    The agent calls this tool with a URL, and gets back the page's title
    and text content — stripped of HTML, scripts, styles, and navigation.

    Error handling: on failure (timeout, bad status code, invalid URL),
    the tool returns an error message STRING instead of raising an
    exception. This lets the agent decide what to do — try a different
    URL, skip this source, or work with partial data.
    """

    # These class attributes define how the tool appears to the agent.
    # The agent reads the name and description to decide WHEN to use it.
    # A good description is critical — it's the agent's only guidance
    # on what the tool does and when to call it.
    name: str = "Web Scraper"
    description: str = (
        "Scrapes a website URL and returns clean text content. "
        "Input: a valid URL (e.g., 'https://example.com'). "
        "Output: the page title and main text content, cleaned of HTML tags. "
        "Use this to gather information about companies from their websites, "
        "Wikipedia pages, news articles, or any public web page."
    )

    # Maximum characters to return. Web pages can be huge — we truncate
    # to avoid overwhelming the LLM's context window. 4000 chars is
    # roughly 1000 tokens, which is enough for the agent to extract
    # key information without losing important context.
    max_content_length: int = 4000

    def _run(self, url: str) -> str:
        """Scrape a URL and return clean text.

        This is the method CrewAI calls when the agent uses the tool.
        It MUST be named _run (with underscore) — that's CrewAI's convention.

        Compare to LangGraph where the function name IS the tool name:
            @tool
            def search_knowledge_base(query: str) -> str:  # ← called directly

        Here, the class has a name attribute, and _run is the implementation.
        """
        try:
            # Step 1: Fetch the page
            # We set a User-Agent header because many websites block requests
            # without one (returning 403 Forbidden). This is a common gotcha
            # when building web scrapers.
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # raises HTTPError for 4xx/5xx

            # Step 2: Parse with BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")

            # Step 3: Remove elements that add noise, not content
            # Scripts, styles, nav bars, footers — these are HTML structure,
            # not the information we want. Removing them gives us cleaner text.
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            # Step 4: Extract the title
            title = soup.title.string.strip() if soup.title and soup.title.string else "No title"

            # Step 5: Get clean text
            # get_text() extracts all text from the remaining HTML.
            # separator="\n" puts each block element on its own line.
            # strip=True removes leading/trailing whitespace from each piece.
            text = soup.get_text(separator="\n", strip=True)

            # Step 6: Truncate to avoid overwhelming the LLM
            if len(text) > self.max_content_length:
                text = text[: self.max_content_length] + "\n\n[Content truncated]"

            return f"Page: {title}\nURL: {url}\n\n{text}"

        except requests.exceptions.Timeout:
            return f"Error scraping {url}: Request timed out after 10 seconds."

        except requests.exceptions.HTTPError as e:
            return f"Error scraping {url}: HTTP {e.response.status_code}."

        except requests.exceptions.ConnectionError:
            return f"Error scraping {url}: Could not connect to the server."

        except requests.exceptions.RequestException as e:
            return f"Error scraping {url}: {e}"

        except Exception as e:
            return f"Error scraping {url}: Unexpected error — {e}"
