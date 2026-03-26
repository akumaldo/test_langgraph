"""LLM factory — local or cloud Ollama, env-configurable.

CONCEPT: Shared LLM factory
All frameworks in this project (LangGraph, CrewAI, AG2, LlamaIndex) need an LLM.
Instead of configuring each separately, we have ONE factory that reads env vars.

Environment variables:
  OLLAMA_BASE_URL — defaults to http://localhost:11434 (local Ollama)
                    set to https://ollama.com for cloud
  OLLAMA_API_KEY  — only needed for cloud; adds Bearer token header

The factory returns a ChatOllama instance (langchain_ollama), which works
directly with LangGraph and LangChain. For CrewAI and AG2, we extract
the model name and base_url to build their own config formats.
"""

import os

from langchain_ollama import ChatOllama

DEFAULT_MODEL = "qwen3.5:35b"
DEFAULT_BASE_URL = "http://localhost:11434"


def get_llm(
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
) -> ChatOllama:
    """Create a ChatOllama instance configured from environment.

    Returns a ChatOllama that works with LangGraph nodes directly.
    For CrewAI, use get_crewai_llm_string() instead.
    For AG2, use get_ag2_llm_config() instead.
    """
    base_url = os.environ.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL)
    api_key = os.environ.get("OLLAMA_API_KEY")

    kwargs: dict = {}
    if api_key:
        kwargs["client_kwargs"] = {
            "headers": {"Authorization": f"Bearer {api_key}"}
        }

    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        **kwargs,
    )


def get_crewai_llm_string(model: str = DEFAULT_MODEL) -> str:
    """Return the LiteLLM model string CrewAI expects.

    CrewAI uses LiteLLM under the hood. For Ollama, the format is
    'ollama/<model_name>'. CrewAI reads OLLAMA_HOST env var for the URL.
    """
    return f"ollama/{model}"


def get_ag2_llm_config(
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
) -> dict:
    """Return the LLM config dict AG2 expects.

    AG2 uses an OpenAI-compatible endpoint. Ollama exposes one at /v1.
    """
    base_url = os.environ.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL)
    api_key = os.environ.get("OLLAMA_API_KEY", "ollama")

    return {
        "config_list": [
            {
                "model": model,
                "base_url": f"{base_url}/v1",
                "api_key": api_key,
            }
        ],
        "temperature": temperature,
    }
