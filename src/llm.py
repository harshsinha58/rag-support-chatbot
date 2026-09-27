"""
Thin provider switch so the bot can run on Gemini, Claude, or OpenAI
by changing one env var — same pattern as the multi-backend setup
used in the earlier YouTube RAG CLI tool.
"""
from src import config


def get_llm():
    provider = config.LLM_PROVIDER

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-3.8-flash",
            google_api_key=config.GOOGLE_API_KEY,
            temperature=0.2,
        )

    elif provider == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            anthropic_api_key=config.ANTHROPIC_API_KEY,
            temperature=0.2,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=config.OPENAI_API_KEY,
            temperature=0.2,
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. Use 'gemini', 'claude', or 'openai'."
        )
