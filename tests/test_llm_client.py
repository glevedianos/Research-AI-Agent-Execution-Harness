
from unittest.mock import MagicMock

import httpx
import pytest
from openai import BadRequestError, RateLimitError

from llm_client import LLMClient


def make_api_error(error_class, message, status_code):
    """
    Simulate an API error without contacting Groq.
    """
    request = httpx.Request(
        "POST",
        "https://api.groq.com/openai/v1/chat/completions"
    )

    response = httpx.Response(
        status_code,
        request=request,
        json={
            "error": {
                "message": message,
                "code": (
                    "tool_use_failed"
                    if status_code == 400
                    else "rate_limit_exceeded"
                ),
            }
        },
    )

    return error_class(
        message,
        response=response,
        body=response.json(),
    )


def create_client(monkeypatch):
    """
    Create an LLMClient using a fake API key
    and replace its API client with a mock.
    """
    monkeypatch.setenv("GROQ_API_KEY", "fake-test-key")

    client = LLMClient()
    client.client = MagicMock()

    return client


def test_recovers_from_invalid_tool_call(monkeypatch):
    """
    Test whether the LLM client recovers when
    Groq rejects a nonexistent tool call.
    """
    client = create_client(monkeypatch)

    invalid_tool_error = make_api_error(
        BadRequestError,
        "Tool call validation failed: "
        "attempted to call tool 'search_in_page' "
        "which was not in request.tools",
        400,
    )

    successful_message = MagicMock()
    successful_message.content = "Research completed."
    successful_message.tool_calls = []

    successful_response = MagicMock()
    successful_response.model = "openai/gpt-oss-120b"
    successful_response.usage = None
    successful_response.choices = [
        MagicMock(message=successful_message)
    ]

    # First request fails; second succeeds.
    client.client.chat.completions.create.side_effect = [
        invalid_tool_error,
        successful_response,
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    result = client.chat_with_tools(
        [{"role": "user", "content": "Research Python"}],
        tools,
    )

    assert result.content == "Research completed."

    assert (
        client.client.chat.completions.create.call_count
        == 2
    )


def test_rate_limit_eventually_raises(monkeypatch):
    """
    Test whether repeated rate-limit errors
    eventually stop the retry process.
    """
    client = create_client(monkeypatch)

    rate_limit_error = make_api_error(
        RateLimitError,
        "Rate limit reached. Please try again in 10s.",
        429,
    )

    # Every simulated API request fails.
    client.client.chat.completions.create.side_effect = (
        rate_limit_error
    )

    # Prevent actual waiting during the test.
    monkeypatch.setattr(
        "llm_client.time.sleep",
        lambda seconds: None,
    )

    with pytest.raises(RateLimitError):
        client.chat(
            [{"role": "user", "content": "Hello"}]
        )

    # Assumes 1 initial attempt + 3 retries.
    assert (
        client.client.chat.completions.create.call_count
        == 4
    )
