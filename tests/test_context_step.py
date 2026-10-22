
import json

from agent.brain import ResearchAgent


def test_compact_webpage_tool_message():
    agent = ResearchAgent.__new__(ResearchAgent)
    agent.max_tool_message_chars = 2500

    raw = json.dumps({
        "tool": "read_webpage",
        "success": True,
        "result": {
            "url": "https://docs.python.org/3.14/",
            "content": "X" * 5000
        },
        "error": None,
    })

    compact = json.loads(
        agent._compact_tool_message(raw)
    )

    assert compact["result"]["url"] == (
        "https://docs.python.org/3.14/"
    )
    assert len(compact["result"]["content"]) == 2500
    assert len(json.loads(raw)["result"]["content"]) == 5000


def test_compact_search_result_is_valid_json():
    agent = ResearchAgent.__new__(ResearchAgent)
    agent.max_tool_message_chars = 2500

    raw = json.dumps({
        "tool": "web_search",
        "success": True,
        "result": ["X" * 5000],
        "error": None
    })

    compact = json.loads(
        agent._compact_tool_message(raw)
    )

    assert compact["tool"] == "web_search"
    assert compact["truncated"] is True


def test_detects_token_limit_but_not_generic_errors():
    assert ResearchAgent._is_token_limit_error(
        RuntimeError(
            "Request too large on tokens per minute "
            "(TPM): Limit 8000"
        )
    )

    assert not ResearchAgent._is_token_limit_error(
        RuntimeError("connection timeout")
    )
