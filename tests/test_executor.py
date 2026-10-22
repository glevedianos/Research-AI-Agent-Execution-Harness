
import json

import harness.executor as executor


def test_successful_tool_execution(monkeypatch):
    """A registered tool should execute successfully."""

    def fake_search(query):
        return [{"title": "Test", "url": "https://example.com"}]

    monkeypatch.setitem(
        executor.TOOL_FUNCTIONS,
        "web_search",
        fake_search
    )

    result = json.loads(
        executor.execute_tool(
            "web_search",
            json.dumps({"query": "Python"}),
            run_id="test-success"
        )
    )

    assert result["success"] is True
    assert result["tool"] == "web_search"
    assert result["error"] is None
    assert result["result"][0]["title"] == "Test"


def test_reject_unknown_tool():
    """Unknown tools must never be executed."""

    result = json.loads(
        executor.execute_tool(
            "dangerous_tool",
            "{}",
            run_id="test-unknown"
        )
    )

    assert result["success"] is False
    assert "not permitted" in result["error"]


def test_reject_invalid_json():
    """Tool arguments must be valid JSON."""

    result = json.loads(
        executor.execute_tool(
            "web_search",
            "{invalid json}",
            run_id="test-json"
        )
    )

    assert result["success"] is False
    assert result["result"] is None


def test_reject_oversized_arguments():
    """Oversized tool arguments must be rejected."""

    arguments = json.dumps({"query": "A" * 3000})

    result = json.loads(
        executor.execute_tool(
            "web_search",
            arguments,
            run_id="test-size"
        )
    )

    assert result["success"] is False
    assert "exceed 2000 characters" in result["error"]


def test_reject_missing_arguments():
    """A tool must receive its required arguments."""

    result = json.loads(
        executor.execute_tool(
            "web_search",
            "{}",
            run_id="test-missing"
        )
    )

    assert result["success"] is False
    assert "Missing required arguments" in result["error"]


def test_tool_execution_failure(monkeypatch):
    """Exceptions from tools should become safe JSON errors."""

    def fake_search(query):
        raise RuntimeError("Simulated search failure")

    monkeypatch.setitem(
        executor.TOOL_FUNCTIONS,
        "web_search",
        fake_search
    )

    result = json.loads(
        executor.execute_tool(
            "web_search",
            json.dumps({"query": "Python"}),
            run_id="test-failure"
        )
    )

    assert result["success"] is False
    assert result["result"] is None
    assert "Simulated search failure" in result["error"]



def test_large_tool_result_is_truncated(monkeypatch):
    """
    Large webpage results should be truncated
    before being returned to the agent.
    """

    def fake_reader(url):
        return {
            "url": url,
            "title": "Large Test Page",
            "content": "A" * 10000,
        }

    monkeypatch.setitem(
        executor.TOOL_FUNCTIONS,
        "read_webpage",
        fake_reader,
    )

    result = json.loads(
        executor.execute_tool(
            "read_webpage",
            json.dumps({
                "url": "https://example.com"
            }),
            run_id="test-truncation",
        )
    )

    assert result["success"] is True

    content = result["result"]["content"]

    assert len(content) < 10000
    assert "[Content truncated]" in content
    assert result["result"]["url"] == "https://example.com"



def test_read_webpage_accepts_optional_query(monkeypatch):
    """
    The executor should forward an optional query
    to the webpage reader.
    """

    def fake_reader(url, query=""):
        return {
            "url": url,
            "query": query,
            "found": True,
            "content": "Interpreter.call_in_thread"
        }

    monkeypatch.setitem(
        executor.TOOL_FUNCTIONS,
        "read_webpage",
        fake_reader
    )

    result = json.loads(
        executor.execute_tool(
            "read_webpage",
            json.dumps({
                "url": "https://docs.python.org/3.14/",
                "query": "call_in_thread"
            })
        )
    )

    assert result["success"] is True
    assert result["result"]["query"] == "call_in_thread"
    assert result["result"]["found"] is True
