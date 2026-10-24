
import json
from unittest.mock import MagicMock, patch

from agent.brain import ResearchAgent


def make_agent(responses, max_tool_calls=10):
    agent = object.__new__(ResearchAgent)
    agent.llm = MagicMock()
    agent.max_iterations = 8
    agent.max_tool_calls = max_tool_calls
    agent.max_duration_seconds = 240
    agent.max_history_turns = 5
    agent.max_tool_message_chars = 2500
    agent.reset()
    agent.llm.chat_with_tools.side_effect = responses
    agent.llm.chat.return_value = "Final answer from collected evidence."
    return agent


def tool_response(name, arguments, call_id):
    call = MagicMock()
    call.id = call_id
    call.function.name = name
    call.function.arguments = json.dumps(arguments)

    response = MagicMock()
    response.tool_calls = [call]
    response.content = None
    response.model_dump.return_value = {
        "role": "assistant",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": call.function.arguments,
            },
        }],
    }
    return response


def final_response():
    response = MagicMock()
    response.tool_calls = []
    response.content = "Research completed successfully."
    response.model_dump.return_value = {
        "role": "assistant",
        "content": response.content,
    }
    return response


def fake_result(name, success=True):
    return json.dumps({
        "tool": name,
        "success": success,
        "result": (
            {
                "url": "https://docs.python.org/3.14/",
                "title": "Python Documentation",
                "content": "Official Python documentation.",
            }
            if success else None
        ),
        "error": None if success else "Simulated tool failure",
    })


def run_scenario(agent, results):
    events = []

    with (
        patch("agent.brain.execute_tool", side_effect=results) as tool_mock,
        patch("agent.brain.log_event"),
    ):
        answer = agent.run(
            "Test research",
            progress_callback=events.append,
        )

    return answer, events, tool_mock.call_count


# TEST 1: Webpage reading
agent = make_agent([
    tool_response(
        "read_webpage",
        {"url": "https://docs.python.org/3.14/"},
        "call_read",
    ),
    final_response(),
])

answer, events, calls = run_scenario(
    agent,
    [fake_result("read_webpage")],
)

assert calls == 1
assert any(
    e.stage == "read_webpage" and e.status == "completed"
    for e in events
)
print("PASS: Webpage reading emits completion")


# TEST 2: Failed tool
agent = make_agent([
    tool_response(
        "web_search",
        {"query": "Python documentation"},
        "call_fail",
    ),
    final_response(),
])

answer, events, calls = run_scenario(
    agent,
    [fake_result("web_search", success=False)],
)

assert any(
    e.stage == "web_search" and e.status == "failed"
    for e in events
)
print("PASS: Failed tool emits failure event")


# TEST 3: Cached tool result
arguments = {"query": "Python documentation"}

agent = make_agent([
    tool_response("web_search", arguments, "call_1"),
    tool_response("web_search", arguments, "call_2"),
    final_response(),
])

answer, events, calls = run_scenario(
    agent,
    [fake_result("web_search")],
)

assert calls == 1
print("PASS: Identical tool calls use cached result")


# TEST 4: Tool budget
agent = make_agent(
    [
        tool_response(
            "web_search",
            {"query": "First search"},
            "call_budget_1",
        ),
        tool_response(
            "web_search",
            {"query": "Second search"},
            "call_budget_2",
        ),
    ],
    max_tool_calls=1,
)

answer, events, calls = run_scenario(
    agent,
    [fake_result("web_search")],
)

assert calls == 1
assert "Final answer" in answer
print("PASS: Tool execution budget enforced")

print("\nAll four tool scenario tests passed!")
