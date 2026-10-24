
import json
from unittest.mock import MagicMock, patch

from agent.brain import ResearchAgent


events = []


def capture(event):
    events.append(event)


# Create the real agent without initializing the API client.
agent = object.__new__(ResearchAgent)
agent.llm = MagicMock()

agent.max_iterations = 8
agent.max_tool_calls = 10
agent.max_duration_seconds = 240
agent.max_history_turns = 5
agent.max_tool_message_chars = 2500
agent.reset()


# First LLM response requests web_search.
tool_call = MagicMock()
tool_call.id = "call_001"
tool_call.function.name = "web_search"
tool_call.function.arguments = json.dumps({
    "query": "Python 3.14 release"
})

tool_response = MagicMock()
tool_response.tool_calls = [tool_call]
tool_response.content = None
tool_response.model_dump.return_value = {
    "role": "assistant",
    "tool_calls": [
        {
            "id": "call_001",
            "type": "function",
            "function": {
                "name": "web_search",
                "arguments": tool_call.function.arguments,
            },
        }
    ],
}

# Second LLM response produces a final answer.
final_response = MagicMock()
final_response.tool_calls = []
final_response.content = "Python 3.14 is a Python release."
final_response.model_dump.return_value = {
    "role": "assistant",
    "content": final_response.content,
}

agent.llm.chat_with_tools.side_effect = [
    tool_response,
    final_response,
]


# Simulate the executor returning a successful search.
fake_tool_result = json.dumps({
    "tool": "web_search",
    "success": True,
    "result": {
        "results": [
            {
                "title": "Python Documentation",
                "url": "https://docs.python.org/3.14/",
            }
        ]
    },
    "error": None,
})


with (
    patch("agent.brain.execute_tool", return_value=fake_tool_result) as tool_mock,
    patch("agent.brain.log_event"),
):
    answer = agent.run(
        "Research Python 3.14",
        progress_callback=capture,
    )


assert "Python 3.14" in answer
assert tool_mock.call_count == 1

assert any(
    event.stage == "web_search"
    and event.status == "running"
    for event in events
)

assert any(
    event.stage == "web_search"
    and event.status == "completed"
    for event in events
)

assert events[-1].stage == "research"
assert events[-1].status == "completed"

print("PASS: Web search executes once")
print("PASS: Web search emits running event")
print("PASS: Web search emits completed event")
print("PASS: Research completes successfully")
print("No API tokens or external web requests used.")
