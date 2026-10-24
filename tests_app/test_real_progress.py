
from unittest.mock import MagicMock, patch

from agent.brain import ResearchAgent


events = []

def capture(event):
    events.append(event)


# Construct without contacting the API.
agent = object.__new__(ResearchAgent)
agent.llm = MagicMock()

agent.max_iterations = 8
agent.max_tool_calls = 10
agent.max_duration_seconds = 240
agent.max_history_turns = 5
agent.max_tool_message_chars = 2500

agent.reset()

# Simulate a direct LLM answer.
response = MagicMock()
response.tool_calls = []
response.content = "This is a simulated research answer."

agent.llm.chat_with_tools.return_value = response

with patch("agent.brain.log_event"):
    answer = agent.run(
        "Explain Python decorators",
        progress_callback=capture,
    )

assert "simulated research answer" in answer

assert any(
    event.stage == "research"
    and event.status == "running"
    for event in events
)

assert any(
    event.stage == "analysis"
    for event in events
)

assert any(
    event.stage == "research"
    and event.status == "completed"
    for event in events
)

print("PASS: Real agent emits progress events")
print("PASS: Real agent returns the simulated answer")
print("PASS: No API calls were made")
