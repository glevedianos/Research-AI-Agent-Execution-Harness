
from harness.progress import emit_progress
from mock_agent import MockResearchAgent
from api_errors import APILimitError


events = []

def capture(event):
    events.append(event)


agent = MockResearchAgent(
    scenario="success",
    step_delay=0,
)

answer = agent.run(
    "Test question",
    progress_callback=capture,
)

assert "Simulated Research Answer" in answer
assert events[0].stage == "research"
assert events[0].status == "running"
assert events[-1].status == "completed"
assert any(
    event.stage == "web_search"
    for event in events
)

print("PASS: Successful research emits progress")


events.clear()

agent = MockResearchAgent(
    scenario="daily",
    step_delay=0,
)

try:
    agent.run(
        "Test daily limit",
        progress_callback=capture,
    )
    raise AssertionError("Expected APILimitError")
except APILimitError as error:
    assert error.limit_type == "daily"

assert events[-1].status == "failed"

print("PASS: Daily limit emits failure event")
print("All progress tests passed!")
