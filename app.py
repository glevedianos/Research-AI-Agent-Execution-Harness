"""Streamlit interface for the AI Research Agent."""

from dataclasses import asdict

import streamlit as st

from agent.brain import ResearchAgent
from api_errors import APILimitError
from mock_agent import MockResearchAgent


st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔎",
    layout="wide",
)

STATUS_ICONS = {
    "running": "🔄",
    "completed": "✅",
    "failed": "❌",
}

STAGE_LABELS = {
    "research": "Research",
    "analysis": "Analysis",
    "web_search": "Web search",
    "read_webpage": "Reading sources",
    "llm_retry": "API retry",
    "answer": "Final answer",
}


def event_text(event):
    """Format a stored event (dictionary) for display."""
    symbol = STATUS_ICONS.get(event["status"], "•")
    return f'{symbol} {event["message"]}'


def render_execution_history(message, detailed=False):
    """Render events saved with an assistant response."""
    events = message.get("progress_events", [])
    if not events:
        return

    outcome = message.get("execution_status", "completed")
    heading = (
        "Execution history — completed"
        if outcome == "completed"
        else "Execution history — interrupted"
    )

    with st.expander(heading, expanded=(outcome != "completed")):
        if detailed:
            # Show the latest event for each stage, in first-seen order.
            latest = {}
            for event in events:
                latest[event["stage"]] = event
            for event in latest.values():
                st.markdown(event_text(event))
        else:
            st.caption(f"{len(events)} execution events recorded")

        if show_execution:
            st.divider()
            st.caption("Full event log (including repeated stages and retries)")
            for event in events:
                timestamp = event.get("timestamp", "")
                st.caption(f'{timestamp} · {event["stage"]} · {event_text(event)}')
                if event.get("details"):
                    st.json(event["details"], expanded=False)


# Session state survives Streamlit reruns.
defaults = {
    "messages": [],
    "agent": None,
    "pending_question": None,
    "api_error": None,
    "retry_requested": False,
    "mock_mode": True,
    "mock_scenario": "success",
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


with st.sidebar:
    st.title("⚙️ Settings")
    st.session_state.mock_mode = st.toggle(
        "Mock Mode",
        value=st.session_state.mock_mode,
        help="Simulate research without API requests.",
    )

    if st.session_state.mock_mode:
        st.warning("Mock Mode: No API calls")
        st.session_state.mock_scenario = st.selectbox(
            "Test Scenario",
            ["success", "daily", "temporary"],
            format_func=lambda x: {
                "success": "Successful Research",
                "daily": "Daily API Limit",
                "temporary": "Temporary Rate Limit",
            }[x],
        )
    else:
        st.info("Real AI Mode: Uses Groq API")

    st.divider()
    progress_mode = st.radio(
        "Progress Display", ["Detailed", "Simple", "Minimal"]
    )
    show_execution = st.checkbox("Show execution details", value=False)
    st.divider()

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.session_state.api_error = None
        st.session_state.retry_requested = False
        if st.session_state.agent is not None:
            st.session_state.agent.reset()
        st.rerun()


st.title("🔎 AI Research Assistant")
if st.session_state.mock_mode:
    st.warning(
        "TEST MODE — Responses are simulated. No actual research is performed."
    )
st.caption("AI-powered research with web search and a controlled execution harness.")
st.divider()

# Previous responses and execution histories.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("content"):
            st.markdown(message["content"])
        if message["role"] == "assistant":
            detailed_history = (
                message.get("progress_mode") == "Detailed" or show_execution
            )
            if detailed_history or message.get("execution_status") == "failed":
                render_execution_history(message, detailed=detailed_history)


# Keep the existing API-limit notification and manual retry behavior.
if st.session_state.api_error is not None:
    error = st.session_state.api_error
    if error.limit_type == "daily":
        st.error("Daily API Quota Reached")
        st.write(
            "The daily allowance has been exhausted. "
            "Please retry after the quota resets."
        )
    elif error.limit_type == "temporary":
        st.warning("Temporary API Rate Limit")
        st.write("The provider is temporarily limiting requests.")
    else:
        st.warning("API Limit Reached")

    if error.retry_after is not None:
        st.info(f"Provider retry delay: {error.retry_after} seconds")
    else:
        st.caption("Reset time unavailable.")

    if st.session_state.pending_question:
        st.write("**Pending question:** " + st.session_state.pending_question)
        if st.button("🔄 Retry Question"):
            st.session_state.retry_requested = True
            st.session_state.api_error = None
            st.rerun()


question = st.chat_input("Ask a research question...")
retrying = st.session_state.retry_requested

if retrying:
    question = st.session_state.pending_question
    st.session_state.retry_requested = False
elif question:
    st.session_state.pending_question = question
    st.session_state.api_error = None
    st.session_state.messages.append({"role": "user", "content": question})


def get_agent():
    if st.session_state.mock_mode:
        return MockResearchAgent(scenario=st.session_state.mock_scenario)
    if st.session_state.agent is None:
        st.session_state.agent = ResearchAgent()
    return st.session_state.agent


if question:
    # Store this attempt before execution. Events remain after st.rerun().
    attempt = {
        "role": "assistant",
        "content": "",
        "progress_events": [],
        "execution_status": "running",
        "progress_mode": progress_mode,
    }
    st.session_state.messages.append(attempt)

    with st.chat_message("assistant"):
        slots = {}
        progress_area = None
        status_box = None
        simple_placeholder = None

        if progress_mode == "Detailed":
            status_box = st.status("Researching your question...", expanded=True)
            progress_area = status_box.container()
        elif progress_mode == "Simple":
            simple_placeholder = st.empty()
            simple_placeholder.info("Researching your question...")

        def on_progress(event):
            # Convert dataclass to a plain dictionary for session storage.
            stored = asdict(event)
            attempt["progress_events"].append(stored)

            if progress_mode == "Detailed":
                stage = stored["stage"]
                if stage not in slots:
                    slots[stage] = progress_area.empty()
                slots[stage].markdown(event_text(stored))
            elif progress_mode == "Simple":
                simple_placeholder.info(event_text(stored))
            # Minimal mode still records events, without showing them live.

        try:
            agent = get_agent()
            answer = agent.run(question, progress_callback=on_progress)

            attempt["content"] = answer
            attempt["execution_status"] = "completed"
            st.session_state.pending_question = None
            st.session_state.api_error = None

            if status_box is not None:
                status_box.update(
                    label="Research completed", state="complete", expanded=False
                )
            if simple_placeholder is not None:
                simple_placeholder.empty()
            st.rerun()

        except APILimitError as error:
            attempt["execution_status"] = "failed"
            attempt["content"] = "Research interrupted by an API limit."
            st.session_state.api_error = error
            st.rerun()

        except Exception as error:
            attempt["execution_status"] = "failed"
            attempt["content"] = "Research could not be completed."
            st.session_state.pending_question = None
            st.session_state.api_error = None
            print(f"[GUI ERROR] {type(error).__name__}: {error}")
            st.rerun()
