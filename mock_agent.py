
import time

from api_errors import APILimitError
from harness.progress import emit_progress


class MockResearchAgent:
    """Simulates research and progress events."""

    def __init__(self, scenario="success", step_delay=0.6):
        self.scenario = scenario
        self.step_delay = step_delay

    def run(self, question: str, progress_callback=None):

        emit_progress(
            progress_callback,
            "research",
            "Research started",
            "running",
        )
        time.sleep(self.step_delay)

        if self.scenario in ("daily", "temporary"):

            emit_progress(
                progress_callback,
                "llm",
                "Contacting AI provider",
                "running",
            )
            time.sleep(self.step_delay)

            if self.scenario == "daily":
                emit_progress(
                    progress_callback,
                    "llm",
                    "Daily API quota exhausted",
                    "failed",
                )
                raise APILimitError(
                    "Daily API quota exhausted.",
                    limit_type="daily",
                )

            emit_progress(
                progress_callback,
                "llm",
                "Temporary rate limit detected",
                "failed",
            )
            raise APILimitError(
                "Temporary API rate limit.",
                limit_type="temporary",
                retry_after=60,
            )

        stages = [
            ("web_search", "Searching the web"),
            ("read_webpage", "Reading sources"),
            ("analysis", "Analyzing information"),
            ("answer", "Preparing the final answer"),
        ]

        for stage, message in stages:
            emit_progress(
                progress_callback,
                stage,
                message,
                "running",
            )

            time.sleep(self.step_delay)

            emit_progress(
                progress_callback,
                stage,
                message,
                "completed",
            )

        emit_progress(
            progress_callback,
            "research",
            "Research completed",
            "completed",
        )

        return (
            "**Simulated Research Answer**\n\n"
            f"Your question was: {question}\n\n"
            "This response was generated in Mock Mode. "
            "No real web search or API request occurred."
        )
