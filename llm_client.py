
import os
import time
import random

from dotenv import load_dotenv
from openai import (
    OpenAI,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    BadRequestError,
)

from harness.logger import log_event
from api_errors import classify_api_error
from harness.progress import emit_progress

load_dotenv()


class LLMClient:
    #OpenRouter Key
    """
    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is missing from .env"
            )

        self.model = os.getenv(
            "OPENROUTER_MODEL",
            "openrouter/free"
        )

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=30.0,
            max_retries=0
        )
        """
    #Groq Key
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing from .env"
            )

        self.model = os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b"
        )

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=30.0,
            max_retries=0
        )


    def _request(self, messages, tools=None, progress_callback=None):
        max_attempts = 4
        invalid_tool_retries = 0

        # Copy messages so we don't modify agent memory.
        request_messages = list(messages)

        for attempt in range(max_attempts):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": request_messages,
                }

                if tools is not None:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"


                response = self.client.chat.completions.create(**kwargs)

                print(f"[DEBUG] Model: {response.model}")

                usage = response.usage

                if usage is not None:
                    print(
                        f"[TOKENS] Prompt: {usage.prompt_tokens} | "
                        f"Completion: {usage.completion_tokens} | "
                        f"Total: {usage.total_tokens}"
                    )

                    log_event(
                        "llm_token_usage",
                        model=response.model,
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                        total_tokens=usage.total_tokens
                    )

                return response.choices[0].message



            except BadRequestError as error:
                error_text = str(error)

                # Prefer the structured API error code.
                error_code = getattr(error, "code", None)

                if error_code is None:
                    error_body = getattr(error, "body", None)

                    if isinstance(error_body, dict):
                        error_code = error_body.get("code")

                        if error_code is None:
                            nested_error = error_body.get("error", {})

                            if isinstance(nested_error, dict):
                                error_code = nested_error.get("code")

                is_invalid_tool_call = (
                    error_code == "tool_use_failed"
                    or "tool_use_failed" in error_text
                )

                if (
                    tools is not None
                    and is_invalid_tool_call
                    and invalid_tool_retries < 1
                ):

                    invalid_tool_retries += 1

                    allowed_tools = [
                        tool["function"]["name"]
                        for tool in tools
                    ]

                    print(
                        "[HARNESS] Invalid tool call detected. "
                        "Retrying with tool restrictions..."
                    )

                    log_event(
                        "llm_invalid_tool_call",
                        error_type="BadRequestError",
                        allowed_tools=allowed_tools,
                    )

                    # Add a temporary instruction only for
                    # the retry, not for permanent memory.
                    request_messages = [
                        {
                            "role": "system",
                            "content": (
                                "IMPORTANT: You may only call "
                                "these tools: "
                                + ", ".join(allowed_tools)
                                + ". Never invent tool names. "
                                "If none of these tools can "
                                "help, answer directly."
                            ),
                        },
                        *messages,
                    ]

                    continue

                raise


            except (
                RateLimitError,
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
            ) as error:

                attempt_number = attempt + 1

                log_event(
                    "llm_request_error",
                    error_type=type(error).__name__,
                    attempt=attempt_number,
                )

                # Classify provider limits.
                api_limit = classify_api_error(error)

                # Daily quotas should not be retried immediately.
                if (
                    api_limit is not None
                    and api_limit.limit_type == "daily"
                ):
                    print(
                        "[HARNESS] Daily API quota exhausted."
                    )

                    log_event(
                        "llm_quota_exhausted",
                        limit_type="daily",
                        attempt=attempt_number,
                    )

                    raise api_limit from error

                # No retries remain.
                if attempt_number >= max_attempts:
                    if api_limit is not None:
                        raise api_limit from error

                    raise

                # Exponential backoff with jitter.
                delay = min(
                    2 ** attempt, 8
                ) + random.uniform(0, 1)

                # Respect the provider's retry delay.
                if (
                    api_limit is not None
                    and api_limit.retry_after is not None
                ):
                    delay = max(
                        delay,
                        api_limit.retry_after + 1
                    )

                # Keep requests within a reasonable
                # interactive waiting budget.
                max_wait_seconds = 60

                if delay > max_wait_seconds:
                    print(
                        "[HARNESS] Retry delay exceeds "
                        "the interactive waiting budget."
                    )

                    if api_limit is not None:
                        raise api_limit from error

                    raise

                print(
                    f"[HARNESS] Request failed. "
                    f"Retry {attempt_number}/{max_attempts - 1} "
                    f"in {delay:.1f} seconds."
                )

                log_event(
                    "llm_retry_scheduled",
                    attempt=attempt_number,
                    next_attempt=attempt_number + 1,
                    max_attempts=max_attempts,
                    delay_seconds=round(delay, 2),
                    error_type=type(error).__name__,
                )

                emit_progress(
                    progress_callback,
                    "llm_retry",
                    (
                        f"API request failed. "
                        f"Retrying in {delay:.1f} seconds "
                        f"(attempt {attempt_number + 1}"
                        f"/{max_attempts})"
                    ),
                    "running",
                    next_attempt=attempt_number + 1,
                    max_attempts=max_attempts,
                    delay_seconds=round(delay, 2),
                )

                time.sleep(delay)


    def chat(self, messages, progress_callback=None):
        response = self._request(
            messages,
            progress_callback=progress_callback,
        )
        return response.content or ""

    def chat_with_tools(
        self,
        messages,
        tools,
        progress_callback=None,
    ):
        return self._request(
            messages,
            tools=tools,
            progress_callback=progress_callback,
        )
