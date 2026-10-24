
from unittest.mock import MagicMock, patch

from openai import RateLimitError, APIConnectionError
import httpx

from llm_client import LLMClient
from api_errors import APILimitError


# ---------------------------------------------
# Helpers
# ---------------------------------------------

def make_rate_limit_error(message):
    request = httpx.Request(
        "POST",
        "https://api.groq.com/openai/v1/chat/completions"
    )

    response = httpx.Response(
        status_code=429,
        request=request,
        json={"error": {"message": message}}
    )

    return RateLimitError(
        message,
        response=response,
        body={"error": {"message": message}}
    )


def make_connection_error():
    request = httpx.Request(
        "POST",
        "https://api.groq.com/openai/v1/chat/completions"
    )

    return APIConnectionError(
        message="Simulated connection failure",
        request=request
    )


def make_client():
    # Bypass __init__ to avoid reading API credentials.
    client = object.__new__(LLMClient)
    client.model = "test-model"

    client.client = MagicMock()

    return client


def make_success_response():
    response = MagicMock()
    response.model = "test-model"
    response.usage = None
    response.choices[0].message.content = (
        "Simulated successful answer"
    )
    return response


def run_test(name, side_effects, expected_calls,
             expected_error=None):

    client = make_client()
    endpoint = client.client.chat.completions.create

    endpoint.side_effect = side_effects

    with (
        patch("llm_client.time.sleep") as fake_sleep,
        patch("llm_client.random.uniform", return_value=0),
        patch("llm_client.log_event")
    ):

        if expected_error is None:
            answer = client.chat([
                {"role": "user", "content": "Test question"}
            ])

            assert answer == "Simulated successful answer"

        else:
            try:
                client.chat([
                    {"role": "user", "content": "Test question"}
                ])
                raise AssertionError(
                    f"{name}: Expected an exception"
                )
            except expected_error:
                pass

        assert endpoint.call_count == expected_calls, (
            f"{name}: Expected {expected_calls} calls, "
            f"got {endpoint.call_count}"
        )

        assert fake_sleep.call_count == max(
            0,
            expected_calls - 1
        ) if expected_error is not APILimitError or expected_calls > 1 else fake_sleep.call_count == 0

    print(f"PASS: {name}")


# ---------------------------------------------
# Test 1: Temporary limit, then success
# ---------------------------------------------

run_test(
    name="Temporary limit then success",
    side_effects=[
        make_rate_limit_error(
            "Rate limit reached on tokens per minute (TPM)"
        ),
        make_rate_limit_error(
            "Rate limit reached on tokens per minute (TPM)"
        ),
        make_success_response()
    ],
    expected_calls=3
)


# ---------------------------------------------
# Test 2: Daily quota stops immediately
# ---------------------------------------------

run_test(
    name="Daily quota stops immediately",
    side_effects=[
        make_rate_limit_error(
            "Rate limit reached on tokens per day (TPD)"
        )
    ],
    expected_calls=1,
    expected_error=APILimitError
)


# ---------------------------------------------
# Test 3: Persistent rate limit
# ---------------------------------------------

run_test(
    name="Persistent rate limit stops at maximum",
    side_effects=[
        make_rate_limit_error(
            "Rate limit reached on tokens per minute (TPM)"
        )
        for _ in range(4)
    ],
    expected_calls=4,
    expected_error=APILimitError
)


# ---------------------------------------------
# Test 4: Connection failure then success
# ---------------------------------------------

run_test(
    name="Connection failure then success",
    side_effects=[
        make_connection_error(),
        make_success_response()
    ],
    expected_calls=2
)


print("\nAll retry tests passed!")
