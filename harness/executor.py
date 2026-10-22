import json
import time
import traceback
from harness.registry import TOOL_FUNCTIONS
from harness.logger import log_event


ALLOWED_ARGS = {
    "web_search": {
        "required": {"query"},
        "optional": set()
    },
    "read_webpage": {
        "required": {"url"},
        "optional": {"query"}
    }
}

MAX_ARGUMENT_LENGTH = 2000
MAX_TOOL_RESULT_CHARS = 5000


def limit_tool_result(result, max_chars=MAX_TOOL_RESULT_CHARS):
    """
    Limit tool output size while preserving
    valid JSON-compatible data structures.
    """
    serialized = json.dumps(result, ensure_ascii=False)

    if len(serialized) <= max_chars:
        return result

    # Webpage results usually contain a large
    # content field. Truncate that field first.
    if isinstance(result, dict) and "content" in result:
        limited = result.copy()

        original_content = str(limited["content"])
        excess = len(serialized) - max_chars

        new_length = max(
            0,
            len(original_content) - excess - 100
        )

        limited["content"] = (
            original_content[:new_length]
            + "\n[Content truncated]"
        )

        return limited

    # For other large results, return a compact
    # representation rather than malformed JSON.
    return {
        "truncated": True,
        "content": serialized[:max_chars - 100]
    }


def execute_tool(name: str, arguments: str, run_id: str = None) -> str:
    """
    Validate, execute, and log a registered tool.

    Returns a JSON string containing:
    - tool name
    - success status
    - result
    - error message
    """
    start_time = time.perf_counter()

    try:
        # STEP 1: Check whether the tool is registered.
        if name not in TOOL_FUNCTIONS:
            raise ValueError(
                f"Tool '{name}' is not permitted"
            )

        # STEP 2: Convert JSON arguments into a dictionary.
        if len(arguments) > MAX_ARGUMENT_LENGTH:
            raise ValueError(
                f"Tool arguments exceed {MAX_ARGUMENT_LENGTH} characters")

        parsed_args = json.loads(arguments)

        if not isinstance(parsed_args, dict):
            raise ValueError(
                "Arguments must be a JSON object"
            )

        # STEP 3: Validate the arguments.

        arg_rules = ALLOWED_ARGS[name]

        required_args = arg_rules["required"]
        optional_args = arg_rules["optional"]

        provided_args = set(parsed_args)

        if not required_args.issubset(provided_args):
            raise ValueError(
                f"Missing required arguments: "
                f"{sorted(required_args - provided_args)}"
            )

        allowed_args = required_args | optional_args

        if not provided_args.issubset(allowed_args):
            raise ValueError(
                f"Unexpected arguments: "
                f"{sorted(provided_args - allowed_args)}"
            )

        for key, value in parsed_args.items():
            if not isinstance(value, str):
                raise ValueError("Arguments must be strings")

            if key in required_args and not value.strip():
                raise ValueError(
                    "Required arguments must be nonempty strings"
                )

        # STEP 4: Execute the actual Python function.
        print(f"[HARNESS] Executing: {name}")

        # STEP 4: Execute the actual Python function.
        function = TOOL_FUNCTIONS[name]
        result = function(**parsed_args)

        # STEP 5: Detect errors returned as dictionaries.
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(str(result["error"]))

        result = limit_tool_result(result)
        # STEP 6: Calculate execution time.
        duration_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2
        )

        # STEP 7: Log successful execution.
        log_event(
            "tool_execution",
            run_id=run_id,
            tool=name,
            status="success",
            duration_ms=duration_ms
        )

        # STEP 8: Return a standardized JSON result.
        return json.dumps({
            "tool": name,
            "success": True,
            "result": result,
            "error": None
        })

    except Exception as error:
        duration_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2
        )

        log_event(
            "tool_execution",
            run_id=run_id,
            tool=name,
            status="error",
            duration_ms=duration_ms,
            error_type=type(error).__name__
        )

        return json.dumps({
            "tool": name,
            "success": False,
            "result": None,
            "error": str(error)
        })