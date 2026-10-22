
import ast
import re

def validate_python_syntax(code: str) -> dict:
    """
    Check Python code syntax without executing it.
    """

    try:
        ast.parse(code)

        return {
            "valid": True,
            "error": None
        }

    except SyntaxError as error:
        return {
            "valid": False,
            "error": str(error),
            "line": error.lineno,
            "offset": error.offset
        }


def validate_python_blocks(answer: str) -> list[dict]:
    """
    Extract fenced Python code blocks from an LLM answer
    and check their syntax without executing them.
    """

    pattern = r"```(?:python|py)\s*\n(.*?)```"

    blocks = re.findall(
        pattern,
        answer,
        flags=re.DOTALL | re.IGNORECASE
    )

    results = []

    for index, code in enumerate(blocks, start=1):
        validation = validate_python_syntax(code)

        results.append({
            "block": index,
            "code": code,
            **validation
        })

    return results


def validate_api_methods(
    code: str,
    object_name: str,
    allowed_methods: set[str]
) -> dict:
    """
    Check method calls on a specific object against
    an explicitly supplied set of documented methods.

    This is a narrow check, not general API verification.
    """

    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return {
            "valid": False,
            "unknown_methods": [],
            "error": str(error)
        }

    unknown_methods = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        method = node.func

        if not isinstance(method, ast.Attribute):
            continue

        if not isinstance(method.value, ast.Name):
            continue

        if method.value.id != object_name:
            continue

        if method.attr not in allowed_methods:
            unknown_methods.add(method.attr)

    return {
        "valid": len(unknown_methods) == 0,
        "unknown_methods": sorted(unknown_methods),
        "error": None
    }
