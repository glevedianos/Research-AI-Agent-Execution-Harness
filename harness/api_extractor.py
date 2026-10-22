
import ast
import re


def extract_documented_methods(
    documentation: str,
    candidate_methods: set[str]
) -> set[str]:
    """
    Find candidate method names that appear as
    function signatures in documentation text.
    """

    documented = set()

    for method in candidate_methods:
        pattern = (
            r"(?<![\w])"
            + re.escape(method)
            + r"\s*\("
        )

        if re.search(pattern, documentation):
            documented.add(method)

    return documented


def extract_called_methods(
    code: str,
    object_name: str
) -> set[str]:
    """
    Extract methods called on a specific object.
    """

    tree = ast.parse(code)
    methods = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        function = node.func

        if not isinstance(function, ast.Attribute):
            continue

        if not isinstance(function.value, ast.Name):
            continue

        if function.value.id == object_name:
            methods.add(function.attr)

    return methods


def verify_methods_against_docs(
    code: str,
    documentation: str,
    object_name: str
) -> dict:
    """
    Compare methods called in Python code with
    method names found in documentation.

    Missing methods are unverified, not necessarily invalid.
    """

    try:
        called_methods = extract_called_methods(
            code,
            object_name
        )

    except SyntaxError as error:
        return {
            "valid_syntax": False,
            "verified_methods": [],
            "unverified_methods": [],
            "error": str(error)
        }

    documented_methods = extract_documented_methods(
        documentation,
        called_methods
    )

    unverified_methods = called_methods - documented_methods

    return {
        "valid_syntax": True,
        "verified_methods": sorted(documented_methods),
        "unverified_methods": sorted(unverified_methods),
        "error": None
    }

def extract_object_names(code: str) -> set[str]:
    """
    Find variable names used to call methods.

    Example:
        interp.exec("hello")
        interp.close()

    Returns:
        {"interp"}
    """
    tree = ast.parse(code)
    objects = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        function = node.func

        if not isinstance(function, ast.Attribute):
            continue

        if isinstance(function.value, ast.Name):
            objects.add(function.value.id)

    return objects



def documentation_mentions_object(
    documentation: str,
    object_name: str
) -> bool:
    """
    Check whether documentation mentions an object name.

    This is a heuristic, not proof that the documentation
    describes the object's actual Python class.
    """
    if not documentation or not object_name:
        return False

    pattern = (
        r"\b"
        + re.escape(object_name)
        + r"\b"
    )

    return bool(
        re.search(
            pattern,
            documentation,
            flags=re.IGNORECASE
        )
    )



def extract_imports(code: str) -> dict[str, str]:
    """
    Extract imported modules and their local names.

    Examples:
        import pandas as pd
        from concurrent import interpreters
        from pathlib import Path

    Returns:
        {
            "pd": "pandas",
            "interpreters": "concurrent.interpreters",
            "Path": "pathlib.Path"
        }
    """
    tree = ast.parse(code)
    imports = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local_name = (
                    alias.asname
                    or alias.name.split(".")[0]
                )

                imports[local_name] = alias.name

        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or not node.module:
                continue

            for alias in node.names:
                if alias.name == "*":
                    continue

                local_name = alias.asname or alias.name

                imports[local_name] = (
                    f"{node.module}.{alias.name}"
                )

    return imports



def extract_object_origins(code: str) -> dict[str, str]:
    """
    Trace simple assignments from imported modules.

    Example:
        from concurrent import interpreters
        interp = interpreters.create()

    Returns:
        {"interp": "concurrent.interpreters.create"}
    """
    tree = ast.parse(code)
    imports = extract_imports(code)
    origins = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue

        if not isinstance(node.value, ast.Call):
            continue

        function = node.value.func

        if not isinstance(function, ast.Attribute):
            continue

        if not isinstance(function.value, ast.Name):
            continue

        source_name = function.value.id

        if source_name not in imports:
            continue

        origin = (
            f"{imports[source_name]}.{function.attr}"
        )

        for target in node.targets:
            if isinstance(target, ast.Name):
                origins[target.id] = origin

    return origins



def documentation_matches_origin(
    documentation: str,
    origin: str
) -> bool:
    """
    Check whether documentation mentions the module
    associated with an object's origin.

    This is a relevance heuristic, not API verification.
    """
    if not documentation or not origin:
        return False

    # Remove the factory or constructor name.
    # Example:
    # concurrent.interpreters.create
    # becomes concurrent.interpreters
    parts = origin.split(".")

    if len(parts) < 2:
        return False

    module_name = ".".join(parts[:-1])

    return module_name.lower() in documentation.lower()
