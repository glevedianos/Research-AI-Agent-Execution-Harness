
def create_documentation_record(
    url: str,
    title: str,
    content: str
) -> dict:
    """
    Create a structured record for retrieved documentation.
    """
    return {
        "url": url,
        "title": title,
        "content": content
    }


def combine_documentation_content(
    records: list[dict]
) -> str:
    """
    Combine documentation content for existing
    API validation functions.
    """
    return "\n\n".join(
        record["content"]
        for record in records
        if isinstance(record.get("content"), str)
        and record["content"]
    )



def find_relevant_documentation(
    records: list[dict],
    origin: str
) -> list[dict]:
    """
    Find documentation records related to an
    object's module origin.

    Example:
        origin = "concurrent.interpreters.create"

    Returns records mentioning:
        "concurrent.interpreters"
    """
    if not origin or "." not in origin:
        return []

    module_name = origin.rsplit(".", 1)[0].lower()

    relevant = []

    for record in records:
        searchable_text = " ".join([
            str(record.get("url", "")),
            str(record.get("title", "")),
            str(record.get("content", ""))
        ]).lower()

        if module_name in searchable_text:
            relevant.append(record)

    return relevant
