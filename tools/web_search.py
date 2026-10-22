
from ddgs import DDGS


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web and return structured results.
    """
    if not query.strip():
        return []

    try:
        results = DDGS().text(
            query,
            max_results=max_results
        )

        formatted_results = []

        for result in results:
            formatted_results.append({
                "title": result.get("title", ""),
                "url": result.get("href", ""),
                "snippet": result.get("body", "")
            })

        return formatted_results

    except Exception as error:
        print(f"Web search failed: {error}")
        return []
