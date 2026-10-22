
import ipaddress
import re
import socket
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


def validate_public_url(url: str) -> None:
    """Allow only HTTP(S) URLs resolving exclusively to public IPs."""
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only HTTP and HTTPS URLs are allowed")

    if not parsed.hostname:
        raise ValueError("Invalid URL")

    if parsed.username or parsed.password:
        raise ValueError("URLs with credentials are not allowed")

    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Invalid URL port") from exc

    if port not in (None, 80, 443):
        raise ValueError("Unsupported URL port")

    addresses = socket.getaddrinfo(
        parsed.hostname,
        port or (443 if parsed.scheme == "https" else 80),
        type=socket.SOCK_STREAM,
    )

    if not addresses:
        raise ValueError("Could not resolve URL hostname")

    for address in addresses:
        if not ipaddress.ip_address(address[4][0]).is_global:
            raise ValueError("Private or non-public address blocked")


def extract_readable_text(html: bytes) -> tuple[str, str]:
    """Preserve headings and API signatures while removing page chrome."""
    soup = BeautifulSoup(html, "html.parser")

    title = (
        soup.title.get_text(" ", strip=True)
        if soup.title else ""
    )

    for element in soup([
        "script", "style", "nav", "footer",
        "header", "noscript", "svg"
    ]):
        element.decompose()

    main = (
        soup.select_one("main")
        or soup.select_one("article")
        or soup.body
        or soup
    )

    # Sphinx's documentation body excludes sidebar navigation.
    sphinx_body = (
        main.select_one('[role="main"]')
        or main.select_one(".body")
    )

    if sphinx_body:
        main = sphinx_body

    for element in main.select(
        ".headerlink, .copybtn, .viewcode-link"
    ):
        element.decompose()

    # Newlines separate signatures and headings.
    for element in main.find_all([
        "h1", "h2", "h3", "h4",
        "p", "li", "dt", "dd", "pre"
    ]):
        element.insert_before("\n")
        element.insert_after("\n")

    text = main.get_text(" ", strip=False)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return title, text


def _query_pattern(query: str) -> re.Pattern:
    """Allow whitespace around dots in API signatures."""
    parts = [
        re.escape(part.strip())
        for part in query.strip().split(".")
    ]

    return re.compile(
        r"\s*\.\s*".join(parts),
        re.IGNORECASE
    )


def select_query_context(
    text: str,
    query: str,
    max_chars: int
) -> tuple[bool, str]:
    """Find matches across full text and prioritize API definitions."""
    if not query.strip():
        return False, ""

    pattern = _query_pattern(query)
    matches = list(pattern.finditer(text))

    if not matches:
        return False, ""

    def rank(match):
        line_start = text.rfind(
            "\n", 0, match.start()
        ) + 1

        line_end = text.find(
            "\n", match.end()
        )

        if line_end == -1:
            line_end = len(text)

        line = text[line_start:line_end].strip()

        # Prefer definitions over prose.
        signature = bool(
            re.search(r"\([^\n]*\)", line)
        )

        return (not signature, match.start())

    matches.sort(key=rank)

    excerpts = []
    used = []
    remaining = max(0, max_chars)

    for match in matches:
        if remaining < 100 or len(excerpts) >= 3:
            break

        start = max(0, match.start() - 350)
        end = min(len(text), match.end() + 1600)

        if any(
            start < b and end > a
            for a, b in used
        ):
            continue

        excerpt = text[start:end].strip()
        excerpt = excerpt[:remaining]

        if excerpt:
            excerpts.append(excerpt)
            used.append((start, end))
            remaining -= len(excerpt) + 2

    return True, "\n\n".join(excerpts)


def read_webpage(
    url: str,
    query: str = "",
    max_chars: int = 10000
) -> dict:
    """Read a public HTML page and retrieve matching API context."""
    try:
        validate_public_url(url)

        response = requests.get(
            url,
            timeout=10,
            allow_redirects=False,
            headers={
                "User-Agent": "ResearchAgent/0.1"
            },
            stream=True,
        )

        try:
            if 300 <= response.status_code < 400:
                raise ValueError(
                    "Redirects are not supported"
                )

            response.raise_for_status()

            content_type = response.headers.get(
                "Content-Type", ""
            )

            if "text/html" not in content_type.lower():
                raise ValueError(
                    "URL does not return HTML"
                )

            chunks = []
            total = 0

            for chunk in response.iter_content(
                chunk_size=8192
            ):
                total += len(chunk)

                if total > 1_000_000:
                    raise ValueError(
                        "Webpage is too large"
                    )

                chunks.append(chunk)

            html = b"".join(chunks)

        finally:
            response.close()

        title, text = extract_readable_text(html)

        if query:
            found, content = select_query_context(
                text,
                query,
                max_chars
            )

            return {
                "url": url,
                "title": title,
                "query": query,
                "found": found,
                "content": content,
            }

        return {
            "url": url,
            "title": title,
            "content": text[:max_chars]
        }

    except Exception as error:
        return {
            "url": url,
            "error": str(error)
        }
