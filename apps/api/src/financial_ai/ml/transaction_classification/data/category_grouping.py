import re


def normalize_description_group(description: str) -> str:
    """Return a stable heuristic key for related transaction descriptions."""
    normalized = description.casefold()
    normalized = re.sub(
        r"^\[(?:debit|credit)\]\s*",
        "",
        normalized,
    )

    normalized = re.sub(
        r"^(?:pp|pypl|paypal|sq)\s*\*\s*",
        "",
        normalized,
    )

    normalized = re.sub(
        r"\s+\d+\s+(?:ppd|web)\s+id:.*$",
        "",
        normalized,
    )

    normalized = re.sub(
        r"[#*]\s*[a-z0-9-]+",
        " ",
        normalized,
    )

    normalized = re.sub(
        r"(?<![a-z0-9-])\d{3,}(?![a-z0-9-])",
        " ",
        normalized,
    )

    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = " ".join(normalized.split())

    if not normalized:
        raise ValueError("Description group must not be empty")

    return normalized
