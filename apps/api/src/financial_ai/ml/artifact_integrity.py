import hashlib
import re
from pathlib import Path


def calculate_canonical_text_sha256(path: Path) -> str:
    """Hash UTF-8 text with platform-independent LF line endings."""
    canonical_bytes = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(canonical_bytes).hexdigest()


def normalize_artifact_version(version: str) -> str:
    """Return a filesystem-safe explicit artifact version."""
    normalized = version.strip().casefold()
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", normalized):
        raise ValueError(
            "Artifact version must contain only lowercase letters, numbers, '.', '_' or '-'"
        )
    return normalized
