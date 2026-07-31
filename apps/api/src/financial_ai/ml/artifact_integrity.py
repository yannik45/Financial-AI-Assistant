import hashlib
from pathlib import Path


def calculate_canonical_text_sha256(path: Path) -> str:
    """Hash UTF-8 text with platform-independent LF line endings."""
    canonical_bytes = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(canonical_bytes).hexdigest()
