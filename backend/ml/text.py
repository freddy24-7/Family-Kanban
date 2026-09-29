"""Text normalisation shared by duplicate detection (generator) and holdout
exclusion (training). Two texts with the same normalised form count as the same
ticket: 'Gras maaien!!' == 'gras maaien'."""

import hashlib
import re
import unicodedata


def normalise_text(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalise_text(text).encode()).hexdigest()
