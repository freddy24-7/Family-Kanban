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


# --- Role tokens -------------------------------------------------------------------------
# Household members' names are replaced by their role, so the model learns "a child's
# name -> probably kids" instead of "Lieke -> kids" (which only works in one family).
# The SAME function runs at training and at serving time (no training-serving skew);
# models record the version they expect in their metadata.

ROLE_MASK_VERSION = "role-mask-v1"
CHILD_TOKEN, ADULT_TOKEN = "naamkind", "naamouder"


def mask_member_names(text: str, members: list[tuple[str, bool]]) -> str:
    """members: (display name, is_child) of the ticket's household."""
    # Longest names first, so "Anne-Marie" is replaced before "Anne".
    for name, is_child in sorted(members, key=lambda m: -len(m[0])):
        if not name.strip():
            continue
        token = CHILD_TOKEN if is_child else ADULT_TOKEN
        # Also the Dutch possessive: "Liekes tas" / "Lieke's tas".
        pattern = rf"(?<![\w-]){re.escape(name.strip())}(?:'?s)?(?![\w-])"
        text = re.sub(pattern, f" {token} ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()
