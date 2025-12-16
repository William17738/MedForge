"""
MedForge Text Utilities

Text normalization and cleaning utilities for document processing.
"""

import unicodedata
import re


def normalize_text(text: str) -> str:
    """
    Normalize text with enhanced cleaning for OCR and document processing.

    Handles:
    - Unicode normalization (NFKC)
    - BOM and zero-width character removal
    - Punctuation standardization
    - Whitespace consolidation
    - Invisible control character removal

    Args:
        text: Input text to normalize

    Returns:
        Normalized text string
    """
    if not text:
        return ""

    # Unicode normalization (NFKC form)
    text = unicodedata.normalize("NFKC", text)

    # Remove BOM and zero-width characters
    text = text.replace("\ufeff", "")
    text = re.sub(r"[\u200b-\u200f]", "", text)

    # Consolidate whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove invisible control characters (preserve newlines/tabs)
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\r\t")

    return text.strip()


def extract_chapter_number(text: str) -> int | None:
    """
    Extract chapter number from text.

    Args:
        text: Text potentially containing chapter number

    Returns:
        Chapter number if found, None otherwise
    """
    patterns = [
        r"chapter\s*(\d+)",
        r"ch\.\s*(\d+)",
        r"section\s*(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))

    return None
