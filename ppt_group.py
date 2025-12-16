"""
MedForge PPT Integration Pipeline

Generates integrated lecture notes by combining PPT content with textbook material.
Uses intelligent matching to find relevant PPT files for each chapter.
"""

import re
import textwrap
from collections import Counter
from pathlib import Path
from typing import Optional, List
 
from config import OUTPUT_DIR, PPT_DIR
from llm_client import call_llm_with_smart_routing
from utils_fs import atomic_write_text


# System prompt for lecture note generation
PPT_SYSTEM_PROMPT = """You are an educational content specialist responsible for creating integrated lecture notes.

**Core Task**: Using PPT slides as the structural framework and textbook content as detailed reference, generate comprehensive lecture notes.

**Input**:
1. PPT OCR text (may contain fragmented information)
2. Textbook OCR text (detailed content)

**Output Requirements**:
1. **Pure Markdown format**
2. **Structured Integration**:
   - Follow the PPT's logical flow (typically: definition -> causes -> symptoms -> diagnosis -> treatment)
   - Where PPT only has headings or keywords, **must** supplement with detailed textbook content
3. **Required Sections**:
   - **Key Insights**: Summary of important points at each major section
   - **Core Concepts**: Precise definitions of key terms
   - **Comparisons**: Differentiation of similar concepts (use tables or lists)
   - **Clinical Cases**: If available in source material
   - **Treatment Flow**: For clinical chapters, outline diagnosis and treatment steps
   - **Additional Details**: Important textbook details not covered in PPT
4. **Markers**:
   - `*`: High-frequency exam points
   - `!`: Common mistakes/confusion points
   - **Bold**: Key terminology
5. **Error Correction**: Fix OCR errors when detected
"""


def _clean_markdown_duplicates(md: str) -> str:
    """
    Remove duplicate paragraphs from generated markdown.
    - Split by blank lines
    - Preserve headers (# lines)
    - Remove adjacent duplicates
    - Remove non-adjacent duplicates for longer paragraphs
    """
    blocks = [b for b in md.split("\n\n") if b.strip()]
    seen = set()
    out_blocks = []
    last_norm = None

    for block in blocks:
        stripped = block.strip()

        # Preserve header lines
        if re.match(r"^#{1,6}\s+\S+", stripped) and "\n" not in stripped:
            out_blocks.append(block)
            last_norm = None
            continue

        # Normalize whitespace
        norm = re.sub(r"\s+", " ", stripped)

        # Remove adjacent duplicates
        if norm == last_norm:
            continue
        last_norm = norm

        # Remove non-adjacent duplicates for longer paragraphs
        if len(norm) >= 10:
            if norm in seen:
                continue
            seen.add(norm)

        out_blocks.append(block)

    return "\n\n".join(out_blocks)


def _read_text_if_exists(path: Path, limit: Optional[int] = None) -> str:
    """Read text file with optional truncation and error handling."""
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if limit is not None:
            return text[:limit]
        return text
    except Exception:
        return ""


def _extract_keywords(text: str, top_k: int = 20) -> List[str]:
    """
    Extract high-frequency keywords from text.
    Uses simple word tokenization for keyword extraction.
    """
    if not text:
        return []

    # Use first 8000 chars for analysis
    text = text[:8000].lower()

    # Extract words (alphanumeric sequences)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    if not words:
        return []

    # Common stop words
    stop_words = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
        'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
        'have', 'been', 'that', 'this', 'with', 'will', 'from',
        'chapter', 'section', 'page', 'figure', 'table'
    }

    freq = Counter(words)
    for word in list(freq.keys()):
        if word in stop_words or len(word) < 3:
            freq.pop(word, None)

    if not freq:
        return []

    items = sorted(freq.items(), key=lambda x: (-x[1], -len(x[0])))
    return [w for w, _ in items[:top_k]]


def _score_ppt_file(
    ppt_path: Path,
    keywords: List[str],
    chapter_id: str,
    chapter_name: str,
) -> int:
    """
    Score a PPT file for relevance to a chapter.
    Higher scores indicate better matches.
    """
    text = _read_text_if_exists(ppt_path, limit=6000)
    if not text:
        return 0

    score = 0
    fname = ppt_path.stem.lower()
    text_lower = text.lower()

    # Normalize chapter ID
    chapter_id_str = str(int(chapter_id)) if chapter_id.isdigit() else chapter_id

    # Clean chapter name
    chap_key = chapter_name.replace("_", " ").lower()

    # Filename matching rules
    if chap_key and chap_key in fname:
        score += 20
    elif len(chap_key) > 6 and chap_key[:6] in fname:
        score += 15

    # Chapter ID matching
    if re.search(rf"(^|[\\-_\\.\\s]){chapter_id_str}([\\-_\\.\\s]|$)", fname):
        score += 10

    # Keyword matching
    for kw in keywords:
        if len(kw) < 3:
            continue

        # Filename hits (higher weight)
        hits_name = fname.count(kw.lower())
        if hits_name:
            score += hits_name * 5

        # Content hits (lower weight, capped)
        hits_body = text_lower.count(kw.lower())
        if hits_body:
            score += min(hits_body, 10)

    return score


def _collect_ppt_files(subject: str) -> List[Path]:
    """Collect all PPT text files for a subject."""
    files: List[Path] = []

    if not PPT_DIR.exists():
        print(f"[PPT] PPT directory not found: {PPT_DIR}")
        return files

    subject_lower = subject.lower()

    # Search directories
    search_dirs: List[Path] = []

    # Try subject-named directory
    subj_dir = PPT_DIR / subject
    if subj_dir.exists() and subj_dir.is_dir():
        search_dirs.append(subj_dir)

    # Scan for matching directories
    for d in PPT_DIR.iterdir():
        if not d.is_dir():
            continue
        if subject_lower in d.name.lower():
            if d not in search_dirs:
                search_dirs.append(d)

    # Collect files
    for d in search_dirs:
        for f in d.rglob("*.txt"):
            files.append(f)

    files = list(set(files))
    print(f"[PPT] Subject={subject}: Found {len(files)} PPT files in {len(search_dirs)} directories")
    return files


def _find_ppt_files_for_chapter(
    subject: str,
    chapter_id: str,
    chapter_name: str,
    textbook_text: str | None = None,
) -> List[Path]:
    """
    Find PPT files relevant to a specific chapter using intelligent matching.

    Strategy:
    1. Collect all PPT files for the subject
    2. Extract keywords from chapter content
    3. Score each PPT file for relevance
    4. Return files above threshold
    """
    all_ppt_files = _collect_ppt_files(subject)
    if not all_ppt_files:
        return []

    # Prepare keywords
    query_parts = [chapter_name]
    if textbook_text:
        query_parts.append(textbook_text)
    query_text = "\n".join(query_parts)

    keywords = _extract_keywords(query_text, top_k=30)
    if not keywords:
        keywords = chapter_name.split()

    # Score all files
    scored: List[tuple[Path, int]] = []
    for f in all_ppt_files:
        sc = _score_ppt_file(f, keywords, chapter_id, chapter_name)
        if sc > 0:
            scored.append((f, sc))

    if not scored:
        print(f"[PPT] No matching PPT files for {subject} Chapter {chapter_id}")
        return []

    # Sort and filter
    scored.sort(key=lambda x: x[1], reverse=True)
    best_score = scored[0][1]

    # Dynamic threshold: 40% of best score
    threshold = max(5, int(best_score * 0.4))

    selected = [f for f, sc in scored if sc >= threshold]

    print(f"[PPT] {subject} Chapter {chapter_id}: Selected {len(selected)} PPTs (best score: {best_score})")
    return selected


def generate_ppt_notes(
    subject: str,
    chapter_id: str,
    chapter_name: str,
    api_key: Optional[str] = None,
) -> None:
    """
    Generate integrated lecture notes for a chapter.

    Combines PPT content with textbook material using LLM.
    """
    base_dir = OUTPUT_DIR / subject
    raw_dir = base_dir / "raw"
    out_dir = base_dir / "chapters"
    out_dir.mkdir(parents=True, exist_ok=True)

    chap_id_str = str(chapter_id).zfill(2)
    out_name = f"{chap_id_str}_{chapter_name}_lecture_integrated.md"
    out_path = out_dir / out_name

    # Chapter-level caching
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[PPT] SKIP: {out_name} already exists")
        return

    # Load textbook content
    textbook_text = ""
    for suffix in ["_textbook.txt", "_content.txt", "_combined.txt"]:
        candidate = raw_dir / f"{chap_id_str}_{chapter_name}{suffix}"
        if candidate.exists():
            textbook_text = _read_text_if_exists(candidate, limit=8000)
            break

    # Find relevant PPT files
    ppt_files = _find_ppt_files_for_chapter(
        subject, chap_id_str, chapter_name, textbook_text=textbook_text
    )

    if not ppt_files:
        print(f"[PPT] No PPT files found for {subject} Chapter {chap_id_str}, skipping")
        return

    # Collect PPT content
    ppt_blocks = []
    for f in ppt_files:
        txt = _read_text_if_exists(f, limit=6000)
        ppt_blocks.append(f"[Source: {f.name}]\n{txt}")

    ppt_text = "\n\n".join(ppt_blocks)

    # Build prompt
    debug_id = f"PPT-{subject}-{chap_id_str}"

    user_prompt = f"""
Below are the OCR text materials for this chapter. Please generate integrated lecture notes.

[PPT Slides OCR Text]
{ppt_text}

[Textbook OCR Text]
{textbook_text or "(Textbook content not available - generate notes based on PPT only)"}

Please generate comprehensive lecture notes in Markdown format.
"""

    full_prompt = PPT_SYSTEM_PROMPT + "\n\n" + textwrap.dedent(user_prompt).strip()

    # Call LLM
    response = call_llm_with_smart_routing(full_prompt, debug_id=debug_id, api_key=api_key)
    if not response:
        print(f"[PPT] LLM returned empty for {subject} Chapter {chap_id_str}")
        return

    response = response.strip()

    # Remove code blocks if present
    if response.startswith("```"):
        lines = response.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        response = "\n".join(lines).strip()

    # Clean duplicates
    original_len = len(response)
    response = _clean_markdown_duplicates(response)
    if len(response) < original_len:
        print(f"[PPT] Dedup: {original_len} -> {len(response)} chars")

    # Write output
    atomic_write_text(out_path, response, encoding="utf-8")
    print(f"[PPT] Generated: {out_name} ({len(response)} chars)")
