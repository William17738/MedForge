"""
MedForge Key Points Pipeline

Generates key knowledge point summaries from processed exercises.
Uses LLM to extract and organize important concepts from questions.
"""

import json
import re
from pathlib import Path

from llm_client import call_llm_with_smart_routing
from config import OUTPUT_DIR, MAX_PROMPT_CHARS
from utils_fs import atomic_write_text
from utils_text import normalize_text, truncate_text


MIN_VALID_CACHE_BYTES = 100


def _is_valid_markdown_cache(path: Path) -> bool:
    try:
        if path.stat().st_size < MIN_VALID_CACHE_BYTES:
            return False
        with path.open("r", encoding="utf-8") as handle:
            head = handle.read(4096).lstrip("\ufeff")
        return head.strip().startswith("#")
    except Exception:
        return False


SYSTEM_PROMPT = """You are an educational content specialist responsible for extracting key knowledge points.

**Core Task**: Based on the chapter's exercises and textbook content, create a comprehensive key points summary.

**Input**:
1. All questions from this chapter (stem, options, answers)
2. Textbook content for this chapter

**Output Requirements**:
1. **Pure Markdown format** - no JSON, no code blocks
2. **Structure**:
   - **Part 1: Chapter Overview** (~100 words on this chapter's importance)
   - **Part 2: Key Concepts** (main section)
     - Organize by topic/category
     - Mark important points with `*`
     - Mark common mistakes with `!`
   - **Part 3: Question Mapping** (required)
     - Create a mapping table:
       | Q# | Key Concept | Notes |
       | :--- | :--- | :--- |
       | Q1 | Concept A | Common mistake |
       | Q2 | Concept B | * High frequency |

3. **Content Principles**:
   - Focus on concepts tested in exercises
   - Use textbook as authoritative reference
"""


def _check_question_coverage(md_text: str, questions: list[dict], debug_id: str) -> None:
    """
    Check if output text covers all question numbers.
    Logs warnings but doesn't affect main flow.
    """
    present: set[int] = set()
    for match in re.finditer(r"Q(\d+)", md_text):
        try:
            present.add(int(match.group(1)))
        except ValueError:
            continue

    missing_ids = [q["id"] for q in questions if q.get("id") not in present]
    if missing_ids:
        print(f"[QPOINTS] WARN: {debug_id} missing Q ids: {missing_ids[:10]}...")


def generate_question_based_points(
    subject: str,
    chapter_id: str,
    chapter_name: str,
    api_key: str | None = None,
) -> Path | None:
    """
    Generate key points summary from questions and textbook content.

    Args:
        subject: Subject name
        chapter_id: Chapter identifier
        chapter_name: Chapter name
        api_key: Optional API key override

    Returns:
        Path to generated file, or None if failed
    """
    base_dir = OUTPUT_DIR / subject
    struct_dir = base_dir / "questions_structured"
    raw_dir = base_dir / "raw"
    out_dir = base_dir / "chapters"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Chapter-level caching
    out_file = out_dir / f"{chapter_id}_{chapter_name}_key_points.md"
    if out_file.exists():
        if _is_valid_markdown_cache(out_file):
            print(f"[QPOINTS] SKIP: {out_file.name} valid cache exists")
            return out_file

        print(f"[QPOINTS] WARN: Invalid cache {out_file.name}, removing and reprocessing")
        try:
            out_file.unlink()
        except Exception as e:
            print(f"[QPOINTS] WARN: Failed to remove invalid cache {out_file.name}: {e}")

    # Load questions
    q_file = struct_dir / f"{chapter_id}_{chapter_name}_questions.json"
    if not q_file.exists():
        print(f"[QPOINTS] Questions file not found: {q_file}")
        return None

    questions: list[dict] = json.loads(q_file.read_text(encoding="utf-8"))

    # Load textbook content (try multiple suffixes)
    textbook_text = ""
    for suffix in ["_textbook.txt", "_content.txt", "_combined.txt"]:
        candidate = raw_dir / f"{chapter_id}_{chapter_name}{suffix}"
        if candidate.exists():
            textbook_text = candidate.read_text(encoding="utf-8", errors="ignore")
            break

    textbook_text = normalize_text(textbook_text)[:8000]

    # Format questions for prompt
    q_blocks: list[str] = []
    for q in questions:
        stem = normalize_text(q.get("stem", ""))
        opts = {k: normalize_text(v) for k, v in q.get("options", {}).items()}
        answer = q.get("raw_answer", "")
        q_blocks.append(
            f"Q{q.get('id')}: {stem}\n"
            f"Options: {json.dumps(opts, ensure_ascii=False)}\n"
            f"Answer: {answer}\n"
        )
    q_text = "\n\n".join(q_blocks)

    debug_id = f"QPOINTS-{subject}-{chapter_id}"

    # Enforce prompt size limits by truncating the largest components first.
    prompt_template = """
================= Chapter Questions =================
{q_text}

================= Textbook Content =================
{textbook_text}

Please generate a key points summary for this chapter, including the question mapping table.
"""
    overhead_prompt = SYSTEM_PROMPT + "\n\n" + prompt_template.strip().format(q_text="", textbook_text="")
    remaining_budget = max(0, MAX_PROMPT_CHARS - len(overhead_prompt))

    textbook_text_for_prompt = (
        textbook_text
        or "(Textbook content not available - summarize based on questions only)"
    )

    if textbook_text:
        q_budget = int(remaining_budget * 0.65)
        tb_budget = max(0, remaining_budget - q_budget)
    else:
        tb_budget = min(len(textbook_text_for_prompt), remaining_budget)
        q_budget = max(0, remaining_budget - tb_budget)

    q_len = len(q_text)
    q_text, q_truncated = truncate_text(q_text, q_budget)
    if q_truncated:
        print(f"[QPOINTS] WARN: {debug_id} truncated q_text {q_len} -> {len(q_text)} chars")

    tb_len = len(textbook_text_for_prompt)
    textbook_text_for_prompt, tb_truncated = truncate_text(textbook_text_for_prompt, tb_budget)
    if tb_truncated:
        print(
            f"[QPOINTS] WARN: {debug_id} truncated textbook_text "
            f"{tb_len} -> {len(textbook_text_for_prompt)} chars"
        )

    # Build prompt
    user_prompt = f"""
================= Chapter Questions =================
{q_text}

================= Textbook Content =================
{textbook_text_for_prompt}

Please generate a key points summary for this chapter, including the question mapping table.
"""

    full_prompt = SYSTEM_PROMPT + "\n\n" + user_prompt
    prompt_len = len(full_prompt)
    full_prompt, prompt_truncated = truncate_text(full_prompt, MAX_PROMPT_CHARS)
    if prompt_truncated:
        print(
            f"[QPOINTS] WARN: {debug_id} truncated full_prompt "
            f"{prompt_len} -> {len(full_prompt)} chars (limit {MAX_PROMPT_CHARS})"
        )

    # Call LLM
    response = call_llm_with_smart_routing(full_prompt, debug_id, api_key=api_key)
    if not response:
        print(f"[QPOINTS] LLM returned empty for {subject} {chapter_id}")
        return None

    response = response.strip()

    # Remove markdown code blocks if present
    if response.startswith("```"):
        lines = response.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        response = "\n".join(lines).strip()

    # Check coverage (warning only)
    _check_question_coverage(response, questions, debug_id)

    # Write output
    atomic_write_text(out_file, response, encoding="utf-8")
    print(f"[QPOINTS] Generated: {out_file.name} ({len(response)} chars)")
    return out_file
