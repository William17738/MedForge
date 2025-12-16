"""
MedForge Exercise Processing Pipeline

Generates comprehensive exercise solutions with detailed explanations.
Uses LLM for answer validation and explanation generation.
"""

import json
import re
import random
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm_client import call_llm_with_smart_routing
from config import (
    OUTPUT_DIR,
    THREADS_PER_PROCESS,
    NUM_PROCESSES,
    EXERCISES_CHAPTER_SUFFIX,
    LEGACY_EXERCISES_CHAPTER_SUFFIXES,
)
from utils_text import normalize_text


def _maybe_migrate_legacy_exercises_output(out_dir: Path, chapter_id: str, chapter_name: str) -> None:
    """
    If a legacy chapter output exists, rename/copy it to the canonical suffix.

    This avoids downstream mismatches when users have older `_exercises_solved.md` artifacts.
    """
    out_file = out_dir / f"{chapter_id}_{chapter_name}{EXERCISES_CHAPTER_SUFFIX}.md"
    if out_file.exists():
        return

    for legacy_suffix in LEGACY_EXERCISES_CHAPTER_SUFFIXES:
        legacy_file = out_dir / f"{chapter_id}_{chapter_name}{legacy_suffix}.md"
        if not legacy_file.exists():
            continue

        try:
            legacy_file.rename(out_file)
            print(f"[MIGRATE] {legacy_file.name} -> {out_file.name}")
        except Exception as e:
            try:
                out_file.write_text(legacy_file.read_text(encoding="utf-8"), encoding="utf-8")
                print(f"[MIGRATE] Copied {legacy_file.name} -> {out_file.name} (rename failed: {e})")
            except Exception:
                pass
        return


# System prompt for exercise explanation generation
SYSTEM_PROMPT = """You are an educational AI assistant responsible for generating comprehensive exercise solutions.

Please follow the Exercise Processing Protocol:

1. **Source Text Priority**:
   - Always prioritize the provided "original answer" and "original explanation"
   - Only correct final_answer when the original has obvious errors (e.g., logic completely reversed)
   - Always preserve original_answer even when correcting
   - Only rewrite explanations when original is missing, extremely brief, or has scientific errors

2. **One Question at a Time**:
   - Process only one question per request
   - Do not skip or merge questions

3. **Output Format**:
   - Output valid JSON format without Markdown code block markers
   - JSON structure must contain:
     {
       "final_answer": "Corrected final answer (letter)",
       "original_answer": "Original book answer (if available)",
       "final_expl_markdown": "Detailed explanation in Markdown format"
     }

4. **Explanation Requirements**:
   - **Smart Correction**: If original answer is A but explanation supports B, set final_answer to B and note "(Original answer A appears incorrect, corrected to B)"
   - **Cite Authority**: Include textbook references in explanations
   - **Option Analysis**: Analyze why correct options are right AND why incorrect options are wrong
   - **Key Highlighting**: Use **bold** for core concepts and key terms
   - **Frequency Marking**: Add "⭐ High-frequency" at start if question is classic/common

5. **Manual Review Flag**:
   - If OCR quality is too poor to understand, output "> OCR unclear, requires manual review" in final_expl_markdown
"""


def validate_brush_result(data: dict, q: dict) -> tuple[bool, str]:
    """
    Validate generated exercise result for correctness.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not data:
        return False, "JSON is empty"

    final_ans = (data.get("final_answer") or "").strip().upper()
    raw_ans = (q.get("raw_answer") or "").strip().upper()
    expl = (data.get("final_expl_markdown") or "").strip()

    # 1) final_answer must be non-empty and contain only A-E
    if not final_ans:
        return False, "final_answer is empty"
    if any(ch not in "ABCDE" for ch in final_ans):
        return False, f"final_answer contains invalid options: {final_ans}"

    # 2) Options must exist in question
    option_keys = set(q.get("options", {}).keys())
    if not set(final_ans).issubset(option_keys):
        return False, f"final_answer has non-existent options: {final_ans} vs {option_keys}"

    # 3) Explanation must have minimum length
    if len(expl) < 20:
        return False, "Explanation text too short"

    # 4) If original answer exists and completely different from final_answer,
    #    explanation must mention the discrepancy
    if raw_ans and not (set(raw_ans) & set(final_ans)):
        if ("original answer" not in expl.lower()) and ("corrected" not in expl.lower()):
            return False, "final_answer differs from original without explanation"

    # 5) Force backfill original_answer if LLM missed it
    if raw_ans and data.get("original_answer") != raw_ans:
        data["original_answer"] = raw_ans

    # 6) Normalize final_answer order
    data["final_answer"] = "".join(sorted(set(final_ans)))

    return True, ""


def extract_json_block(raw: str) -> dict | None:
    """
    Extract the first valid JSON object from model output.

    Handles:
    - ```json ... ``` wrapped content
    - Output with surrounding explanation text
    """
    if not raw:
        return None

    text = raw.strip()

    # Remove ```json ``` code block wrappers
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # 1) Try parsing entire text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2) Try finding { ... } fragments
    for m in re.finditer(r"\{.*?\}", text, flags=re.DOTALL):
        snippet = m.group(0)
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            continue

    return None


def ask_llm_with_repair(q: dict, tb_context: str, subject: str, chapter_name: str, api_key=None):
    """
    LLM call with error correction retry mechanism.
    """
    MAX_ATTEMPTS_PER_QUESTION = 3

    stem = normalize_text(q.get("stem", ""))
    options_norm = {k: normalize_text(v) for k, v in q.get("options", {}).items()}
    debug_id_base = f"{subject}-{chapter_name}-Q{q['id']}"

    last_error = ""
    last_raw_resp = ""

    for attempt in range(1, MAX_ATTEMPTS_PER_QUESTION + 1):
        debug_id = f"{debug_id_base}-try{attempt}"

        # Base user prompt
        user_prompt = f"""
Textbook excerpt (reference):
{tb_context}

Question:
{stem}

Options:
{json.dumps(options_norm, ensure_ascii=False)}

Original Answer: {q.get('raw_answer', 'Unknown')}
Original Explanation: {q.get('raw_expl', 'None')}
"""

        # If retry, include previous error
        if last_error:
            user_prompt += f"""

[IMPORTANT] Your previous JSON had the following issue, please correct:
{last_error}

Your previous raw output (for reference only, do not copy errors):
{last_raw_resp}
"""

        user_prompt += """

Please generate a JSON solution following the Exercise Processing Protocol.
"""

        full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"

        resp = call_llm_with_smart_routing(full_prompt, debug_id, api_key=api_key)
        if not resp:
            last_error = "LLM no response"
            last_raw_resp = ""
            continue

        last_raw_resp = resp
        data = extract_json_block(resp)
        ok, err_msg = validate_brush_result(data, q)
        if ok:
            return data

        last_error = err_msg

    # All attempts failed
    raw_ans = (q.get("raw_answer") or "").strip().upper() or "?"
    return {
        "final_answer": raw_ans,
        "original_answer": q.get("raw_answer", ""),
        "final_expl_markdown": f"> Multiple LLM attempts failed to produce reliable solution (last error: {last_error}). Manual review recommended."
    }


# ================= Question Quality Classification System =================

BAD_EXPL_PATTERNS = [
    r"OCR unclear",
    r"requires manual review",
    r"Multiple LLM attempts failed",
    r"manual review recommended",
    r"cannot parse",
    r"options missing",
    r"serious issue",
]


def is_structurally_good(q: dict) -> tuple[bool, list[str]]:
    """
    Structural filter: check question stem and options.

    Returns:
        Tuple of (is_valid, list_of_reasons)
    """
    reasons = []
    opts = q.get("options", {}) or {}
    stem = (q.get("stem") or "").strip()

    # 1) At least 2 options
    if len(opts) < 2:
        reasons.append("option count < 2")

    # 2) Check valid options count
    valid_opts_count = 0
    for k, v in opts.items():
        txt = v.strip()
        if len(txt) >= 1 and re.search(r"[A-Za-z0-9\u4e00-\u9fa5]", txt):
            valid_opts_count += 1

    if valid_opts_count < 2:
        reasons.append("too few valid options")

    # 3) Stem must have minimum content
    if len(stem) < 4:
        reasons.append("stem too short")

    return (len(reasons) == 0, reasons)


def is_expl_bad(expl: str) -> bool:
    """Check if explanation is flagged for manual review."""
    for pat in BAD_EXPL_PATTERNS:
        if re.search(pat, expl, re.IGNORECASE):
            return True
    return False


def has_basic_semantics(expl: str, final_answer: str) -> bool:
    """Check if explanation actually discusses the question."""
    # Has at least one common explanation keyword
    keywords = r"(this question|correct|incorrect|option|therefore|answer|because|since)"
    if not re.search(keywords, expl, re.IGNORECASE):
        return False
    # Mentions the correct answer letter at least once
    if not any(ch in expl for ch in final_answer):
        return False
    return True


def is_explanation_good(data: dict, q: dict) -> tuple[bool, list[str]]:
    """
    Explanation quality filter: check LLM output quality.
    """
    ok, err_msg = validate_brush_result(data, q)
    reasons = []
    if not ok:
        reasons.append(err_msg)

    expl = (data.get("final_expl_markdown") or "").strip()
    final_ans = (data.get("final_answer") or "").strip().upper()

    if is_expl_bad(expl):
        reasons.append("Explanation flagged for manual review/LLM failed")

    if not has_basic_semantics(expl, final_ans):
        reasons.append("Explanation lacks basic semantics")

    return (len(reasons) == 0, reasons)


def classify_question(q: dict, brush_data: dict) -> tuple[str, list[str]]:
    """
    Final classification: good/bad question.
    """
    # 1) Structure check
    struct_ok, struct_reasons = is_structurally_good(q)

    # 2) Explanation check
    expl_ok, expl_reasons = is_explanation_good(brush_data, q)

    reasons = []
    if not struct_ok:
        reasons += struct_reasons
    if not expl_ok:
        reasons += expl_reasons

    if struct_ok and expl_ok:
        return "good", []
    else:
        return "bad", reasons


def process_single_question(
    subject: str,
    chapter_id: str,
    chapter_name: str,
    tb_context: str,
    q: dict,
    cache_dir: Path,
) -> tuple[int, bool, str]:
    """
    Process a single question:
    - Call LLM
    - Generate MD
    - Write to cache_dir / "{q_id}.md"

    Returns:
        Tuple of (qid, success, error_msg)
    """
    q_id = q["id"]
    try:
        stem = normalize_text(q.get("stem", ""))
        options_norm = {k: normalize_text(v) for k, v in q.get("options", {}).items()}

        data = ask_llm_with_repair(
            q,
            tb_context,
            subject,
            chapter_name,
            api_key=None,
        )
        expl = data.get("final_expl_markdown", "> Explanation generation failed")
        final_ans = data.get("final_answer", q.get("raw_answer", "?"))

        md_block = ""
        md_block += f"### {q_id}. {stem}\n\n"
        for k, v in options_norm.items():
            md_block += f"- **{k}**. {v}\n"
        md_block += "\n"
        md_block += f"> **Answer**: {final_ans}\n\n"
        md_block += expl.strip() + "\n\n---\n\n"

        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{q_id}.md").write_text(md_block, encoding="utf-8")

        return q_id, True, ""
    except Exception as e:
        return q_id, False, str(e)


def assemble_chapter_from_cache(
    subject: str,
    chapter_id: str,
    chapter_name: str,
) -> None:
    """
    Assemble chapter from cached question files:
    - Check for duplicate question IDs
    - Apply structure + explanation filters
    - Combine into chapter MD file
    """
    base_dir = OUTPUT_DIR / subject
    struct_dir = base_dir / "questions_structured"
    out_dir = base_dir / "chapters"
    cache_dir = base_dir / "cache" / "brush" / f"{chapter_id}_{chapter_name}"

    out_dir.mkdir(parents=True, exist_ok=True)
    _maybe_migrate_legacy_exercises_output(out_dir, chapter_id, chapter_name)
    out_file = out_dir / f"{chapter_id}_{chapter_name}{EXERCISES_CHAPTER_SUFFIX}.md"

    if out_file.exists():
        print(f"[SKIP-ASSEMBLE] {out_file} already exists, skipping assembly.")
        return

    q_file = struct_dir / f"{chapter_id}_{chapter_name}_questions.json"
    if not q_file.exists():
        print(f"[WARN] Cannot assemble: question file not found {q_file}")
        return

    questions = json.loads(q_file.read_text(encoding="utf-8"))

    # 1) Duplicate ID check
    id_counts = Counter(q["id"] for q in questions)
    dup_ids = [qid for qid, cnt in id_counts.items() if cnt > 1]
    if dup_ids:
        print(f"[WARN] {q_file.name} has duplicate question IDs: {dup_ids}")

    # 2) Post-processing filter
    valid_questions = []
    skipped_questions = []

    for q in questions:
        qid = q["id"]
        cache_path = cache_dir / f"{qid}.md"
        if not cache_path.exists():
            skipped_questions.append((qid, "LLM processing failed or cache not generated"))
            continue

        try:
            md_content = cache_path.read_text(encoding="utf-8")
            brush_data = {
                "final_expl_markdown": md_content,
                "final_answer": "A",
                "original_answer": q.get("raw_answer", "")
            }
            # Extract final_answer from MD
            answer_match = re.search(r"\*\*Answer\*\*:\s*([A-E]+)", md_content)
            if answer_match:
                brush_data["final_answer"] = answer_match.group(1)

            cls, reasons = classify_question(q, brush_data)
            if cls == "good":
                valid_questions.append(q)
            else:
                skipped_questions.append((qid, "; ".join(reasons)))
        except Exception as e:
            skipped_questions.append((qid, f"Cache read error: {e}"))

    if skipped_questions:
        print(
            f"[BRUSH] {chapter_id}_{chapter_name} post-filter skipped "
            f"{len(skipped_questions)} questions (based on LLM output quality)."
        )

    if not valid_questions:
        print(f"[BRUSH] {chapter_id}_{chapter_name} all questions filtered out, skipping chapter.")
        return

    # 3) Assemble chapter MD
    final_md_content = f"# {chapter_name} Exercise Solutions\n\n"

    ordered_ids = []
    seen = set()
    for q in valid_questions:
        qid = q["id"]
        if qid not in seen:
            seen.add(qid)
            ordered_ids.append(qid)

    # Map to consecutive numbering
    id2display = {}
    display_idx = 1
    for qid in ordered_ids:
        id2display[qid] = display_idx
        display_idx += 1

    for qid in ordered_ids:
        cache_path = cache_dir / f"{qid}.md"
        display_idx = id2display.get(qid, qid)
        if cache_path.exists():
            md = cache_path.read_text(encoding="utf-8")
            md = re.sub(
                r"^###\s*\d+\.",
                f"### {display_idx}.",
                md,
                count=1,
                flags=re.MULTILINE,
            )
            final_md_content += md
        else:
            final_md_content += f"### {display_idx}. Explanation missing\n\n---\n\n"

    out_file.write_text(final_md_content, encoding="utf-8")
    print(f"[BRUSH] {chapter_id}_{chapter_name} assembly complete -> {out_file}")


def run_subject_questions_global(subject: str) -> None:
    """
    Subject-wide exercise processing entry point:
    - Scan all chapters
    - Collect all pending questions as global tasks
    - Use a large thread pool (NUM_PROCESSES * THREADS_PER_PROCESS) for work stealing
    - Assemble by chapter
    """
    base_dir = OUTPUT_DIR / subject
    struct_dir = base_dir / "questions_structured"
    raw_dir = base_dir / "raw"

    if not struct_dir.exists():
        print(f"[GLOBAL] {subject} has no questions_structured, skipping.")
        return

    # 1) Collect all chapters + pending questions
    chapter_files = sorted(struct_dir.glob("*_questions.json"))
    global_tasks = []

    print(f"[GLOBAL] Scanning {subject} all chapter cache status...")

    for qf in chapter_files:
        stem = qf.stem.replace("_questions", "")
        parts = stem.split("_", 1)
        if len(parts) == 2:
            chap_id, chap_name = parts
        else:
            chap_id, chap_name = "00", stem

        cache_dir = base_dir / "cache" / "brush" / f"{chap_id}_{chap_name}"

        try:
            questions = json.loads(qf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Cannot read {qf}: {e}")
            continue

        # Textbook context
        tb_context = ""
        tb_file_full = raw_dir / f"{chap_id}_{chap_name}_textbook.txt"
        tb_file_content = raw_dir / f"{chap_id}_{chap_name}_content.txt"
        if tb_file_full.exists():
            tb_context = tb_file_full.read_text(encoding="utf-8")[:3000]
        elif tb_file_content.exists():
            tb_context = tb_file_content.read_text(encoding="utf-8")[:2000]
        tb_context = normalize_text(tb_context)

        done_ids = {int(p.stem) for p in cache_dir.glob("*.md")}
        pending_questions = [q for q in questions if q["id"] not in done_ids]

        for q in pending_questions:
            global_tasks.append((chap_id, chap_name, tb_context, q, cache_dir))

    if not global_tasks:
        print(f"[GLOBAL] {subject} all questions cached, proceeding to assembly.")
    else:
        total = len(global_tasks)
        workers = max(1, NUM_PROCESSES * THREADS_PER_PROCESS)
        print(
            f"[GLOBAL] {subject} subject-wide pending questions={total}, "
            f"starting thread pool workers={workers} for question-level work stealing."
        )

        # 2) Global question-level thread pool
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut2info = {}
            for idx, (chap_id, chap_name, tb_context, q, cache_dir) in enumerate(global_tasks):
                fut = ex.submit(
                    process_single_question,
                    subject,
                    chap_id,
                    chap_name,
                    tb_context,
                    q,
                    cache_dir,
                )
                fut2info[fut] = (chap_id, chap_name, q["id"])

            finished = 0
            for fut in as_completed(fut2info):
                chap_id, chap_name, qid = fut2info[fut]
                try:
                    qid_ret, success, err = fut.result()
                    finished += 1
                    if success:
                        print(
                            f"[GLOBAL-OK] {subject} / {chap_name} Q{qid_ret} "
                            f"done ({finished}/{total})"
                        )
                    else:
                        print(
                            f"[GLOBAL-FAIL] {subject} / {chap_name} Q{qid_ret}: {err} "
                            f"({finished}/{total})"
                        )
                except Exception as e:
                    finished += 1
                    print(
                        f"[GLOBAL-EXC] {subject} / {chap_name} Q{qid}: {e} "
                        f"({finished}/{total})"
                    )

    # 3) Assemble all chapters
    print(f"[GLOBAL] Starting assembly for {subject} all chapters...")
    for qf in chapter_files:
        stem = qf.stem.replace("_questions", "")
        parts = stem.split("_", 1)
        if len(parts) == 2:
            chap_id, chap_name = parts
        else:
            chap_id, chap_name = "00", stem

        assemble_chapter_from_cache(subject, chap_id, chap_name)

    print(f"[GLOBAL] {subject} subject-wide exercise processing + chapter assembly complete.")


def generate_explanations(subject, chapter_id, chapter_name, api_key=None):
    """
    Generate explanations for a single chapter.
    """
    base_dir = OUTPUT_DIR / subject
    struct_dir = base_dir / "questions_structured"
    raw_dir = base_dir / "raw"
    out_dir = base_dir / "chapters"
    cache_dir = base_dir / "cache" / "brush" / f"{chapter_id}_{chapter_name}"

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Chapter-level caching
    _maybe_migrate_legacy_exercises_output(out_dir, chapter_id, chapter_name)
    out_file = out_dir / f"{chapter_id}_{chapter_name}{EXERCISES_CHAPTER_SUFFIX}.md"
    if out_file.exists():
        print(f"[SKIP] {out_file} already exists, skipping chapter.")
        return

    q_file = struct_dir / f"{chapter_id}_{chapter_name}_questions.json"

    if not q_file.exists():
        print(f"Questions file not found: {q_file}")
        return

    questions = json.loads(q_file.read_text(encoding="utf-8"))

    # Load textbook content for context
    tb_context = ""
    tb_file_full = raw_dir / f"{chapter_id}_{chapter_name}_textbook.txt"
    tb_file_content = raw_dir / f"{chapter_id}_{chapter_name}_content.txt"

    if tb_file_full.exists():
        tb_context = tb_file_full.read_text(encoding="utf-8")[:3000]
    elif tb_file_content.exists():
        tb_context = tb_file_content.read_text(encoding="utf-8")[:2000]

    tb_context = normalize_text(tb_context)

    # Phase 2: Per-question cache filtering
    done_ids = {int(p.stem) for p in cache_dir.glob("*.md")}
    pending_questions = [q for q in questions if q["id"] not in done_ids]

    print(f"[{chapter_name}] Total: {len(questions)}, Cached: {len(done_ids)}, Pending: {len(pending_questions)} (parallel {THREADS_PER_PROCESS})...")

    def handle_question(q: dict):
        q_id, success, err = process_single_question(
            subject,
            chapter_id,
            chapter_name,
            tb_context,
            q,
            cache_dir,
        )
        if success:
            print(f"  - [{chapter_name}] Q{q_id} done.")
        else:
            print(f"  - [{chapter_name}] Q{q_id} failed: {err}")
        return q_id, success

    # Parallel Execution (Chapter Level)
    if pending_questions:
        with ThreadPoolExecutor(max_workers=THREADS_PER_PROCESS) as ex:
            fut2id = {ex.submit(handle_question, q): q["id"] for q in pending_questions}
            for fut in as_completed(fut2id):
                q_id = fut2id[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"Error processing Q{q_id}: {e}")

    # Assemble final chapter MD
    assemble_chapter_from_cache(subject, chapter_id, chapter_name)
