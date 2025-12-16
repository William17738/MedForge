"""
MedForge OCR Question Parser

Parses OCR text from exercise files into structured question format.
Uses LLM fallback for re-segmentation when parsing quality is poor.
"""

import re
import json
import concurrent.futures
from pathlib import Path

from config import OUTPUT_DIR
from utils_text import normalize_text
from llm_client import call_llm_with_smart_routing

MAX_LLM_RESEG_ATTEMPTS = 3


def read_text_safely(path: Path) -> str:
    """
    Try multiple common encodings to read file, avoiding data loss.
    """
    for enc in ("utf-8", "gbk", "gb18030"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    # Fallback with ignore
    return path.read_text(encoding="utf-8", errors="ignore")


def _repair_order_and_ids(questions: list[dict]) -> list[dict]:
    """
    Normalize question order within chapter:
    - Sort by original ID
    - Save raw_id
    - Reassign ID to consecutive 1..N (used by downstream pipeline)
    """
    if not questions:
        return questions

    # Add index for stable sort
    for idx, q in enumerate(questions):
        q["_index"] = idx

    questions_sorted = sorted(
        questions,
        key=lambda q: (q.get("id", 0), q["_index"])
    )

    for new_id, q in enumerate(questions_sorted, 1):
        q["raw_id"] = q["id"]  # Preserve original question number for reference
        q["id"] = new_id       # Pipeline uses consecutive IDs internally
        q.pop("_index", None)

    return questions_sorted


def parse_file(file_path: Path) -> list[dict]:
    """
    Parse OCR text file into structured questions.
    """
    print(f"Parsing {file_path}")
    raw = read_text_safely(file_path)
    content = normalize_text(raw)

    # 1. Locate Answer Section (supports various chapter formats)
    split_match = re.search(
        r'(?:^|\n)\s*(?:Chapter|Section|Part)\s*\d*\s*(?:Answer|Solution|Key)',
        content,
        re.IGNORECASE
    )

    if split_match:
        questions_part = content[:split_match.start()]
        answers_part = content[split_match.start():]
    else:
        questions_part = content
        answers_part = ""

    # 2. Extract Answers (MCQ only: ID + A-E, supports multi-select like ABC)
    answer_map = {}
    ans_matches = re.findall(r'(?:^|\s|、|．|\.)(\d+)\s*[\.．、\s]\s*([A-E]{1,5})(?![a-z])', answers_part)
    for num, ans in ans_matches:
        answer_map[int(num)] = ans

    # 3. Parse Questions
    questions = []
    lines = questions_part.split('\n')
    current_q = None

    q_start_pattern = re.compile(r'^\s*(\d+)\s*[\.．、](.*)')
    opt_pattern = re.compile(r'^\s*([A-E])\s*[\.．、](.*)')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        opt_match = opt_pattern.match(line)
        if opt_match and current_q:
            label = opt_match.group(1)
            text = opt_match.group(2).strip()
            current_q['options'][label] = text
            current_q['raw_block'] += "\n" + line
            continue

        q_match = q_start_pattern.match(line)
        if q_match:
            if current_q:
                # Filter: Only keep if it has options (MCQ)
                if current_q['options']:
                    questions.append(current_q)

            q_id = int(q_match.group(1))
            q_stem = q_match.group(2).strip()
            ans = answer_map.get(q_id, "")

            current_q = {
                "id": q_id,
                "stem": q_stem,
                "options": {},
                "raw_block": line,
                "raw_answer": ans,
                "raw_expl": "",
                "flags": []
            }
            continue

        if current_q:
            if current_q['options']:
                last_key = list(current_q['options'].keys())[-1]
                current_q['options'][last_key] += " " + line
            else:
                current_q['stem'] += " " + line
            current_q['raw_block'] += "\n" + line

    if current_q and current_q['options']:
        questions.append(current_q)

    # Normalize order after parsing
    questions = _repair_order_and_ids(questions)
    return questions


def _needs_llm_repair(questions: list[dict]) -> bool:
    """
    Relaxed quality check: only call LLM when severely broken.
    - No questions: definitely a problem
    - Missing options ratio > 30%: likely OCR failure
    - Duplicate ID ratio > 30%: possible block misalignment
    - Non-increasing edges > 50%: completely scrambled
    """
    if not questions:
        return True

    ids = [q["id"] for q in questions]
    n = len(ids)

    # Duplicate ID ratio
    dup_ratio = 1 - (len(set(ids)) / n)

    # Missing options ratio
    missing_opts_ratio = sum(
        1 for q in questions if len(q.get("options", {})) < 2
    ) / n

    # Non-strictly-increasing edge ratio
    non_inc_count = sum(
        1 for i in range(n - 1) if ids[i] >= ids[i + 1]
    )
    non_inc_ratio = non_inc_count / max(n - 1, 1)

    if missing_opts_ratio > 0.3:
        return True
    if dup_ratio > 0.3:
        return True
    if non_inc_ratio > 0.5:
        return True

    # Minor disorder / occasional duplicate is acceptable
    return False


def _quality_report(questions: list[dict]) -> str:
    """Return simple quality issue description for LLM retry prompt."""
    if not questions:
        return "No questions parsed."
    ids = [q["id"] for q in questions]
    dup_ids = [i for i in ids if ids.count(i) > 1]
    non_inc = [f"{ids[i]}->{ids[i+1]}" for i in range(len(ids)-1) if ids[i] >= ids[i+1]]
    missing_opts = [q["id"] for q in questions if len(q.get("options", {})) < 2]
    issues = []
    if dup_ids:
        issues.append(f"Duplicate IDs: {sorted(set(dup_ids))}")
    if non_inc:
        issues.append(f"Non-increasing IDs: {non_inc[:5]}")
    if missing_opts:
        issues.append(f"Questions missing options: {missing_opts[:10]}")
    if not issues:
        issues.append("No obvious issues but quality check failed.")
    return "; ".join(issues)


def _extract_json_array_from_text(resp: str) -> str | None:
    """
    Extract JSON array string from LLM output.

    Handles:
    1. Content in ```json...``` or ```...``` code blocks
    2. Plain text, extract from first '[' to last ']'
    """
    text = resp.strip()

    # 1) Prefer extracting from code fence
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            inner = m.group(1).strip()
            if inner.startswith("[") and inner.endswith("]"):
                return inner
            text = inner

    # 2) Extract first '[' to last ']' from full text
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None

    return text[start:end + 1]


def llm_resegment(file_path: Path, feedback: str = "") -> list[dict] | None:
    """
    Call LLM to re-segment entire chapter's questions.

    Returns:
        Normalized questions list on success, None on failure.
    """
    raw = file_path.read_text(encoding="utf-8", errors="ignore")
    prompt = f"""You are an OCR post-processing assistant. Extract multiple-choice questions from the raw OCR text below and output a JSON array (without code block markers).

Each element should contain:
- id: question number (integer)
- stem: question stem text
- options: option dictionary, e.g., {{"A": "...", "B": "..."}}
- raw_answer: original answer string (leave empty if not provided)

Requirements:
- Strictly follow original text, do not invent or modify questions
- Do not fabricate missing options or answers
- Leave empty if options or answers are missing

Raw text:
{raw}
"""
    if feedback:
        prompt = f"{prompt}\n\n[Previous parsing issues] {feedback}\nPlease fix these issues in your output."

    try:
        resp = call_llm_with_smart_routing(
            prompt,
            debug_id=f"LLM-SEG-{file_path.name}"
        )
        if not resp:
            return None

        json_text = _extract_json_array_from_text(resp)
        if not json_text:
            preview = resp.replace("\n", "\\n")[:200]
            print(f"[LLM-SEG] {file_path.name}: No JSON array found in output. First 200 chars: {preview!r}")
            return None

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            preview = json_text.replace("\n", "\\n")[:200]
            print(f"[LLM-SEG] {file_path.name}: JSON parse failed: {e}. First 200 chars: {preview!r}")
            return None

        if not isinstance(data, list):
            print(f"[LLM-SEG] {file_path.name}: Result is not a list, got {type(data)}")
            return None

        # Normalize fields
        normalized = []
        for item in data:
            try:
                qid = int(item.get("id"))
            except Exception:
                # Skip items without valid ID
                continue

            stem = (item.get("stem") or "").strip()
            options = item.get("options") or {}
            if not isinstance(options, dict):
                options = {}

            raw_answer = (item.get("raw_answer") or "").strip()

            normalized.append({
                "id": qid,
                "stem": stem,
                "options": options,
                "raw_block": "",
                "raw_answer": raw_answer,
                "raw_expl": "",
                "flags": ["llm_resegment"],
            })

        return normalized

    except Exception as e:
        print(f"[LLM-SEG] Failed on {file_path.name}: {e}")
        return None


def get_output_name(filename: str) -> str:
    """Convert various input file formats to _questions.json format."""
    if filename.endswith("_exercises.txt"):
        return filename.replace("_exercises.txt", "_questions.json")
    elif filename.endswith("_guide_full.txt"):
        return filename.replace("_guide_full.txt", "_questions.json")
    else:
        # Fallback: remove extension and add _questions.json
        return Path(filename).stem + "_questions.json"


def process_single_file(f: Path, base_dir: Path, struct_dir: Path) -> None:
    """
    Process a single file: parse questions and optionally use LLM repair.
    """
    try:
        qs = parse_file(f)
        out_name = get_output_name(f.name)
        out_path = struct_dir / out_name

        if _needs_llm_repair(qs):
            print(f"[WARN] {f.name} parsing quality poor, attempting LLM re-segmentation...")
            feedback = _quality_report(qs)
            best_qs = qs
            any_llm_success = False

            for attempt in range(1, MAX_LLM_RESEG_ATTEMPTS + 1):
                llm_qs = llm_resegment(f, feedback=feedback)
                if not llm_qs:
                    print(f"[LLM] {f.name} attempt {attempt}/{MAX_LLM_RESEG_ATTEMPTS} failed.")
                    continue

                any_llm_success = True

                if not _needs_llm_repair(llm_qs):
                    qs = llm_qs
                    print(f"[LLM] {f.name} re-segmentation successful, got {len(qs)} questions.")
                    break
                else:
                    feedback = _quality_report(llm_qs)
                    best_qs = llm_qs
                    print(f"[LLM] {f.name} attempt {attempt} still has issues: {feedback}")
                    if attempt == MAX_LLM_RESEG_ATTEMPTS:
                        qs = best_qs

            if not any_llm_success:
                print(f"[LLM] {f.name} all re-segmentation attempts failed, using rule-based result, {len(qs)} questions.")

        with open(out_path, 'w', encoding='utf-8') as jf:
            json.dump(qs, jf, ensure_ascii=False, indent=2)
        print(f"Saved {len(qs)} questions to {out_name}")
    except Exception as e:
        print(f"Error parsing {f.name}: {e}")


def run_for_subject(subject: str) -> None:
    """
    Parse all exercise files for a subject.
    """
    base_dir = OUTPUT_DIR / subject
    raw_dir = base_dir / "raw"
    struct_dir = base_dir / "questions_structured"
    struct_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f"Raw dir not found: {raw_dir}")
        return

    # Try to find exercise files
    files = list(raw_dir.glob("*_exercises.txt"))

    # Fallback to guide_full files
    if not files:
        print(f"[INFO] No '*_exercises.txt' found for subject '{subject}'. Falling back to '*_guide_full.txt'.")
        files = list(raw_dir.glob("*_guide_full.txt"))
        if not files:
            print(f"[WARN] No suitable source files ('*_exercises.txt' or '*_guide_full.txt') found for subject '{subject}'.")
            return

    print(f"Found {len(files)} files to process for subject '{subject}'")

    # Use ThreadPoolExecutor for parallel processing
    MAX_WORKERS = 10
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_file, f, base_dir, struct_dir): f for f in files}
        for future in concurrent.futures.as_completed(futures):
            f = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {f.name}: {e}")


if __name__ == "__main__":
    root = OUTPUT_DIR
    for d in root.iterdir():
        if d.is_dir() and (d / "raw").exists():
            print(f"Processing subject: {d.name}")
            run_for_subject(d.name)
