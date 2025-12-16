"""
MedForge Run All

Main orchestrator for the parallel document processing pipeline.
Coordinates preprocessing, three processing pipelines, and final assembly.
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from config import OUTPUT_DIR, SUBJECT_CONFIG_FILE, NUM_PROCESSES
from status_manager import SubjectStatusManager
from preprocessor import main as preprocess_main
from parser_ocr_questions import run_for_subject
from brush_group import run_subject_questions_global
from qpoints_group import generate_question_based_points
from ppt_group import generate_ppt_notes
from final_assembler import assemble_subject


def _iter_chapters(subject_name: str) -> list[tuple[str, str]]:
    """Enumerate chapters from questions_structured directory."""
    struct_dir = OUTPUT_DIR / subject_name / "questions_structured"
    if not struct_dir.exists():
        return []

    chapters = []
    for json_file in struct_dir.glob("*_questions.json"):
        name = json_file.stem.replace("_questions", "")
        if "_" not in name:
            continue
        chapter_id, chapter_name = name.split("_", 1)
        chapters.append((chapter_id, chapter_name))

    return sorted(chapters, key=lambda x: x[0])


def run_questions_pipeline(subjects: list[str]) -> None:
    """
    Step 1: Exercise Processing Pipeline

    - Parse questions from source files
    - Generate explanations using LLM
    """
    print("=== Pipeline 1: Exercise Processing ===")

    # Parse all questions first
    for subject in subjects:
        print(f"[QUESTIONS] Parsing questions for: {subject}")
        try:
            run_for_subject(subject)
        except Exception as e:
            print(f"[QUESTIONS] Parsing failed for {subject}: {e}")

    # Generate explanations (uses global thread pool internally)
    for subject in subjects:
        try:
            run_subject_questions_global(subject)
        except Exception as e:
            print(f"[QUESTIONS] Explanation generation failed for {subject}: {e}")


def run_keypoints_pipeline(subjects: list[str]) -> None:
    """
    Step 2: Key Points Pipeline

    Generate knowledge point summaries based on exercise content.
    """
    print("=== Pipeline 2: Key Points Generation ===")
    _wait_for_questions_structured(subjects, label="KEYPOINTS")

    tasks = []
    for subject in subjects:
        chapters = _iter_chapters(subject)
        for chapter_id, chapter_name in chapters:
            tasks.append((subject, chapter_id, chapter_name))

    if not tasks:
        print("[KEYPOINTS] No tasks found, skipping pipeline")
        return

    print(f"[KEYPOINTS] Processing {len(tasks)} chapters across {len(subjects)} subjects")

    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        futures = {
            executor.submit(generate_question_based_points, subj, cid, cname): (subj, cid, cname)
            for subj, cid, cname in tasks
        }

        for future in as_completed(futures):
            subject, chapter_id, chapter_name = futures[future]
            try:
                future.result()
                print(f"[KEYPOINTS] Completed: {subject} - Chapter {chapter_id}")
            except Exception as e:
                print(f"[KEYPOINTS] Error: {subject} Chapter {chapter_id}: {e}")


def run_ppt_pipeline(subjects: list[str]) -> None:
    """
    Step 3: PPT Integration Pipeline

    Generate integrated lecture notes combining PPT with textbook content.
    """
    print("=== Pipeline 3: PPT Integration ===")
    _wait_for_questions_structured(subjects, label="PPT")

    tasks = []
    for subject in subjects:
        chapters = _iter_chapters(subject)
        for chapter_id, chapter_name in chapters:
            tasks.append((subject, chapter_id, chapter_name))

    if not tasks:
        print("[PPT] No tasks found, skipping pipeline")
        return

    print(f"[PPT] Processing {len(tasks)} chapters across {len(subjects)} subjects")

    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        futures = {
            executor.submit(generate_ppt_notes, subj, cid, cname): (subj, cid, cname)
            for subj, cid, cname in tasks
        }

        for future in as_completed(futures):
            subject, chapter_id, chapter_name = futures[future]
            try:
                future.result()
                print(f"[PPT] Completed: {subject} - Chapter {chapter_id}")
            except Exception as e:
                print(f"[PPT] Error: {subject} Chapter {chapter_id}: {e}")


def discover_subjects() -> list[str]:
    """Discover subjects from config file or output directory."""
    subjects = []

    # Try config file first
    if SUBJECT_CONFIG_FILE.exists():
        print(f"[CONFIG] Loading subjects from {SUBJECT_CONFIG_FILE}")
        try:
            with open(SUBJECT_CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
                subjects = list(config.keys())
        except Exception as e:
            print(f"[CONFIG] Error reading config: {e}")

    # Fallback to directory scan
    if not subjects and OUTPUT_DIR.exists():
        print("[CONFIG] Scanning output directory for subjects...")
        subjects = [
            d.name for d in OUTPUT_DIR.iterdir()
            if d.is_dir() and (d / "raw").exists()
        ]

    return subjects


def _wait_for_questions_structured(
    subjects: list[str],
    *,
    label: str,
    timeout_s: float = 1800.0,
    poll_interval_s: float = 2.0,
) -> None:
    """
    Avoid a race where downstream pipelines start before question JSONs are produced.

    Waits until each subject reports a terminal parsing state ("done"/"error") in
    `output/<subject>/.status/pipeline.json`.
    """
    if not subjects:
        return

    pending = set(subjects)
    start = time.monotonic()
    last_log = 0.0

    while pending:
        for subject in list(pending):
            state = (
                SubjectStatusManager(subject)
                .get_pipeline_status()
                .get("questions_structured_state")
            )
            if state in {"done", "error"}:
                pending.remove(subject)

        if not pending:
            return

        elapsed = time.monotonic() - start
        if elapsed >= timeout_s:
            print(f"[{label}] WARN: Timeout waiting for questions_structured: {sorted(pending)}")
            return

        if elapsed - last_log >= 10.0:
            sample = ", ".join(sorted(pending)[:5])
            suffix = "..." if len(pending) > 5 else ""
            print(f"[{label}] Waiting for question parsing ({len(pending)}): {sample}{suffix}")
            last_log = elapsed

        time.sleep(poll_interval_s)


def main():
    """Main entry point for the processing pipeline."""
    print("=" * 60)
    print("MedForge - Parallel Document Processing Pipeline")
    print("=" * 60)

    # Step 1: Preprocessing
    print("\n=== Step 1: Preprocessing ===")
    preprocess_main()

    # Discover subjects
    subjects = discover_subjects()
    if not subjects:
        print("[ERROR] No subjects found. Exiting.")
        return

    print(f"\n[INFO] Detected subjects: {subjects}")

    # Filter by command line argument if provided
    if len(sys.argv) > 1:
        target = sys.argv[1]
        matching = [s for s in subjects if target.lower() in s.lower()]
        if matching:
            subjects = matching
            print(f"[FILTER] Processing only: {subjects}")
        else:
            print(f"[ERROR] Subject '{target}' not found")
            return

    # Reset per-subject parsing state to avoid stale "done" from previous runs
    for subject in subjects:
        SubjectStatusManager(subject).update_pipeline_status(
            questions_structured_state="pending",
            questions_structured_error=None,
        )

    # Step 2-4: Run all three pipelines in parallel
    print("\n=== Starting Parallel Pipelines ===")
    pipelines = [
        ("Questions", run_questions_pipeline),
        ("KeyPoints", run_keypoints_pipeline),
        ("PPT", run_ppt_pipeline),
    ]

    # NOTE: Use threads here to avoid nested process pools on Windows.
    # The key points and PPT pipelines already use ProcessPoolExecutor internally.
    with ThreadPoolExecutor(max_workers=len(pipelines)) as executor:
        futures = {
            executor.submit(func, subjects): name
            for name, func in pipelines
        }

        for future in as_completed(futures):
            pipeline_name = futures[future]
            try:
                future.result()
                print(f"[PIPELINE] {pipeline_name} completed successfully")
            except Exception as e:
                print(f"[PIPELINE] {pipeline_name} failed: {e}", file=sys.stderr)

    # Step 5: Final Assembly
    print("\n=== Step 5: Final Assembly ===")
    for subject in subjects:
        print(f"[ASSEMBLE] Processing: {subject}")
        assemble_subject(subject)

    print("\n" + "=" * 60)
    print("Processing Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
