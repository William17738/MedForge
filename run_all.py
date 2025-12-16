"""
MedForge Run All

Main orchestrator for the parallel document processing pipeline.
Coordinates preprocessing, three processing pipelines, and final assembly.
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from config import OUTPUT_DIR, SUBJECT_CONFIG_FILE, NUM_PROCESSES
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
        run_for_subject(subject)

    # Generate explanations (uses global thread pool internally)
    for subject in subjects:
        run_subject_questions_global(subject)


def run_keypoints_pipeline(subjects: list[str]) -> None:
    """
    Step 2: Key Points Pipeline

    Generate knowledge point summaries based on exercise content.
    """
    print("=== Pipeline 2: Key Points Generation ===")

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

    # Step 2-4: Run all three pipelines in parallel
    print("\n=== Starting Parallel Pipelines ===")
    pipelines = [
        ("Questions", run_questions_pipeline),
        ("KeyPoints", run_keypoints_pipeline),
        ("PPT", run_ppt_pipeline),
    ]

    with ProcessPoolExecutor(max_workers=3) as executor:
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
