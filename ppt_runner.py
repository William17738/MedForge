"""
MedForge PPT Runner

Orchestrates PPT integration pipeline.
Generates integrated lecture notes combining PPT content with textbook material.
"""

import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import OUTPUT_DIR, NUM_PROCESSES
from ppt_group import generate_ppt_notes


def _collect_chapters(subject_name: str) -> list[tuple[str, str]]:
    """
    Collect chapter information from questions_structured directory.

    Returns:
        List of (chapter_id, chapter_name) tuples, sorted by chapter ID
    """
    base_dir = OUTPUT_DIR / subject_name
    struct_dir = base_dir / "questions_structured"

    if not struct_dir.exists():
        print(f"[PPT] questions_structured not found for {subject_name}")
        return []

    chapters = []
    for question_file in struct_dir.glob("*_questions.json"):
        stem = question_file.stem.replace("_questions", "")
        parts = stem.split("_", 1)
        if len(parts) == 2:
            chapter_id, chapter_name = parts
        else:
            chapter_id, chapter_name = "00", stem
        chapters.append((chapter_id, chapter_name))

    # Sort by chapter number
    def sort_key(item):
        try:
            return int(item[0])
        except ValueError:
            return 999

    chapters = sorted(chapters, key=sort_key)
    print(f"[PPT] Found {len(chapters)} chapters for {subject_name}")

    return chapters


def process_subject(subject_name: str) -> None:
    """
    Process all chapters for PPT integration.

    Generates integrated lecture notes for each chapter in parallel.
    """
    print(f"[PPT] Processing subject: {subject_name}")

    chapters = _collect_chapters(subject_name)
    if not chapters:
        return

    print(f"[PPT] Starting parallel PPT integration: {len(chapters)} chapters, {NUM_PROCESSES} processes")

    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        futures = {
            executor.submit(generate_ppt_notes, subject_name, chapter_id, chapter_name): (chapter_id, chapter_name)
            for chapter_id, chapter_name in chapters
        }

        for future in as_completed(futures):
            chapter_id, chapter_name = futures[future]
            try:
                future.result()
                print(f"[PPT] Completed: Chapter {chapter_id} - {chapter_name}")
            except Exception as e:
                print(f"[PPT] Error in Chapter {chapter_id}: {e}")


def main():
    """Main entry point."""
    if len(sys.argv) >= 2:
        process_subject(sys.argv[1])
    else:
        if OUTPUT_DIR.exists():
            for subject_dir in OUTPUT_DIR.iterdir():
                if not subject_dir.is_dir():
                    continue
                if (subject_dir / "questions_structured").exists() or (subject_dir / "raw").exists():
                    process_subject(subject_dir.name)
        else:
            print(f"[PPT] Output directory not found: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
