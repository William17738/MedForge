"""
MedForge Key Points Runner

Orchestrates key points generation pipeline.
Generates knowledge point summaries based on exercise content.
"""

import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import OUTPUT_DIR, NUM_PROCESSES
from qpoints_group import generate_question_based_points


def _iter_chapters(subject_name: str) -> list[tuple[str, str]]:
    """
    Enumerate chapters from questions_structured directory.

    Returns:
        List of (chapter_id, chapter_name) tuples
    """
    base_dir = OUTPUT_DIR / subject_name
    struct_dir = base_dir / "questions_structured"

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


def process_subject(subject_name: str) -> None:
    """Process all chapters for key points generation."""
    print(f"[QPOINTS] Processing subject: {subject_name}")

    chapters = _iter_chapters(subject_name)
    if not chapters:
        print(f"[QPOINTS] No chapters found for {subject_name}")
        return

    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        futures = {}
        for chapter_id, chapter_name in chapters:
            future = executor.submit(
                generate_question_based_points,
                subject_name,
                chapter_id,
                chapter_name,
            )
            futures[future] = (chapter_id, chapter_name)

        for future in as_completed(futures):
            chapter_id, chapter_name = futures[future]
            try:
                future.result()
                print(f"[QPOINTS] Completed: Chapter {chapter_id} - {chapter_name}")
            except Exception as e:
                print(f"[QPOINTS] Error in Chapter {chapter_id}: {e}")


def main():
    """Main entry point."""
    if len(sys.argv) >= 2:
        process_subject(sys.argv[1])
    else:
        if OUTPUT_DIR.exists():
            for subject_dir in OUTPUT_DIR.iterdir():
                if subject_dir.is_dir() and (subject_dir / "questions_structured").exists():
                    process_subject(subject_dir.name)
        else:
            print(f"[QPOINTS] Output directory not found: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
