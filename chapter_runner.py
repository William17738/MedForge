"""
MedForge Chapter Runner

Orchestrates chapter-level processing for exercises pipeline.
Handles question parsing and solution generation with parallel execution.
"""

import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import OUTPUT_DIR, NUM_PROCESSES, THREADS_PER_PROCESS
from parser_ocr_questions import run_for_subject
from brush_group import run_subject_questions_global


def process_subject(subject_name: str) -> None:
    """
    Process all chapters for a subject.

    Steps:
    1. Parse questions from source files
    2. Generate explanations using LLM (with work stealing for load balancing)
    """
    print(f"[CHAPTER] Processing subject: {subject_name}")
    print(f"[CHAPTER] Concurrency: {NUM_PROCESSES} processes x {THREADS_PER_PROCESS} threads")

    # Step 1: Parse questions
    run_for_subject(subject_name)

    # Step 2: Generate explanations (global work stealing)
    run_subject_questions_global(subject_name)


def main():
    """Main entry point."""
    if len(sys.argv) >= 2:
        # Process specified subject
        process_subject(sys.argv[1])
    else:
        # Auto-detect subjects from output directory
        if OUTPUT_DIR.exists():
            for subject_dir in OUTPUT_DIR.iterdir():
                if subject_dir.is_dir() and (subject_dir / "raw").exists():
                    process_subject(subject_dir.name)
        else:
            print(f"[CHAPTER] Output directory not found: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
