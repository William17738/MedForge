"""
MedForge Preprocessor

Splits source documents into chapter-based files for downstream processing.
Supports both configuration-based and auto-detection modes.
"""

import re
import json
import hashlib
from pathlib import Path
from typing import Optional

from config import OUTPUT_DIR, SUBJECT_CONFIG_FILE
from status_manager import SubjectStatusManager


# Subject mapping for auto-detection (customizable)
SUBJECT_MAP = {
    # Add your subject ID mappings here
    # Example: "01": "InternalMedicine",
}

# Keyword-based subject detection (fallback)
SUBJECT_KEYWORDS = {
    # Add keyword -> subject name mappings
    # Example: "cardio": "Cardiology",
}


def _match_existing_subject_dir(subject_name: str) -> str:
    """
    Match against existing subject directories (case-insensitive).
    Returns existing directory name if found, otherwise original name.
    """
    if not OUTPUT_DIR.exists():
        return subject_name

    lower_map = {d.name.lower(): d.name for d in OUTPUT_DIR.iterdir() if d.is_dir()}
    return lower_map.get(subject_name.lower(), subject_name)


def _compute_hash(files: list[Path]) -> str:
    """Compute SHA256 hash of file contents."""
    h = hashlib.sha256()
    for path in sorted(files, key=lambda x: x.name):
        if path.exists():
            h.update(path.read_bytes())
    return h.hexdigest()


def _should_skip(source_files: list[Path], subject: str) -> bool:
    """Check if preprocessing can be skipped (source unchanged)."""
    manager = SubjectStatusManager(subject)
    status = manager.get_preprocess_status()
    prev_hash = status.get("source_hash")

    if not prev_hash:
        return False
    return _compute_hash(source_files) == prev_hash


def process_file(file_path: Path, subject_override: Optional[str] = None) -> None:
    """
    Process a source file and split into chapter files.

    Args:
        file_path: Path to the source file
        subject_override: Force specific subject name (optional)
    """
    filename = file_path.name
    subject_name = subject_override

    # Auto-detect subject from filename
    if not subject_name:
        # Try numeric prefix mapping
        match = re.search(r'^(\d+)\.', filename)
        if match:
            subject_id = match.group(1)
            subject_name = SUBJECT_MAP.get(subject_id)

        # Try keyword matching
        if not subject_name:
            for keyword, subject in SUBJECT_KEYWORDS.items():
                if keyword.lower() in filename.lower():
                    subject_name = subject
                    break

        # Fallback to generic name
        if not subject_name and match:
            subject_name = f"Subject_{match.group(1)}"

    if not subject_name:
        print(f"[SKIP] Cannot determine subject for: {filename}")
        return

    # Match existing directory
    subject_name = _match_existing_subject_dir(subject_name)
    print(f"[PROCESS] {filename} -> {subject_name}")

    # Setup directories
    base_dir = OUTPUT_DIR / subject_name
    raw_dir = base_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Read content
    content = file_path.read_text(encoding='utf-8', errors='ignore')

    # Find chapters using regex pattern
    # Matches: "Chapter X: Title" or similar patterns
    chapter_pattern = re.compile(
        r'(?:^|\n)\s*(?:Chapter|Ch\.?|Section)\s*(\d+)[:\s]+([^\n]+)',
        re.IGNORECASE
    )

    matches = list(chapter_pattern.finditer(content))

    if not matches:
        print(f"  [WARN] No chapters found in {filename}")
        # Save entire file
        output_file = raw_dir / f"00_full_content.txt"
        output_file.write_text(content, encoding='utf-8')
        return

    # Process each chapter
    for i, match in enumerate(matches):
        start_idx = match.start()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(content)

        chapter_num = match.group(1)
        chapter_name = match.group(2).strip()
        chapter_content = content[start_idx:end_idx]

        # Sanitize chapter name for filename
        chapter_name = re.sub(r'[\\/:*?"<>|]', '_', chapter_name).strip()
        chapter_id = f"{int(chapter_num):02d}"

        print(f"  [CHAPTER] {chapter_id}: {chapter_name}")

        # Determine file type based on content
        has_exercises = bool(re.search(r'(?:exercise|question|problem)', chapter_content, re.I))

        if has_exercises:
            # Try to split content and exercises
            exercise_match = re.search(
                r'(?:^|\n)\s*(?:Exercises?|Questions?|Problems?)\s*(?:\n|:)',
                chapter_content, re.IGNORECASE
            )

            if exercise_match:
                content_part = chapter_content[:exercise_match.start()]
                exercise_part = chapter_content[exercise_match.start():]

                # Save content
                content_file = raw_dir / f"{chapter_id}_{chapter_name}_content.txt"
                content_file.write_text(content_part, encoding='utf-8')

                # Save exercises
                exercise_file = raw_dir / f"{chapter_id}_{chapter_name}_exercises.txt"
                exercise_file.write_text(exercise_part, encoding='utf-8')
            else:
                # Save combined
                output_file = raw_dir / f"{chapter_id}_{chapter_name}_combined.txt"
                output_file.write_text(chapter_content, encoding='utf-8')
        else:
            # Save as textbook content
            output_file = raw_dir / f"{chapter_id}_{chapter_name}_textbook.txt"
            output_file.write_text(chapter_content, encoding='utf-8')


def main():
    """Main preprocessing function."""
    print("[PREPROCESS] Starting preprocessing...")

    # Configuration-based processing
    if SUBJECT_CONFIG_FILE.exists():
        print(f"[PREPROCESS] Using config: {SUBJECT_CONFIG_FILE}")
        try:
            config = json.loads(SUBJECT_CONFIG_FILE.read_text(encoding="utf-8"))

            for subject, mapping in config.items():
                subject_dir = _match_existing_subject_dir(subject)

                # Collect source files
                source_files = []
                for key in ["textbook", "exercises", "source"]:
                    filename = mapping.get(key)
                    if filename:
                        filepath = OUTPUT_DIR / filename
                        if filepath.exists():
                            source_files.append(filepath)

                if not source_files:
                    print(f"[SKIP] No source files for: {subject}")
                    continue

                # Check if processing needed
                if _should_skip(source_files, subject_dir):
                    print(f"[SKIP] {subject_dir} - sources unchanged")
                    continue

                # Process files
                for filepath in source_files:
                    try:
                        process_file(filepath, subject_override=subject_dir)
                    except Exception as e:
                        print(f"[ERROR] Processing {filepath.name}: {e}")

                # Update status
                manager = SubjectStatusManager(subject_dir)
                manager.set_preprocess_status(
                    source_hash=_compute_hash(source_files),
                    source_files=[f.name for f in source_files]
                )

            return

        except Exception as e:
            print(f"[WARN] Config processing failed: {e}")
            print("[PREPROCESS] Falling back to directory scan...")

    # Legacy scan mode
    if not OUTPUT_DIR.exists():
        print(f"[ERROR] Output directory not found: {OUTPUT_DIR}")
        return

    files = sorted(OUTPUT_DIR.glob("*.txt"))
    for filepath in files:
        try:
            process_file(filepath)
        except Exception as e:
            print(f"[ERROR] Processing {filepath.name}: {e}")

    print("[PREPROCESS] Preprocessing complete")


if __name__ == "__main__":
    main()
