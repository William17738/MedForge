"""
MedForge Final Assembler

Assembles processed chapter files into complete subject documents.
Produces three output types per subject:
1. Lecture notes (PPT + textbook integrated)
2. Key knowledge points summary
3. Complete exercise collection
"""

from __future__ import annotations
import re
from pathlib import Path
from config import OUTPUT_DIR, EXERCISES_CHAPTER_SUFFIX, LEGACY_EXERCISES_CHAPTER_SUFFIXES


def _sorted_chapter_files(chapter_dir: Path, suffix: str) -> list[Path]:
    """
    Return chapter files with given suffix, sorted by chapter number.

    Chapter number extraction:
    - Takes leading digits from filename (without extension)
    - Example: "05_cardiovascular_..." -> 5
    - Files without numbers are placed last (chapter=999)
    """
    files = list(chapter_dir.glob(f"*{suffix}.md"))

    def sort_key(path: Path):
        stem = path.stem
        match = re.match(r"(\d+)", stem)
        chapter_num = int(match.group(1)) if match else 999
        return chapter_num, stem

    return sorted(files, key=sort_key)


def _sorted_chapter_files_multi(chapter_dir: Path, suffixes: tuple[str, ...]) -> list[Path]:
    """
    Return chapter files matching any suffix, sorted by chapter number.

    If multiple files map to the same base stem (stem without suffix), earlier suffixes win.
    """
    files_by_base: dict[str, Path] = {}

    for suffix in suffixes:
        for path in chapter_dir.glob(f"*{suffix}.md"):
            stem = path.stem
            base_stem = stem[:-len(suffix)] if stem.endswith(suffix) else stem
            files_by_base.setdefault(base_stem, path)

    def sort_key(item: tuple[str, Path]):
        base_stem, _path = item
        match = re.match(r"(\d+)", base_stem)
        chapter_num = int(match.group(1)) if match else 999
        return chapter_num, base_stem

    return [path for _base, path in sorted(files_by_base.items(), key=sort_key)]


def assemble_subject(subject: str) -> None:
    """
    Assemble all chapter files for a subject into complete documents.

    Creates three output files:
    - {subject}_lecture_notes_complete.md
    - {subject}_key_points_complete.md
    - {subject}_exercises_complete.md
    """
    base_dir = OUTPUT_DIR / subject
    chapter_dir = base_dir / "chapters"

    if not chapter_dir.exists():
        print(f"[ASSEMBLE] Skipping {subject}: chapters directory not found")
        return

    # 1) Lecture Notes: PPT + Textbook integrated
    lecture_files = _sorted_chapter_files(chapter_dir, "_lecture_integrated")
    if lecture_files:
        output_path = base_dir / f"{subject}_lecture_notes_complete.md"
        parts: list[str] = [f"# {subject} - Complete Lecture Notes\n"]

        for file in lecture_files:
            name = file.name
            chapter_id = name.split("_", 1)[0]
            chapter_name = name.split("_lecture_integrated", 1)[0]
            if chapter_name.startswith(chapter_id + "_"):
                chapter_name = chapter_name[len(chapter_id) + 1:]

            parts.append(f"\n\n---\n\n## Chapter {chapter_id}: {chapter_name}\n\n")
            parts.append(file.read_text(encoding="utf-8"))

        output_path.write_text("".join(parts), encoding="utf-8")
        print(f"[ASSEMBLE] Lecture notes merged -> {output_path}")
    else:
        print(f"[ASSEMBLE] {subject}: No lecture files found")

    # 2) Key Points: Exercise-driven knowledge points
    points_files = _sorted_chapter_files(chapter_dir, "_key_points")
    if points_files:
        output_path = base_dir / f"{subject}_key_points_complete.md"
        parts = [f"# {subject} - Key Knowledge Points Summary\n"]

        for file in points_files:
            name = file.name
            chapter_id = name.split("_", 1)[0]
            chapter_name = name.split("_key_points", 1)[0]
            if chapter_name.startswith(chapter_id + "_"):
                chapter_name = chapter_name[len(chapter_id) + 1:]

            parts.append(f"\n\n---\n\n## Chapter {chapter_id}: {chapter_name}\n\n")
            parts.append(file.read_text(encoding="utf-8"))

        output_path.write_text("".join(parts), encoding="utf-8")
        print(f"[ASSEMBLE] Key points merged -> {output_path}")
    else:
        print(f"[ASSEMBLE] {subject}: No key points files found")

    # 3) Exercises: Complete exercise collection with solutions
    exercises_suffixes = (EXERCISES_CHAPTER_SUFFIX,) + tuple(LEGACY_EXERCISES_CHAPTER_SUFFIXES)
    exercise_files = _sorted_chapter_files_multi(chapter_dir, exercises_suffixes)
    if exercise_files:
        output_path = base_dir / f"{subject}_exercises_complete.md"
        parts = [f"# {subject} - Complete Exercise Collection\n"]

        for file in exercise_files:
            stem = file.stem
            base_stem = stem
            for suffix in exercises_suffixes:
                if base_stem.endswith(suffix):
                    base_stem = base_stem.removesuffix(suffix)
                    break

            chapter_id = base_stem.split("_", 1)[0]
            chapter_name = base_stem
            if chapter_name.startswith(chapter_id + "_"):
                chapter_name = chapter_name[len(chapter_id) + 1:]

            parts.append(f"\n\n---\n\n## Chapter {chapter_id}: {chapter_name}\n\n")
            parts.append(file.read_text(encoding="utf-8"))

        output_path.write_text("".join(parts), encoding="utf-8")
        print(f"[ASSEMBLE] Exercises merged -> {output_path}")
    else:
        print(f"[ASSEMBLE] {subject}: No exercise files found")
