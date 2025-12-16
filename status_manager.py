"""
MedForge Status Manager

Atomic per-subject state management to avoid race conditions in parallel processing.
Each subject maintains independent status files in its own directory.
"""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from config import OUTPUT_DIR


class SubjectStatusManager:
    """
    Manages independent status files for each subject.

    Status files are stored in output/<subject>/.status/ directory.
    This design replaces global JSON files to avoid race conditions
    and support fine-grained state tracking in parallel processing.
    """

    def __init__(self, subject_name: str):
        self.subject_dir = OUTPUT_DIR / subject_name
        self.status_dir = self.subject_dir / ".status"
        self.status_dir.mkdir(parents=True, exist_ok=True)

    def _read_status(self, file_path: Path) -> dict:
        """Read status from a JSON file."""
        try:
            return json.loads(file_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write_status(self, file_path: Path, data: dict) -> None:
        """Write status to a JSON file."""
        try:
            file_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"[WARN] Failed to write status to {file_path}: {e}")

    # Preprocessing status (preprocess.json)
    def get_preprocess_status(self) -> dict:
        """Get preprocessing status for this subject."""
        return self._read_status(self.status_dir / "preprocess.json")

    def set_preprocess_status(self, source_hash: str, **metadata) -> None:
        """Update preprocessing status with source hash and metadata."""
        data = {
            "source_hash": source_hash,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        data.update(metadata)
        self._write_status(self.status_dir / "preprocess.json", data)

    # Pipeline processing status (pipeline.json)
    def get_pipeline_status(self) -> dict:
        """Get pipeline processing status."""
        return self._read_status(self.status_dir / "pipeline.json")

    def update_pipeline_status(self, **kwargs) -> None:
        """Update pipeline processing status."""
        path = self.status_dir / "pipeline.json"
        data = self._read_status(path)
        data.update(kwargs)
        data["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self._write_status(path, data)
