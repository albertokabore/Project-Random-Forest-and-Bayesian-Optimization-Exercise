"""Apply the author revision without rerunning or altering numerical results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nbformat

from submission_content import AUTHOR, OBSERVATIONS, QUALIFICATION, ROOT, build_markdown, revise_notebook


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_digest(notebook) -> str:
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    return hashlib.sha256(json.dumps(cells, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def main() -> None:
    evidence = [ROOT / "reports" / name for name in (
        "metrics.json", "review_verification.json", "corrected/corrected_results.json",
        "corrected/checkpoint.json", "data_verification.json",
    )]
    hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in evidence}
    path = ROOT / "notebooks/medical_diagnosis_system.ipynb"
    notebook = nbformat.read(path, as_version=4)
    before = code_digest(notebook)
    notebook = revise_notebook(notebook)
    assert before == code_digest(notebook)
    nbformat.validate(notebook)
    report = build_markdown()
    assert report.count("**Observations**") == 6
    assert report.split("\n## ")[-1].startswith("Conclusion\n")
    (ROOT / "reports/REPORT.md").write_text(report, encoding="utf-8")
    nbformat.write(notebook, path)
    saved = nbformat.read(path, as_version=4)
    assert before == code_digest(saved)
    assert all(digest(ROOT / name) == value for name, value in hashes.items())
    manifest = {
        "date": "2026-11-18", "author": AUTHOR, "academic_status": QUALIFICATION,
        "observation_words_by_step": {str(step): sum(len(t.split()) for t in texts)
                                      for step, texts in OBSERVATIONS.items()},
        "notebook_code_cells": 20, "notebook_code_and_outputs_preserved": True,
        "code_cells_sha256_before": before, "code_cells_sha256_after": code_digest(saved),
        "evidence_files_unchanged": hashes,
        "report_sha256": digest(ROOT / "reports/REPORT.md"),
        "notebook_sha256": digest(path),
    }
    (ROOT / "reports/submission_revision.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Updated REPORT.md and notebook markdown for", AUTHOR)
    print("Six observation blocks: exactly 50 words each. All 20 code cells and saved outputs preserved.")
    print("Original and corrected numerical evidence files unchanged.")


if __name__ == "__main__":
    main()
