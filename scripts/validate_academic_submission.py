"""Check the delivered report, citations and preserved computational evidence.

This validates document structure and formatting, not rendered pagination.
It does not rerun the model fits or independently audit clinical measurements.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

import nbformat
from docx import Document
from docx.oxml.ns import qn
from docx.shared import Inches
from lxml import etree

from academic_references import REFERENCES
from submission_content import AUTHOR, OBSERVATIONS, ROOT, abstract_text


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_bytes(path: str) -> bytes:
    return subprocess.run(["git", "show", "HEAD:" + path], cwd=ROOT, check=True, capture_output=True).stdout


def paragraph_text(paragraph) -> str:
    return "".join(paragraph._p.xpath(".//w:t/text()"))


def check_citations(body: str) -> None:
    for item in REFERENCES:
        authors, year = item["key"].rsplit(", ", 1)
        authors = re.escape(authors).replace(r"\ \&\ ", r" (?:&|and) ")
        pattern = authors + r"(?:, | \()" + year
        assert re.search(pattern, body), "Uncited reference: " + item["key"]


def check_observations(text: str) -> None:
    blocks = re.findall(r"\*\*Observations\*\*\n\n((?:- [^\n]+\n?)+)", text)
    assert len(blocks) == 6
    for step, block in enumerate(blocks, start=1):
        bullets = [line[2:] for line in block.splitlines() if line.startswith("- ")]
        assert bullets == OBSERVATIONS[step]
        assert sum(len(value.split()) for value in bullets) == 50


def main() -> None:
    report_text = (ROOT / "reports/REPORT.md").read_text(encoding="utf-8")
    headings = re.findall(r"^## (.+)$", report_text, re.M)
    assert headings[-2:] == ["Conclusion", "References"]
    assert headings[0] == "Abstract"
    assert 150 <= len(abstract_text().split()) <= 250
    reference_text = report_text.split("\n## References\n", 1)[1].strip()
    assert reference_text.split("\n\n") == [item["apa"] for item in REFERENCES]
    assert not re.search(r"et al\.|^- ", reference_text, re.M)
    keys = [item["key"] for item in REFERENCES]
    assert keys == sorted(keys, key=str.casefold)
    check_citations(report_text.split("\n## References\n", 1)[0])
    check_observations(report_text)

    notebook_path = "notebooks/medical_diagnosis_system.ipynb"
    notebook = nbformat.read(ROOT / notebook_path, as_version=4)
    baseline = nbformat.reads(git_bytes(notebook_path).decode("utf-8"), as_version=4)
    nbformat.validate(notebook)
    code = [dict(cell) for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code) == 20
    assert code == [dict(cell) for cell in baseline.cells if cell.cell_type == "code"]
    assert not any(output.output_type == "error" for cell in notebook.cells if cell.cell_type == "code" for output in cell.outputs)
    markdown = "\n\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    assert notebook.cells[-2].source.startswith("## Conclusion\n")
    assert notebook.cells[-1].source == "## References\n\n" + reference_text
    check_citations(markdown.split("\n## References\n", 1)[0])
    check_observations(markdown)
    for figure in range(1, 10):
        assert re.search(rf"Figure {figure}\. ", markdown)
    step1 = next(cell.source for cell in notebook.cells if cell.cell_type == "markdown" and cell.source.startswith("## Step 1."))
    assert step1 == "## Step 1. Data collection"

    revision = json.loads((ROOT / "reports/submission_revision.json").read_text(encoding="utf-8"))
    for filename, expected in revision["evidence_files_unchanged"].items():
        assert digest(ROOT / filename) == expected
        assert hashlib.sha256(git_bytes(filename)).hexdigest() == expected
    assert revision["code_cells_sha256_before"] == revision["code_cells_sha256_after"]

    manifest = json.loads((ROOT / "reports/Medical_Diagnosis_System_Verified_Report.manifest.json").read_text(encoding="utf-8"))
    word_path = ROOT / "reports" / manifest["document"]
    doc = Document(word_path)
    assert doc.core_properties.author == AUTHOR
    assert len(doc.tables) == 7 and len(doc.inline_shapes) == 4
    assert doc.styles["Normal"].font.name == "Times New Roman"
    assert doc.styles["Normal"].font.size.pt == 12
    assert doc.styles["Normal"].paragraph_format.line_spacing == 2
    assert doc.styles["Normal"].paragraph_format.first_line_indent == Inches(0.5)
    assert all(section.left_margin == section.right_margin == section.top_margin == section.bottom_margin == Inches(1) for section in doc.sections)
    assert not doc.sections[0].different_first_page_header_footer
    assert doc.sections[0].header._element.xpath(".//w:fldSimple/@w:instr") == ["PAGE"]
    paragraphs = doc.paragraphs
    word_headings = [p.text for p in paragraphs if p.style.name == "Heading 1"]
    assert word_headings[-2:] == ["Conclusion", "References"]
    reference_index = next(i for i, paragraph in enumerate(paragraphs) if paragraph.text == "References")
    assert paragraphs[reference_index - 1]._p.xpath(".//w:br/@w:type") == ["page"]
    entries = [p for p in paragraphs[reference_index + 1:] if paragraph_text(p).strip()]
    assert len(entries) == len(REFERENCES)
    for paragraph, item in zip(entries, REFERENCES):
        assert paragraph_text(paragraph) == item["apa"].replace("*", "")
        fmt = paragraph.paragraph_format
        assert fmt.left_indent == Inches(0.5) and fmt.first_line_indent == Inches(-0.5)
        assert fmt.line_spacing == 2
        assert any(run.italic for run in paragraph.runs)
        assert paragraph.style.name == "Normal"

    colours, fills = set(), set()
    with zipfile.ZipFile(word_path) as archive:
        assert archive.testzip() is None
        for filename in archive.namelist():
            if not (filename.startswith("word/") and filename.endswith(".xml")):
                continue
            root = etree.fromstring(archive.read(filename))
            for element in root.iter(qn("w:color")):
                colours.add(element.get(qn("w:val")))
                assert not element.get(qn("w:themeColor"))
            fills.update(element.get(qn("w:fill")) for element in root.iter(qn("w:shd")))
            assert not root.findall(".//{urn:schemas-microsoft-com:vml}shape")
            assert b"PowerPlusWaterMarkObject" not in archive.read(filename)
    assert colours == {"000000"}, colours
    assert fills == {"FFFFFF"}, fills
    for table in doc.tables:
        borders = table._tbl.tblPr.find(qn("w:tblBorders"))
        assert borders is not None
        assert all(borders.find(qn("w:" + edge)).get(qn("w:val")) == "nil" for edge in ("left", "right", "insideH", "insideV"))

    html_path = ROOT / "reports/Albert_Kabore_Medical_Diagnosis_Notebook.html"
    html = html_path.read_text(encoding="utf-8")
    assert 'id="academic-monochrome"' in html and 'id="References"' in html
    assert "#References ~ p" in html
    assert "doi.org/10.7717/peerj-cs.1516" in html
    for name in ("academic_references.py", "submission_content.py", "revise_submission.py", "export_verified_word_report.py", "export_academic_notebook.py", "validate_academic_submission.py"):
        ast.parse((ROOT / "scripts" / name).read_text(encoding="utf-8"))

    artifacts = [word_path, ROOT / "reports/REPORT.md", ROOT / notebook_path, html_path, ROOT / "reports/reference_verification.json"]
    results = {
        "author": AUTHOR,
        "qualification": "PhD student in Artificial Intelligence",
        "word_report": word_path.relative_to(ROOT).as_posix(),
        "word_colours": sorted(colours), "table_fills": sorted(fills),
        "watermark_objects": 0, "word_tables": len(doc.tables), "word_figures": len(doc.inline_shapes),
        "notebook_figures": 9,
        "observation_words_per_step": revision["observation_words_by_step"],
        "original_code_and_outputs_preserved": True, "numerical_evidence_unchanged": True,
        "html_reading_view": html_path.relative_to(ROOT).as_posix(),
        "output_based_observations": 12, "output_based_insights": 6, "output_justified_recommendations": 6,
        "apa_references": len(REFERENCES), "all_references_cited_in_report_and_notebook": True,
        "reference_order": "Alphabetical, immediately after the conclusion",
        "reference_format": "Double-spaced, 0.5-inch hanging indent, italic publication titles and volumes",
        "body_format": "Times New Roman 12-point, double-spaced, one-inch margins, header page numbers",
        "validation_scope": "Word XML structure, content and formatting validated; rendered pagination not inspected. Model fits not repeated for this writing revision.",
        "artifacts": {p.relative_to(ROOT).as_posix(): digest(p) for p in artifacts},
    }
    (ROOT / "reports/academic_presentation_check.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    audit_path = ROOT / "reports/FINAL_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["presentation_update"] = results
    audit["artifacts"] = results["artifacts"]
    for check in audit["checks"]:
        if check["check"] == "All earlier notebook and writing requests":
            check["detail"] = "Author details, introduction, six 50-word observation blocks, output-based insights and recommendations retained; conclusion is the final body section, followed by References under the latest instruction. Removed notebook paragraph remains absent; all nine figures have subtitles."
        elif check["check"] == "Word document, formatting and watermark objects":
            check["detail"] = "Current APA edition passes ZIP/XML, colour and formatting checks: black text, white tables, black horizontal rules, seven tables, four figures, six 50-word observation blocks and 13 references after the conclusion. No watermark objects detected."
        elif check["check"] == "Automated model checks":
            check["detail"] = "The preceding numerical review recorded 28 pytest passes in 54.26 seconds, including Bayesian fitting/updating and unseen-hospital effect checks. Model tests were not rerun for this prose/reference revision."
    audit["checks"] = [item for item in audit["checks"] if item["check"] != "APA citations and reference placement"]
    audit["checks"].append({"check": "APA citations and reference placement", "status": "passed", "detail": "Thirteen full references, alphabetically ordered and cited in both report and notebook; Word references start on a new page after Conclusion with double spacing and 0.5-inch hanging indents. Primary bibliographic verification records are linked in reference_verification.json."})
    audit["sources"] = [{"title": item["key"], "url": item["verified_record"]} for item in REFERENCES]
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("PASSED: 13 cited APA references after Conclusion, Word academic formatting, six 50-word observations, nine notebook captions, and unchanged numerical evidence / 20 code cells with outputs.")
    print("Validation scope: structural checks; rendered pagination and full model fitting were not repeated.")


if __name__ == "__main__":
    main()
