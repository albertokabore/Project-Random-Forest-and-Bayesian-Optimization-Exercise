"""Create Albert Kabore's editable Word report from the shared Markdown source.

Requires python-docx. REPORT.md identifies original and corrected results.
No model fitting, numerical-output edits or watermark elements are introduced.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from submission_content import AUTHOR, QUALIFICATION

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/REPORT.md"
OUTPUT = ROOT / "reports/Medical_Diagnosis_System_Verified_Report.docx"
PERSONALIZED_OUTPUT = ROOT / "reports/Albert_Kabore_Medical_Diagnosis_System_Report_Academic.docx"
BLACK = "000000"
WHITE = "FFFFFF"
INLINE = re.compile(r"(\[[^\]]+\]\([^)]+\)|\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*|https?://[^\s]+)")


def hyperlink(paragraph, label: str, url: str) -> None:
    relationship = paragraph.part.relate_to(
        url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True
    )
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), relationship)
    run = OxmlElement("w:r")
    properties = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), BLACK)
    properties.append(color)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    properties.append(underline)
    run.append(properties)
    text = OxmlElement("w:t")
    text.text = label
    run.append(text)
    link.append(run)
    paragraph._p.append(link)


def rich(paragraph, text: str) -> None:
    for token in INLINE.split(text):
        if not token:
            continue
        match = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", token)
        if match:
            hyperlink(paragraph, *match.groups())
        elif token.startswith("**") and token.endswith("**"):
            paragraph.add_run(token[2:-2]).bold = True
        elif token.startswith("`") and token.endswith("`"):
            run = paragraph.add_run(token[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(9)
        elif token.startswith("*") and token.endswith("*"):
            paragraph.add_run(token[1:-1]).italic = True
        elif token.startswith(("https://", "http://")):
            url = token.rstrip(".,;")
            hyperlink(paragraph, url, url)
            paragraph.add_run(token[len(url):])
        else:
            paragraph.add_run(token)


def shade(cell, fill: str) -> None:
    element = OxmlElement("w:shd")
    element.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(element)


class WordReport:
    def __init__(self) -> None:
        self.doc = Document()
        self.table_number = 0
        self.figure_number = 0
        section = self.doc.sections[0]
        section.page_width, section.page_height = Inches(8.5), Inches(11)
        section.top_margin = section.bottom_margin = Inches(1)
        section.left_margin = section.right_margin = Inches(1)
        section.different_first_page_header_footer = True
        styles = self.doc.styles
        for name in ("Normal", "Body Text", "List Paragraph", "List Bullet", "Caption", "Subtitle"):
            styles[name].font.name = "Times New Roman"
            styles[name].font.size = Pt(12 if name != "Caption" else 10)
            styles[name].font.color.rgb = RGBColor.from_string(BLACK)
        normal = styles["Normal"].paragraph_format
        normal.line_spacing = 1.12
        normal.space_after = Pt(6)
        normal.widow_control = True
        for name, size in (("Title", 20), ("Heading 1", 14), ("Heading 2", 12), ("Heading 3", 12)):
            style = styles[name]
            style.font.name = "Times New Roman"
            style.font.size = Pt(size)
            style.font.color.rgb = RGBColor.from_string(BLACK)
            style.paragraph_format.keep_with_next = True
            style.paragraph_format.space_before = Pt(14)
            style.paragraph_format.space_after = Pt(6)
        styles["Caption"].font.color.rgb = RGBColor.from_string(BLACK)
        header = section.header.paragraphs[0]
        header.text = "Medical Diagnosis System | Random Forest and Bayesian Optimization"
        header.runs[0].font.size = Pt(8)
        footer = section.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        footer.add_run("Page ")
        field = OxmlElement("w:fldSimple")
        field.set(qn("w:instr"), "PAGE")
        footer._p.append(field)
        props = self.doc.core_properties
        props.title = "Medical Diagnosis System: Random Forest and Bayesian Optimization"
        props.subject = "Process, findings, predictive accuracy and methodological limitations"
        props.author = AUTHOR
        props.last_modified_by = AUTHOR
        props.keywords = "Random Forest, Bayesian Optimization, PyMC, CAD, UCI Heart Disease"
        props.comments = ""

    def paragraph(self, text: str, style: str | None = None):
        paragraph = self.doc.add_paragraph(style=style)
        rich(paragraph, text)
        if text == "**Observations**":
            paragraph.paragraph_format.keep_with_next = True
        return paragraph

    def caption(self, text: str) -> None:
        paragraph = self.paragraph(text, "Caption")
        paragraph.paragraph_format.keep_with_next = True

    def table(self, rows: list[list[str]], caption: str, widths: list[float] | None = None) -> None:
        self.table_number += 1
        self.caption(f"Table {self.table_number}. {caption}")
        count = len(rows[0])
        widths = widths or ([2.2] + [(6.5 - 2.2) / (count - 1)] * (count - 1))
        table = self.doc.add_table(rows=0, cols=count)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.style = "Table Grid"
        table.autofit = False
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            border = OxmlElement(f"w:{edge}")
            border.set(qn("w:val"), "single")
            border.set(qn("w:sz"), "4")
            border.set(qn("w:color"), BLACK)
            borders.append(border)
        table._tbl.tblPr.append(borders)
        for column, width in zip(table.columns, widths):
            column.width = Inches(width)
        for row_index, source in enumerate(rows):
            row = table.add_row()
            prevent_split = OxmlElement("w:cantSplit")
            row._tr.get_or_add_trPr().append(prevent_split)
            if row_index == 0:
                repeat = OxmlElement("w:tblHeader")
                row._tr.get_or_add_trPr().append(repeat)
            for cell, value, width in zip(row.cells, source, widths):
                cell.width = Inches(width)
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                paragraph = cell.paragraphs[0]
                paragraph.paragraph_format.space_after = Pt(4)
                paragraph.paragraph_format.space_before = Pt(3)
                paragraph.paragraph_format.line_spacing = 1.03
                rich(paragraph, value)
                for run in paragraph.runs:
                    run.font.size = Pt(10)
                    run.font.color.rgb = RGBColor.from_string(BLACK)
                    if row_index == 0:
                        run.bold = True
                shade(cell, WHITE)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)

    def markdown_table(self, rows: list[list[str]]) -> None:
        header = rows[0]
        if header[:2] == ["Model", "ROC-AUC (95% CI)"]:
            for columns, caption, widths in (
                ([0, 1, 2, 3, 4], "Held-out discrimination and classification (decision threshold 0.5).", [2.25, 1.35, 0.9, 1, 1]),
                ([0, 5, 6, 7, 8], "Predictive values and probability scores on the same test records.", [2.5, 1, 1, 1, 1]),
            ):
                self.table([[row[c] for c in columns] for row in rows], caption, widths)
            self.paragraph("PPV = positive predictive value; NPV = negative predictive value. Lower Brier score and log loss are better.")
            return
        captions = {
            "Question": "Summary of findings and their interpretation.",
            "Search": "Best cross-validation log loss in the original optimization comparison.",
            "Patient": "Selected case illustrations from the original saved analysis.",
            "Held-out hospital": "Completed corrected hospital evaluation with nested Random Forest tuning.",
            "Corrected main fit": "Repeated main fits using stricter sampling settings; separate from the original test table.",
            "Hospital dataset": "Record counts and observed CAD prevalence by source dataset.",
            "Step": "Model refinement and training-data PSIS-LOO comparison.",
            "Patients seen": "Retrospective M1 sequential updating experiment.",
            "Assignment requirement": "Alignment with the supplied assignment instructions.",
        }
        caption = captions.get(header[0], "Disease prevalence and record counts by source hospital.")
        if header[:2] == ["Hospital", "Prior mean"]:
            caption = "Separate Beta prevalence updates by hospital."
        widths = None
        if header[0] == "Question":
            widths = [1.8, 4.7]
        elif header[0] == "Step":
            widths = [0.45, 2, 1.8, 2.25]
        elif header[0] == "Patients seen":
            widths = [2] + [0.9] * 5
        self.table(rows, caption, widths)

    def figure(self, filename: str, caption: str) -> None:
        self.figure_number += 1
        self.caption(f"Figure {self.figure_number}. {caption}")
        paragraph = self.doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run()
        run.add_picture(str(ROOT / "reports/figures" / filename), width=Inches(6.45))
        image = run._r.xpath(".//wp:docPr")[0]
        image.set("descr", caption)

    def enforce_black_text(self) -> None:
        """Override direct and theme-derived text colours throughout the file."""
        for style in self.doc.styles:
            if hasattr(style, "font"):
                style.font.color.rgb = RGBColor.from_string(BLACK)
        roots = [self.doc._element, self.doc.styles.element]
        for section in self.doc.sections:
            for container in (section.header, section.footer, section.first_page_header, section.first_page_footer):
                roots.append(container._element)
        for root in roots:
            for color in root.iter(qn("w:color")):
                color.set(qn("w:val"), BLACK)
                for name in ("themeColor", "themeTint", "themeShade"):
                    color.attrib.pop(qn("w:" + name), None)
            for fill in root.iter(qn("w:shd")):
                fill.set(qn("w:fill"), WHITE)
                fill.set(qn("w:color"), BLACK)
                for name in ("themeFill", "themeFillTint", "themeFillShade", "themeColor", "themeTint", "themeShade"):
                    fill.attrib.pop(qn("w:" + name), None)
        for table in self.doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    shade(cell, WHITE)

    def body(self, source: str) -> None:
        lines = source.splitlines()
        index = 0
        while index < len(lines):
            line = lines[index].strip()
            if not line or line == "---":
                index += 1
                continue
            if line.startswith("# "):
                index += 1
                continue
            if line.startswith("##"):
                level = len(line) - len(line.lstrip("#")) - 1
                self.doc.add_heading(line.lstrip("# "), level=min(level, 3))
                index += 1
                continue
            if line.startswith("```"):
                index += 1
                block = []
                while index < len(lines) and not lines[index].strip().startswith("```"):
                    block.append(lines[index])
                    index += 1
                paragraph = self.doc.add_paragraph()
                run = paragraph.add_run("\n".join(block))
                run.font.name, run.font.size = "Consolas", Pt(9)
                index += 1
                continue
            if line.startswith("|"):
                rows = []
                while index < len(lines) and lines[index].strip().startswith("|"):
                    cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                    if not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                        rows.append(cells)
                    index += 1
                self.markdown_table(rows)
                continue
            picture = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line)
            if picture:
                caption, filename = picture.groups()
                self.figure(Path(filename).name, caption)
                index += 1
                continue
            bullet = re.match(r"^[-*] (.+)", line)
            numbered = re.match(r"^(\d+)\.\s+(.+)", line)
            text = bullet.group(1) if bullet else line
            index += 1
            while index < len(lines):
                nxt = lines[index].strip()
                if not nxt or nxt.startswith(("#", "|", "```", "![")) or re.match(r"^(?:[-*] |\d+\.\s)", nxt):
                    break
                text += " " + nxt
                index += 1
            style = "List Bullet" if bullet else "List Paragraph" if numbered else None
            paragraph = self.paragraph(text, style)
            if numbered:
                paragraph.paragraph_format.left_indent = Inches(0.25)
                paragraph.paragraph_format.first_line_indent = Inches(-0.25)


def main() -> None:
    report = WordReport()
    doc = report.doc
    doc.add_paragraph("Medical Diagnosis System", "Title")
    doc.add_paragraph("Random Forest and Bayesian Optimization Exercise", "Subtitle")
    report.paragraph("Process, findings and predictive accuracy")
    report.paragraph(f"**{AUTHOR}**")
    report.paragraph(QUALIFICATION)
    report.paragraph("November 18, 2026")
    doc.add_paragraph()
    report.paragraph("Research report on data collection, prior specification, model development, validation, sequential updating, and predictive accuracy.")
    doc.add_page_break()
    source = SOURCE.read_text(encoding="utf-8")
    report.body(source)
    report.enforce_black_text()
    doc.save(OUTPUT)
    # Update previous filenames when they are not locked by an open Word window.
    payload = OUTPUT.read_bytes()
    actual_output = PERSONALIZED_OUTPUT
    try:
        actual_output.write_bytes(payload)
    except PermissionError:
        actual_output = ROOT / "reports/Albert_Kabore_Medical_Diagnosis_System_Report_Results_Analysis.docx"
        actual_output.write_bytes(payload)
        print("The earlier academic file could not be replaced; a results-analysis edition was saved.")
    for filename in ("Albert_Kabore_Medical_Diagnosis_System_Report.docx", "Medical_Diagnosis_System_Report.docx"):
        try:
            (ROOT / "reports" / filename).write_bytes(payload)
        except PermissionError:
            print(f"Could not replace {filename}; the academic edition is saved separately.")
    manifest = {
        "edition": "Author revision with original and completed corrected analyses",
        "author": AUTHOR,
        "academic_status": QUALIFICATION,
        "date": "2026-11-18",
        "document": actual_output.name,
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "verification_sha256": hashlib.sha256((ROOT / "reports/review_verification.json").read_bytes()).hexdigest(),
        "corrected_results_sha256": hashlib.sha256((ROOT / "reports/corrected/corrected_results.json").read_bytes()).hexdigest(),
        "tables": report.table_number,
        "figures": report.figure_number,
        "text_colour": BLACK,
        "table_fill": WHITE,
    }
    OUTPUT.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(actual_output)
    print(f"{report.table_number} tables, {report.figure_number} figures")


if __name__ == "__main__":
    main()
