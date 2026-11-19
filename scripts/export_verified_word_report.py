"""Create Albert Kabore's editable Word report from the shared Markdown source.

Requires python-docx. REPORT.md identifies original and corrected results.
No model fitting, numerical-output edits or watermark elements are introduced.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from lxml import etree

from submission_content import AUTHOR, QUALIFICATION

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/REPORT.md"
OUTPUT = ROOT / "reports/Medical_Diagnosis_System_Verified_Report.docx"
PERSONALIZED_OUTPUT = ROOT / "reports/Albert_Kabore_Medical_Diagnosis_System_Report_APA.docx"
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


def monochrome_package(payload: bytes) -> bytes:
    """Include auxiliary Word style parts that python-docx does not expose."""
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(payload)) as source, zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as target:
        for member in source.infolist():
            data = source.read(member.filename)
            if member.filename.startswith("word/") and member.filename.endswith(".xml"):
                root = etree.fromstring(data)
                for color in root.iter(qn("w:color")):
                    color.set(qn("w:val"), BLACK)
                    for name in ("themeColor", "themeTint", "themeShade"):
                        color.attrib.pop(qn("w:" + name), None)
                for fill in root.iter(qn("w:shd")):
                    fill.set(qn("w:fill"), WHITE)
                    fill.set(qn("w:color"), BLACK)
                    for name in ("themeFill", "themeFillTint", "themeFillShade", "themeColor", "themeTint", "themeShade"):
                        fill.attrib.pop(qn("w:" + name), None)
                data = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
            target.writestr(member, data)
    return output.getvalue()


class WordReport:
    def __init__(self) -> None:
        self.doc = Document()
        self.table_number = 0
        self.figure_number = 0
        section = self.doc.sections[0]
        section.page_width, section.page_height = Inches(8.5), Inches(11)
        section.top_margin = section.bottom_margin = Inches(1)
        section.left_margin = section.right_margin = Inches(1)
        section.different_first_page_header_footer = False
        section.header_distance = Inches(0.5)
        styles = self.doc.styles
        for name in ("Normal", "Body Text", "List Paragraph", "List Bullet", "Caption", "Subtitle"):
            styles[name].font.name = "Times New Roman"
            styles[name].font.size = Pt(12)
            styles[name].font.color.rgb = RGBColor.from_string(BLACK)
        normal = styles["Normal"].paragraph_format
        normal.line_spacing = 2
        normal.space_before = normal.space_after = Pt(0)
        normal.first_line_indent = Inches(0.5)
        normal.widow_control = True
        for name in ("List Bullet", "List Paragraph"):
            styles[name].paragraph_format.left_indent = Inches(0.5)
            styles[name].paragraph_format.first_line_indent = Inches(-0.25)
        for name, size in (("Title", 12), ("Heading 1", 12), ("Heading 2", 12), ("Heading 3", 12)):
            style = styles[name]
            style.font.name = "Times New Roman"
            style.font.size = Pt(size)
            style.font.color.rgb = RGBColor.from_string(BLACK)
            style.font.bold = True
            style.font.italic = name == "Heading 3"
            style.paragraph_format.first_line_indent = Inches(0)
            style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER if name in ("Title", "Heading 1") else WD_ALIGN_PARAGRAPH.LEFT
            style.paragraph_format.keep_with_next = True
            style.paragraph_format.space_before = style.paragraph_format.space_after = Pt(0)
            style.paragraph_format.line_spacing = 2
        styles["Caption"].font.color.rgb = RGBColor.from_string(BLACK)
        styles["Caption"].font.italic = False
        styles["Caption"].paragraph_format.first_line_indent = Inches(0)
        header = section.header.paragraphs[0]
        header.paragraph_format.first_line_indent = Inches(0)
        header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        field = OxmlElement("w:fldSimple")
        field.set(qn("w:instr"), "PAGE")
        field_run = OxmlElement("w:r")
        field_text = OxmlElement("w:t")
        field_text.text = "1"
        field_run.append(field_text)
        field.append(field_run)
        header._p.append(field)
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
            paragraph.paragraph_format.first_line_indent = Inches(0)
        return paragraph

    def caption(self, label: str, title: str) -> None:
        for text in (f"**{label}**", f"*{title.rstrip('.')}*"):
            paragraph = self.paragraph(text, "Caption")
            paragraph.paragraph_format.keep_with_next = True

    def reference(self, text: str) -> None:
        paragraph = self.paragraph(text)
        paragraph.paragraph_format.left_indent = Inches(0.5)
        paragraph.paragraph_format.first_line_indent = Inches(-0.5)
        paragraph.paragraph_format.line_spacing = 2
        paragraph.paragraph_format.keep_with_next = False
        paragraph.paragraph_format.keep_together = True

    def table(self, rows: list[list[str]], caption: str, widths: list[float] | None = None) -> None:
        self.table_number += 1
        self.caption(f"Table {self.table_number}", caption)
        count = len(rows[0])
        widths = widths or ([2.2] + [(6.5 - 2.2) / (count - 1)] * (count - 1))
        table = self.doc.add_table(rows=0, cols=count)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.style = "Normal Table"
        table.autofit = False
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            border = OxmlElement(f"w:{edge}")
            border.set(qn("w:val"), "single" if edge in ("top", "bottom") else "nil")
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
                paragraph.paragraph_format.first_line_indent = Inches(0)
                rich(paragraph, value)
                for run in paragraph.runs:
                    run.font.size = Pt(10)
                    run.font.color.rgb = RGBColor.from_string(BLACK)
                    if row_index == 0:
                        run.bold = True
                shade(cell, WHITE)
                if row_index == 0:
                    cell_borders = OxmlElement("w:tcBorders")
                    bottom = OxmlElement("w:bottom")
                    bottom.set(qn("w:val"), "single")
                    bottom.set(qn("w:sz"), "4")
                    bottom.set(qn("w:color"), BLACK)
                    cell_borders.append(bottom)
                    cell._tc.get_or_add_tcPr().append(cell_borders)
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
        self.caption(f"Figure {self.figure_number}", caption)
        paragraph = self.doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.first_line_indent = Inches(0)
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
        section_name = ""
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
                title = line.lstrip("# ")
                if level == 1:
                    section_name = title
                    if title in ("Introduction", "References"):
                        self.doc.add_page_break()
                self.doc.add_heading(title, level=min(level, 3))
                index += 1
                continue
            if not section_name:
                # Author details are already on the title page.
                index += 1
                continue
            if line.startswith("```"):
                index += 1
                block = []
                while index < len(lines) and not lines[index].strip().startswith("```"):
                    block.append(lines[index])
                    index += 1
                paragraph = self.doc.add_paragraph()
                paragraph.paragraph_format.first_line_indent = Inches(0)
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
            if section_name == "References":
                self.reference(text)
                continue
            paragraph = self.paragraph(text, style)
            if section_name == "Abstract" and not text.startswith("*Keywords:"):
                paragraph.paragraph_format.first_line_indent = Inches(0)
            if numbered:
                paragraph.paragraph_format.left_indent = Inches(0.25)
                paragraph.paragraph_format.first_line_indent = Inches(-0.25)


def main() -> None:
    report = WordReport()
    doc = report.doc
    title = doc.add_paragraph("Medical Diagnosis System: Random Forest and Bayesian Optimization", "Title")
    title.paragraph_format.space_before = Pt(72)
    title.paragraph_format.space_after = Pt(12)
    for text in (AUTHOR, QUALIFICATION, "November 18, 2026"):
        paragraph = report.paragraph(text)
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.first_line_indent = Inches(0)
    doc.add_page_break()
    source = SOURCE.read_text(encoding="utf-8")
    report.body(source)
    report.enforce_black_text()
    buffer = io.BytesIO()
    doc.save(buffer)
    OUTPUT.write_bytes(monochrome_package(buffer.getvalue()))
    # Update previous filenames when they are not locked by an open Word window.
    payload = OUTPUT.read_bytes()
    actual_output = PERSONALIZED_OUTPUT
    try:
        actual_output.write_bytes(payload)
    except PermissionError:
        actual_output = ROOT / "reports/Albert_Kabore_Medical_Diagnosis_System_Report_APA_Updated.docx"
        actual_output.write_bytes(payload)
        print("The earlier APA file could not be replaced; an updated APA edition was saved.")
    for filename in ("Albert_Kabore_Medical_Diagnosis_System_Report.docx", "Medical_Diagnosis_System_Report.docx"):
        try:
            (ROOT / "reports" / filename).write_bytes(payload)
        except PermissionError:
            print(f"Could not replace {filename}; the academic edition is saved separately.")
    manifest = {
        "edition": "Academic report with APA author-date citations and references after the conclusion",
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
        "body_line_spacing": 2,
        "reference_hanging_indent_inches": 0.5,
        "references_follow_conclusion": True,
    }
    OUTPUT.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(actual_output)
    print(f"{report.table_number} tables, {report.figure_number} figures")


if __name__ == "__main__":
    main()
