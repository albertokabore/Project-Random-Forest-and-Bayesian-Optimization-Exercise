"""Export a black-on-white notebook reading view without executing its cells."""

from __future__ import annotations

from pathlib import Path
import sys

import nbformat
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks/medical_diagnosis_system.ipynb"
OUTPUT = ROOT / "reports/Albert_Kabore_Medical_Diagnosis_Notebook.html"

CSS = """
<style id="academic-monochrome">
:root { color-scheme: light !important; }
html, body, #notebook, #notebook-container, .cell, .text_cell_render,
.output_wrapper, .output, .output_area, .output_subarea {
  background: #ffffff !important;
  color: #000000 !important;
}
body, p, li, h1, h2, h3, h4, h5, h6, table, caption {
  font-family: "Times New Roman", Times, serif !important;
}
body * { color: #000000 !important; }
table, thead, tbody, tfoot, tr, th, td, table.dataframe th, table.dataframe td {
  background: #ffffff !important;
  color: #000000 !important;
  border-color: #000000 !important;
  box-shadow: none !important;
}
table { border-collapse: collapse !important; }
th, td { border: 1px solid #000000 !important; padding: 5px 7px !important; }
a, a:visited, a:hover, a:active { color: #000000 !important; }
pre, code { font-family: Consolas, monospace !important; }
#References ~ p { padding-left: 0.5in; text-indent: -0.5in; line-height: 2; }
@media print {
  body, #notebook-container { background: #ffffff !important; box-shadow: none !important; }
  h1, h2, h3, h4, .output_png { break-inside: avoid; }
}
</style>
"""


def main() -> None:
    notebook = nbformat.read(SOURCE, as_version=4)
    # A venv using system packages may not include system Jupyter data paths.
    system_templates = Path(sys.base_prefix) / "share/jupyter/nbconvert/templates"
    extra_paths = [str(system_templates)] if system_templates.is_dir() else []
    exporter = HTMLExporter(
        template_name="classic", extra_template_basedirs=extra_paths, extra_template_paths=extra_paths
    )
    exporter.exclude_input = True
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    html, _ = exporter.from_notebook_node(
        notebook, resources={"metadata": {"name": "Medical Diagnosis System - Albert Kabore"}}
    )
    assert "</head>" in html
    html = html.replace("</head>", CSS + "\n</head>", 1)
    OUTPUT.write_text(html, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
