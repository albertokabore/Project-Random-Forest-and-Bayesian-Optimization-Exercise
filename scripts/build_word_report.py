"""Build reports/Medical_Diagnosis_System_Report.docx from the saved results.

Every number in the tables is read from the analysis outputs, never retyped:
  reports/metrics.json                       original executed notebook (main split, RF search, sequential updating)
  reports/review_verification.json           independent refit of the main models (review, Nov 18 2026)
  reports/corrected/corrected_results.json   corrected hospital holdouts + prior ablation + paired bootstrap

Usage:  python scripts/build_word_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
CORR = REPORTS / "corrected"
OUT = REPORTS / "Medical_Diagnosis_System_Report.docx"

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
GREY = RGBColor(0x52, 0x51, 0x4E)
HEADER_FILL = "1F3A5F"
BAND_FILL = "EEF2F7"
FONT = "Calibri"

M = json.loads((REPORTS / "metrics.json").read_text())
V = json.loads((REPORTS / "review_verification.json").read_text())
C = json.loads((CORR / "corrected_results.json").read_text())
import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_narrative import TEXT  # noqa: E402  (prose kept separate from layout code)


# --------------------------------------------------------------------------- helpers
def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def f3(x: float) -> str:
    return f"{x:.3f}"


def shade(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_margins(table, top=60, bottom=60, left=90, right=90) -> None:
    tbl_pr = table._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for side, val in (("top", top), ("bottom", bottom), ("left", left), ("right", right)):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:w"), str(val))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tbl_pr.append(mar)


def repeat_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    tr_pr.append(el)


def add_field(paragraph, instr: str) -> None:
    """Insert a Word field (PAGE, TOC ...). Word computes it on open / update."""
    run = paragraph.add_run()
    for tag, text in (("begin", None), (None, instr), ("separate", None), ("end", None)):
        if tag:
            fc = OxmlElement("w:fldChar")
            fc.set(qn("w:fldCharType"), tag)
            run._r.append(fc)
        else:
            it = OxmlElement("w:instrText")
            it.set(qn("xml:space"), "preserve")
            it.text = text
            run._r.append(it)


class Report:
    def __init__(self) -> None:
        self.doc = Document()
        self.fig_no = 0
        self.tab_no = 0
        self._styles()
        sec = self.doc.sections[0]
        sec.page_width, sec.page_height = Inches(8.5), Inches(11)
        for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(sec, side, Inches(1))

    def _styles(self) -> None:
        st = self.doc.styles
        normal = st["Normal"]
        normal.font.name, normal.font.size = FONT, Pt(11)
        normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        normal.paragraph_format.space_after = Pt(6)
        normal.paragraph_format.line_spacing = 1.15
        for name, size, before, after in (("Title", 26, 0, 6), ("Heading 1", 16, 18, 6),
                                          ("Heading 2", 13, 12, 4), ("Heading 3", 11.5, 10, 3)):
            s = st[name]
            s.font.name, s.font.size, s.font.bold = FONT, Pt(size), True
            s.font.color.rgb = NAVY
            rfonts = s.element.rPr.find(qn("w:rFonts"))
            if rfonts is None:
                rfonts = OxmlElement("w:rFonts")
                s.element.rPr.append(rfonts)
            for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
                rfonts.set(qn(attr), FONT)
            for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
                rfonts.attrib.pop(qn(attr), None)
            pf = s.paragraph_format
            pf.space_before, pf.space_after, pf.keep_with_next = Pt(before), Pt(after), True
        from docx.enum.style import WD_STYLE_TYPE
        for lvl, indent in (("toc 1", 0.0), ("toc 2", 0.25)):  # Word's built-in TOC style names
            ts = st.add_style(lvl, WD_STYLE_TYPE.PARAGRAPH)
            ts.base_style = st["Normal"]
            ts.paragraph_format.space_after = Pt(1)
            ts.paragraph_format.space_before = Pt(0)
            ts.paragraph_format.line_spacing = 1.0
            ts.paragraph_format.left_indent = Inches(indent)
            ts.font.size = Pt(10.5)
        cap = st["Caption"]
        cap.font.name, cap.font.size, cap.font.italic, cap.font.bold = FONT, Pt(9.5), True, False
        cap.font.color.rgb = GREY
        cap.paragraph_format.space_after = Pt(10)

    # ------------------------------------------------------------------ blocks
    def h(self, text: str, level: int = 1) -> None:
        self.doc.add_heading(text, level=level)

    def p(self, text: str = "", bold_lead: str | None = None, italic: bool = False, align=None, size=None) -> None:
        para = self.doc.add_paragraph()
        if bold_lead:
            para.add_run(bold_lead).bold = True
        self._rich(para, text, italic=italic, size=size)
        if align is not None:
            para.alignment = align

    @staticmethod
    def _rich(para, text: str, italic: bool = False, size=None) -> None:
        """Minimal **bold** markup inside a paragraph."""
        for i, part in enumerate(text.split("**")):
            for j, piece in enumerate(part.split("*")):
                if not piece:
                    continue
                run = para.add_run(piece)
                run.bold = i % 2 == 1
                run.italic = italic or j % 2 == 1
                if size:
                    run.font.size = Pt(size)

    def bullets(self, items: list[str], numbered: bool = False) -> None:
        for n, item in enumerate(items, start=1):
            if numbered:  # explicit numbers so every list restarts at 1
                para = self.doc.add_paragraph()
                para.paragraph_format.left_indent = Inches(0.3)
                para.paragraph_format.first_line_indent = Inches(-0.3)
                para.add_run(f"{n}.	")
                para.paragraph_format.tab_stops.add_tab_stop(Inches(0.3))
            else:
                para = self.doc.add_paragraph(style="List Bullet")
            self._rich(para, item)
            para.paragraph_format.space_after = Pt(3)

    def equation(self, text: str) -> None:
        para = self.doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = para.add_run(text)
        run.font.name, run.font.size = "Cambria Math", Pt(11)
        run.italic = True

    def table(self, header: list[str], rows: list[list[str]], widths: list[float], caption: str,
              bold_rows: tuple[int, ...] = (), align_right_from: int = 1, font_size: float = 9.5,
              right_cols: set[int] | None = None) -> None:
        right = right_cols if right_cols is not None else set(range(align_right_from, len(header)))
        self.tab_no += 1
        cap = self.doc.add_paragraph(f"Table {self.tab_no}. {caption}", style="Caption")
        cap.paragraph_format.keep_with_next = True
        cap.paragraph_format.space_after = Pt(3)
        t = self.doc.add_table(rows=1, cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        set_cell_margins(t)
        for i, text in enumerate(header):
            cell = t.rows[0].cells[i]
            cell.text = ""
            run = cell.paragraphs[0].add_run(text)
            run.bold, run.font.size = True, Pt(font_size)
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            shade(cell, HEADER_FILL)
            if i in right:
                cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
            cell.paragraphs[0].paragraph_format.keep_with_next = True  # never strand the header row
        repeat_header(t.rows[0])
        for r_i, row in enumerate(rows):
            cells = t.add_row().cells
            for i, text in enumerate(row):
                cells[i].text = ""
                para = cells[i].paragraphs[0]
                self._rich(para, str(text), size=font_size)
                if r_i in bold_rows:
                    for run in para.runs:
                        run.bold = True
                para.paragraph_format.space_after = Pt(0)
                if len(rows) <= 9 and r_i < len(rows) - 1:
                    para.paragraph_format.keep_with_next = True  # keep short tables together
                if i in right:
                    para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                if r_i % 2 == 1:
                    shade(cells[i], BAND_FILL)
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(4)

    def figure(self, name: str, caption: str, width: float = 6.3) -> None:
        path = FIG / f"{name}.png"
        if not path.exists():
            path = CORR.parent / "figures" / f"{name}.png"
        self.fig_no += 1
        para = self.doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.keep_with_next = True
        para.add_run().add_picture(str(path), width=Inches(width))
        self.doc.add_paragraph(f"Figure {self.fig_no}. {caption}", style="Caption").alignment = WD_ALIGN_PARAGRAPH.CENTER

    def page_break(self) -> None:
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def footer_page_numbers(self) -> None:
        for sec in self.doc.sections:
            para = sec.footer.paragraphs[0]
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = para.add_run("Medical Diagnosis System — Random Forest and Bayesian Optimization Exercise   |   Page ")
            run.font.size, run.font.color.rgb = Pt(8.5), GREY
            add_field(para, "PAGE")
            for r in para.runs[1:]:
                r.font.size = Pt(8.5)


# --------------------------------------------------------------------------- data shortcuts
TEST = {r["model"]: r for r in M["test_metrics"]}
VTEST = {r["model"]: r for r in V["test_metrics"]} if isinstance(V["test_metrics"][0], dict) and "model" in V["test_metrics"][0] else {}
MAIN = C["main"]
LOHO = C["loho"]
LSUM = C["loho_summary"]
SITES = ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"]
LOHO_MODELS = ["Literature only", "M1 Bayesian", "M2 Bayesian", "RF (nested BO)"]


def build() -> None:
    r = Report()
    d = r.doc

    # ------------------------------------------------------------------ title page
    for _ in range(5):
        d.add_paragraph()
    t = d.add_paragraph(style="Title")
    t.add_run("Medical Diagnosis System")
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = d.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = sub.add_run("Predicting Coronary Artery Disease with Random Forest, Bayesian Optimization\n"
                      "and Bayesian Updating of Prior Medical Knowledge")
    run.font.size, run.font.color.rgb = Pt(14), NAVY
    d.add_paragraph()
    sub2 = d.add_paragraph()
    sub2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = sub2.add_run("Random Forest and Bayesian Optimization Exercise — Final Report")
    run.font.size, run.italic = Pt(12), True
    for _ in range(6):
        d.add_paragraph()
    for label in ("Student:", "Course:", "Instructor:", "Date:"):
        para = d.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.add_run(f"{label} ").bold = True
        para.add_run("November 2026" if label == "Date:" else "[enter here]")
    r.page_break()

    # ------------------------------------------------------------------ contents
    toc_title = d.add_paragraph()  # not a Heading, so it does not list itself
    run = toc_title.add_run("Table of Contents")
    run.bold, run.font.size, run.font.color.rgb = True, Pt(16), NAVY
    add_field(d.add_paragraph(), 'TOC \\o "1-2" \\h \\z \\u')
    r.page_break()

    # ------------------------------------------------------------------ executive summary
    r.h("Executive Summary", 1)
    for para in TEXT["executive_summary"]:
        r.p(para)
    r.table(
        ["Question", "Answer (evidence)"],
        TEXT["summary_rows"],
        [2.0, 4.5], "Key findings at a glance.", align_right_from=99,
    )

    # ------------------------------------------------------------------ 1 introduction
    r.h("1. Introduction and Objectives", 1)
    for para in TEXT["introduction"]:
        r.p(para)
    r.h("1.1 Research questions", 2)
    r.bullets(TEXT["research_questions"], numbered=True)
    r.h("1.2 Scope", 2)
    r.p(TEXT["scope"])

    # ------------------------------------------------------------------ 2 data
    r.h("2. Data Collection", 1)
    r.p(TEXT["data_intro"])
    site_rows = [[s["site"], f"{int(s['patients'])}", pct(s["disease_rate"])] for s in M["sites"]]
    site_rows[-1] = [f"**{c}**" for c in site_rows[-1]]
    r.table(["Hospital dataset", "Patients", "Disease rate"], site_rows, [3.2, 1.3, 1.5],
            "Patients and angiographic CAD prevalence in the four UCI hospital datasets.")
    r.h("2.1 Variables", 2)
    r.table(["Variable", "Meaning"], TEXT["variables"], [1.3, 5.2],
            "The 13 predictors and the diagnosis (UCI documentation).", align_right_from=99, font_size=9)
    r.h("2.2 Data preparation", 2)
    r.bullets(TEXT["data_prep"])
    miss = {row["site"]: row for row in V["missingness_pct"]} if isinstance(V["missingness_pct"][0], dict) else None
    if miss:
        cols = [c for c in ("chol", "fbs", "thalach", "slope", "ca", "thal") if c in next(iter(miss.values()))]
        r.table(["Hospital"] + cols, [[s] + [f"{miss[s][c]:.1f}%" for c in cols] for s in SITES],
                [1.6] + [0.8] * len(cols), "Percentage of missing values by hospital (after cleaning).")

    # ------------------------------------------------------------------ 3 priors
    r.h("3. Prior Knowledge Integration", 1)
    r.p(TEXT["priors_intro"])
    r.h("3.1 Pre-test probability from the medical literature", 2)
    r.p(TEXT["df_text"])
    df_rows = [["30–39", "69.7 / 25.8", "21.8 / 4.2", "5.2 / 0.8", "1.9 / 0.3"],
               ["40–49", "87.3 / 55.2", "46.1 / 13.3", "14.1 / 2.8", "5.5 / 1.0"],
               ["50–59", "92.0 / 79.4", "58.9 / 32.4", "21.5 / 8.4", "9.7 / 3.2"],
               ["60–69", "94.3 / 90.6", "67.1 / 54.4", "28.1 / 18.6", "12.3 / 7.5"]]
    r.table(["Age", "Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"], df_rows,
            [0.9, 1.4, 1.4, 1.4, 1.4],
            "Diamond–Forrester pre-test probability of CAD, % (male / female). Values checked against the "
            "table reproduced by Attia et al. (2004).")
    r.h("3.2 Prior prevalence per hospital and conjugate Bayesian updating", 2)
    r.p(TEXT["beta_text"])
    r.equation("Prior: p ~ Beta(a, b)    →    Posterior after k cases in n patients: p ~ Beta(a + k, b + n − k)")
    prev_rows = [[x["site"], pct(x["prior_mean"]), f"{pct(x['posterior_mean'])} ({pct(x['post_95_low'])}–{pct(x['post_95_high'])})",
                  pct(x["test_prevalence"])] for x in M["prevalence_updating"]]
    r.table(["Hospital", "Prior mean", "Posterior mean (95% interval)", "Test-set prevalence"], prev_rows,
            [1.6, 1.1, 2.4, 1.4], "Beta–Binomial updating of disease prevalence with each hospital's training diagnoses.")
    r.figure("prevalence_updating", "Literature prior (dashed) and posterior (solid) prevalence for each hospital.")
    r.h("3.3 Directional clinical priors on test results", 2)
    r.p(TEXT["clinical_priors_text"])
    r.table(["Feature", "Prior mean (log-odds)", "Prior SD", "Clinical rationale"], TEXT["clinical_prior_rows"],
            [1.0, 1.2, 0.8, 3.5], "Clinical priors used by models M1 and M2 (signs from clinical knowledge; magnitudes are judgements).",
            right_cols={1, 2}, font_size=9)
    r.h("3.4 Finding: a prior–data conflict", 2)
    r.p(TEXT["conflict_text"])
    r.figure("prior_vs_observed_chest_pain", "Observed disease rate vs. literature pre-test probability by chest-pain type (training data).", 5.8)

    # ------------------------------------------------------------------ 4 methods
    r.h("4. Model Development", 1)
    r.h("4.1 Bayesian diagnostic model (PyMC)", 2)
    r.p(TEXT["bayes_intro"])
    r.equation("logit P(CAD_i) = α + γ · logit(PreTest_i) + x_i · β + u_hospital(i),     u_h ~ N(0, τ)")
    r.p(TEXT["bayes_terms"])
    r.table(["Model", "Baseline risk", "Coefficient priors", "Structure", "Purpose"], TEXT["model_rows"],
            [1.05, 1.35, 1.45, 1.25, 1.4], "Bayesian model variants developed during iteration.", align_right_from=99, font_size=9)
    r.p(TEXT["sampling_text"])
    r.h("4.2 Random Forest tuned by Bayesian Optimization", 2)
    for para in TEXT["rf_text"]:
        r.p(para)
    r.table(["Hyperparameter", "Search range", "Selected (main split)"], TEXT["search_space_rows"], [1.8, 2.6, 2.1],
            "Random Forest search space and the configuration chosen by Bayesian Optimization.", align_right_from=99)
    r.figure("rf_optimization", "Best-so-far 5-fold CV log loss: Bayesian Optimization (TPE) vs. random search, 50 trials each.")
    r.h("4.3 Validation design", 2)
    r.bullets(TEXT["validation_design"])

    # ------------------------------------------------------------------ 5 results
    r.h("5. Results: Accuracy of the Diagnostic Tool", 1)
    r.h("5.1 Held-out test set (184 patients)", 2)
    r.p(TEXT["test_intro"])
    order = ["Literature only (Diamond-Forrester)", "M0 Bayesian (weak priors)", "M1 Bayesian (literature priors)",
             "M2 Bayesian (refined)", "Random Forest (default)", "Random Forest (Bayesian-optimised)"]
    short = {"Literature only (Diamond-Forrester)": "Literature only", "M0 Bayesian (weak priors)": "M0 weak priors",
             "M1 Bayesian (literature priors)": "M1 literature priors", "M2 Bayesian (refined)": "**M2 refined**",
             "Random Forest (default)": "RF default", "Random Forest (Bayesian-optimised)": "RF + Bayesian Opt."}
    rows = []
    for k in order:
        x = {**TEST[k], **VTEST.get(k, {})}  # unrounded independently recomputed values where available
        rows.append([short[k], f"{x['roc_auc']:.3f} ({x['auc_ci_low']:.2f}–{x['auc_ci_high']:.2f})", pct(x["accuracy"]),
                     pct(x["sensitivity"]), pct(x["specificity"]), pct(x["ppv"]), pct(x["npv"]), f3(x["brier"]), f3(x["log_loss"])])
    r.table(["Model", "ROC-AUC (95% CI)", "Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "Brier", "Log loss"],
            rows, [1.3, 1.15, 0.7, 0.72, 0.72, 0.55, 0.55, 0.5, 0.6],
            "Test-set accuracy of each model against the actual angiographic diagnosis (threshold 0.5). "
            "All rows were independently reproduced in the review.", bold_rows=(3,), font_size=8.5)
    r.p(TEXT["test_interpret"])
    r.figure("roc_calibration", "ROC curves (left) and calibration (right) on the held-out test set.")
    r.h("5.2 Are the differences real? Paired comparisons", 2)
    r.p(TEXT["paired_intro"])
    labels = {"M2_minus_M0": "M2 refined − M0 weak priors", "M2_minus_M2weak_ablation": "M2 refined − M2 structure with weak priors (ablation)",
              "M1_minus_M0": "M1 literature − M0 weak priors", "RFtuned_minus_RFdefault": "RF Bayesian-optimised − RF default",
              "M2_minus_RFtuned": "M2 refined − RF Bayesian-optimised"}
    prow = []
    for k, lab in labels.items():
        x = MAIN["paired_bootstrap"][k]
        lo, hi = x["auc_diff_95ci"]
        llo, lhi = x["logloss_diff_95ci"]
        prow.append([lab, f"{x['auc_diff']:+.3f}", f"{lo:+.3f} to {hi:+.3f}", f"{x['logloss_diff']:+.3f}", f"{llo:+.3f} to {lhi:+.3f}"])
    r.table(["Comparison", "Δ AUC", "95% CI", "Δ Log loss", "95% CI"], prow, [2.7, 0.75, 1.15, 0.85, 1.15],
            "Paired bootstrap differences (2,000 resamples of the same test patients). Positive Δ AUC and negative "
            "Δ log loss favour the first model.", font_size=9)
    r.p(TEXT["paired_interpret"])
    r.h("5.3 Patient-level Bayesian updating", 2)
    r.p(TEXT["patients_intro"])
    ex_rows = [[f"{e['site']}, {e['sex']} {int(e['age'])}, {e['chest_pain'].lower()}", pct(e["pretest_prob"]),
                f"{pct(e['posttest_prob'])} ({pct(e['94%_low'])}–{pct(e['94%_high'])})", e["actual_diagnosis"]]
               for e in M["patient_examples"]]
    r.table(["Patient", "Literature pre-test", "Post-test, M2 (94% credible interval)", "Actual"], ex_rows,
            [2.4, 1.2, 2.1, 0.8], "Pre-test to post-test probability for six test patients (three with and three without CAD).")
    r.p(TEXT["patients_interpret"])
    r.h("5.4 External validation: leave-one-hospital-out (corrected)", 2)
    r.p(TEXT["loho_intro"])
    lrows = []
    for s in SITES:
        f = LOHO[s]
        lrows.append([f"{s} (n={f['n']}, {pct(f['prevalence'], 0)} CAD)"] + [f3(f["metrics"][m]["roc_auc"]) for m in LOHO_MODELS])
    pooled = LSUM["pooled_out_of_hospital"]
    lrows.append(["**All 920, pooled out-of-hospital**"] + [f"**{f3(pooled[m]['roc_auc'])}**" for m in LOHO_MODELS])
    r.table(["Held-out hospital"] + LOHO_MODELS, lrows, [2.3, 1.05, 1.0, 1.0, 1.15],
            "ROC-AUC on each hospital when the model never saw that hospital (nested RF tuning; shared hospital effect for M2).")
    brow = []
    for s in SITES:
        f = LOHO[s]
        brow.append([s] + [f"{f3(f['metrics'][m]['brier'])} / {pct(f['metrics'][m]['accuracy'], 0)}" for m in LOHO_MODELS])
    brow.append(["**Pooled (920)**"] + [f"**{f3(pooled[m]['brier'])} / {pct(pooled[m]['accuracy'], 0)}**" for m in LOHO_MODELS])
    r.table(["Held-out hospital"] + LOHO_MODELS, brow, [1.6, 1.25, 1.2, 1.2, 1.25],
            "Brier score (lower is better) / accuracy at threshold 0.5 on each unseen hospital.", font_size=9)
    r.figure("loho_auc_corrected", "Corrected leave-one-hospital-out ROC-AUC by held-out hospital.")
    r.p(TEXT["loho_interpret"])
    drow = []
    for s in SITES:
        for lab in ("M1 Bayesian", "M2 Bayesian"):
            g = LOHO[s]["diagnostics"][lab]
            drow.append([s, lab, f"{g['max_rhat']:.2f}", f"{g['min_ess_bulk']:.0f}", str(g["divergences"]), f"{g['target_accept']:.2f}"])
    r.table(["Held-out", "Model", "Max R-hat", "Min bulk ESS", "Divergences", "target_accept"], drow,
            [1.5, 1.2, 0.9, 1.1, 1.0, 1.0], "MCMC diagnostics for every hospital-holdout fit.", align_right_from=2, font_size=9)

    # ------------------------------------------------------------------ 6 iteration
    r.h("6. Iteration: Refining Priors and Likelihood", 1)
    r.p(TEXT["iteration_intro"])
    loo = {x["model"]: x for x in MAIN["loo"]}
    lr = []
    for k in ("M0_weak", "M1_literature", "M2_refined", "M2_weak_ablation"):
        x = loo[k]
        lr.append([k.replace("_", " "), f"{x['elpd_loo']:.1f}", f"{x['p_loo']:.1f}", f"{x['elpd_diff']:.1f}", f"{x['dse']:.1f}", f"{x['weight']:.2f}"])
    r.table(["Model", "ELPD-LOO", "p_loo", "Δ ELPD vs best", "SE of Δ", "Stacking weight"], lr,
            [1.8, 0.95, 0.75, 1.15, 0.85, 1.0], "PSIS-LOO comparison on the training data (higher ELPD is better).")
    r.p(TEXT["loo_interpret"])
    r.h("6.1 What the refined model learned", 2)
    r.bullets(TEXT["m2_learned"])
    r.figure("m2_coefficients", "M2 posterior effects (94% HDI) compared with the clinical prior means.", 4.4)
    r.h("6.2 Bayesian updating as new patient data arrive", 2)
    r.p(TEXT["sequential_intro"])
    seq = M["sequential_updating"]
    r.table(["Patients incorporated"] + [str(int(s["patients_seen"])) for s in seq],
            [["Test ROC-AUC"] + [f3(s["test_auc"]) for s in seq],
             ["Test log loss"] + [f3(s["test_log_loss"]) for s in seq],
             ["Mean coefficient SD"] + [f"{s['mean_posterior_sd']:.2f}" for s in seq]],
            [1.9] + [0.92] * len(seq), "Sequential Bayesian updating of model M1 in four batches of 184 patients.")
    r.figure("sequential_updating", "Test discrimination and posterior uncertainty after each update.")
    r.p(TEXT["sequential_interpret"])

    # ------------------------------------------------------------------ 7-9
    r.h("7. Discussion", 1)
    for para in TEXT["discussion"]:
        r.p(para)
    r.h("8. Limitations and Ethical Considerations", 1)
    r.bullets(TEXT["limitations"])
    r.h("9. Conclusions and Recommendations", 1)
    r.p(TEXT["conclusion"])
    r.bullets(TEXT["recommendations"], numbered=True)

    r.h("References", 1)
    for ref in TEXT["references"]:
        para = d.add_paragraph()
        para.paragraph_format.left_indent = Inches(0.35)
        para.paragraph_format.first_line_indent = Inches(-0.35)
        r._rich(para, ref, size=10)

    # ------------------------------------------------------------------ appendices
    r.page_break()
    r.h("Appendix A. Alignment with the Assignment Steps", 1)
    r.p(TEXT["rubric_note"], italic=True)
    r.table(["Assignment step", "Where addressed", "What was done"], TEXT["alignment_rows"], [1.6, 1.2, 3.7],
            "Mapping of each assignment step to this report.", align_right_from=99, font_size=9)
    r.h("Appendix B. Corrections Made After Independent Review", 1)
    r.p(TEXT["corrections_intro"])
    r.table(["Issue found", "Correction", "Effect"], TEXT["corrections_rows"], [1.9, 2.5, 2.1],
            "Issues identified in review and how they were resolved.", align_right_from=99, font_size=9)
    r.h("Appendix C. Reproducibility", 1)
    r.bullets(TEXT["reproducibility"])
    vers = ", ".join(f"{k} {v}" for k, v in V["versions"].items())
    r.p(f"Python {V['python']}; {vers}.", bold_lead="Software versions: ")

    r.footer_page_numbers()
    d.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    # Use the shared author revision, rather than the legacy narrative above.
    from export_verified_word_report import main
    main()
