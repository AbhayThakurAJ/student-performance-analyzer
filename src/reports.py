"""Report generation module for Student Performance Analyzer.

Provides functions to export data as CSV and generate comprehensive
PDF reports for class-level and individual student analysis using fpdf2.
"""

import io
import os
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
from fpdf import FPDF

from src.analysis import SUBJECT_COLS, detect_at_risk, get_kpi_metrics
from src.grading import get_default_boundaries, get_grade


# ---------------------------------------------------------------------------
# CSV export helpers
# ---------------------------------------------------------------------------

def export_cleaned_csv(df: pd.DataFrame) -> bytes:
    """Convert a cleaned DataFrame to CSV bytes for download.

    Args:
        df: The cleaned student performance DataFrame.

    Returns:
        UTF-8 encoded CSV content as bytes.
    """
    if df is None or df.empty:
        return b""
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8")
    return buf.getvalue()


def export_analysis_csv(stats_df: pd.DataFrame) -> bytes:
    """Convert analysis statistics DataFrame to CSV bytes.

    Args:
        stats_df: DataFrame containing computed statistics
                  (e.g. per-subject mean, median, pass rate).

    Returns:
        UTF-8 encoded CSV content as bytes.
    """
    if stats_df is None or stats_df.empty:
        return b""
    buf = io.BytesIO()
    stats_df.to_csv(buf, index=True, encoding="utf-8")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_str(value) -> str:
    """Return a Latin-1 safe display string for *value*, handling NaN / None."""
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except (TypeError, ValueError):
        pass
    text = str(value)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _embed_image_bytes(
    pdf: FPDF, img_bytes: bytes, x: float, y: float, w: float = 0, h: float = 0
) -> None:
    """Write *img_bytes* to a temp file, embed it in *pdf*, then clean up."""
    if not img_bytes:
        return
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name
        pdf.image(tmp_path, x=x, y=y, w=w, h=h)
    except Exception:
        pass
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Class-level PDF report
# ---------------------------------------------------------------------------

def generate_class_report_pdf(
    df: pd.DataFrame,
    stats_df: pd.DataFrame,
    chart_images: dict[str, bytes] | None = None,
    boundaries: dict[str, int] | None = None,
    pass_mark: int = 40,
    min_failing: int = 2,
) -> bytes:
    """Generate a comprehensive class-level PDF report.

    The report contains:
    * Title page with report name and generation date.
    * Class summary - total students, average score, pass rate, at-risk count.
    * Subject statistics table.
    * Embedded chart images (if provided).
    * Top-10 and bottom-10 student rankings.
    * At-risk students list.

    Args:
        df: Cleaned student performance DataFrame.
        stats_df: Per-subject statistics DataFrame (index = subject names).
        chart_images: Optional mapping of chart name -> PNG bytes.
        boundaries: Optional grade boundaries dict.
        pass_mark: Pass mark threshold used for pass rate and at-risk detection.
        min_failing: Minimum failing subjects to flag a student as at-risk.

    Returns:
        The generated PDF document as bytes.
    """
    if boundaries is None:
        boundaries = get_default_boundaries()

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)

    work_df = df.copy() if df is not None else pd.DataFrame()
    subject_cols = [c for c in SUBJECT_COLS if c in work_df.columns]
    total_students = len(work_df)

    if total_students > 0 and subject_cols:
        if "average" not in work_df.columns:
            work_df["average"] = work_df[subject_cols].mean(axis=1)
        kpis = get_kpi_metrics(work_df, pass_mark=pass_mark, min_failing=min_failing)
        overall_avg = kpis["class_average"]
        pass_rate = kpis["overall_pass_rate"]
        at_risk_df = detect_at_risk(work_df, threshold=pass_mark, min_failing=min_failing)
    else:
        overall_avg = 0.0
        pass_rate = 0.0
        at_risk_df = pd.DataFrame()

    name_col = "name" if "name" in work_df.columns else ("Name" if "Name" in work_df.columns else None)
    id_col = "student_id" if "student_id" in work_df.columns else None

    # ---- Title page --------------------------------------------------------
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.ln(55)
    pdf.cell(0, 15, "Student Performance Analysis Report", align="C", ln=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.ln(8)
    pdf.cell(
        0,
        10,
        f"Generated on: {datetime.now().strftime('%B %d, %Y %I:%M %p')}",
        align="C",
        ln=True,
    )
    pdf.ln(4)
    pdf.cell(0, 10, f"Total Students: {total_students} | Pass Mark: {pass_mark}%", align="C", ln=True)

    # ---- Class summary -----------------------------------------------------
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Class Summary", ln=True)
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Total Students:      {total_students}", ln=True)
    pdf.cell(0, 8, f"Class Average Score: {overall_avg:.2f}", ln=True)
    pdf.cell(0, 8, f"Overall Pass Rate:   {pass_rate:.1f}%", ln=True)
    pdf.cell(0, 8, f"At-Risk Students:    {len(at_risk_df)}", ln=True)
    pdf.ln(6)

    # ---- Subject statistics table ------------------------------------------
    if stats_df is not None and not stats_df.empty:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Subject Statistics", ln=True)
        pdf.ln(2)

        headers = ["Subject"] + [str(c) for c in stats_df.columns]
        usable_w = 190.0
        col_w = usable_w / len(headers)

        short_headers = {
            "pass_count": "Pass",
            "fail_count": "Fail",
            "pass_percentage": "Pass %",
        }

        pdf.set_font("Helvetica", "B", 8)
        for h in headers:
            label = short_headers.get(h, h.title())
            pdf.cell(col_w, 8, _safe_str(label)[:12], border=1, align="C")
        pdf.ln()

        pdf.set_font("Helvetica", "", 8)
        for idx, row in stats_df.iterrows():
            if pdf.get_y() > 265:
                pdf.add_page()
            pdf.cell(col_w, 7, _safe_str(str(idx).title())[:14], border=1)
            for col_name, val in row.items():
                try:
                    if col_name in ("pass_count", "fail_count"):
                        display = f"{int(val)}"
                    else:
                        display = f"{float(val):.1f}"
                except (TypeError, ValueError):
                    display = _safe_str(val)
                pdf.cell(col_w, 7, display[:12], border=1, align="C")
            pdf.ln()
        pdf.ln(6)

    # ---- Charts ------------------------------------------------------------
    if chart_images:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Charts & Visualizations", ln=True)
        pdf.ln(4)

        for chart_name, img_bytes in chart_images.items():
            if pdf.get_y() > 165:
                pdf.add_page()
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 8, _safe_str(chart_name), ln=True)
            pdf.ln(2)
            y_before = pdf.get_y()
            _embed_image_bytes(pdf, img_bytes, x=15, y=y_before, w=175)
            pdf.set_y(y_before + 98)
            pdf.ln(4)

    # ---- Rankings ----------------------------------------------------------
    if total_students > 0 and "average" in work_df.columns:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Student Rankings", ln=True)
        pdf.ln(3)

        sorted_df = work_df.sort_values("average", ascending=False).reset_index(drop=True)
        sorted_df["_rank"] = sorted_df.index + 1

        for section_title, section_df in [
            ("Top 10 Students", sorted_df.head(10)),
            ("Bottom 10 Students", sorted_df.tail(10)),
        ]:
            if pdf.get_y() > 215:
                pdf.add_page()
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 9, section_title, ln=True)
            pdf.ln(2)

            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(15, 7, "Rank", border=1, align="C")
            pdf.cell(25, 7, "ID", border=1, align="C")
            pdf.cell(75, 7, "Name", border=1, align="C")
            pdf.cell(30, 7, "Average", border=1, align="C")
            pdf.cell(20, 7, "Grade", border=1, align="C")
            pdf.ln()

            pdf.set_font("Helvetica", "", 9)
            for _, row in section_df.iterrows():
                if pdf.get_y() > 275:
                    pdf.add_page()
                rank = int(row["_rank"])
                sid = _safe_str(row.get(id_col, "N/A")) if id_col else "N/A"
                name = _safe_str(row.get(name_col, "N/A")) if name_col else "N/A"
                avg = row.get("average", 0)
                pdf.cell(15, 7, str(rank), border=1, align="C")
                pdf.cell(25, 7, sid[:12], border=1, align="C")
                pdf.cell(75, 7, name[:34], border=1)
                pdf.cell(
                    30,
                    7,
                    f"{float(avg):.2f}" if pd.notna(avg) else "N/A",
                    border=1,
                    align="C",
                )
                pdf.cell(20, 7, get_grade(float(avg), boundaries) if pd.notna(avg) else "N/A", border=1, align="C")
                pdf.ln()
            pdf.ln(6)

    # ---- At-risk students --------------------------------------------------
    if not at_risk_df.empty:
        if pdf.get_y() > 200:
            pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, f"At-Risk Students (<{pass_mark}% in {min_failing}+ subjects)", ln=True)
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(22, 7, "ID", border=1, align="C")
        pdf.cell(52, 7, "Name", border=1, align="C")
        pdf.cell(24, 7, "Average", border=1, align="C")
        pdf.cell(92, 7, "Failing Subjects", border=1, align="C")
        pdf.ln()

        pdf.set_font("Helvetica", "", 8)
        for _, row in at_risk_df.sort_values("average").iterrows():
            if pdf.get_y() > 275:
                pdf.add_page()
            sid = _safe_str(row.get(id_col, "N/A")) if id_col else "N/A"
            name = _safe_str(row.get(name_col, "-")) if name_col else "-"
            avg = row.get("average", 0)
            failing = ", ".join(s.title() for s in row.get("failing_subjects", []))
            pdf.cell(22, 7, sid[:10], border=1, align="C")
            pdf.cell(52, 7, name[:26], border=1)
            pdf.cell(
                24,
                7,
                f"{float(avg):.2f}" if pd.notna(avg) else "N/A",
                border=1,
                align="C",
            )
            pdf.cell(92, 7, _safe_str(failing)[:52], border=1)
            pdf.ln()

    return bytes(pdf.output())


# ---------------------------------------------------------------------------
# Individual student PDF report
# ---------------------------------------------------------------------------

def generate_student_report_pdf(
    student_row: pd.Series,
    subject_cols: list[str],
    class_avg: dict[str, float],
    radar_image: bytes | None = None,
    boundaries: dict[str, int] | None = None,
) -> bytes:
    """Generate an individual student report-card PDF.

    Args:
        student_row: A single row (Series) from the student DataFrame.
        subject_cols: List of subject column names to include.
        class_avg: Mapping of subject name -> class average score.
        radar_image: Optional radar-chart PNG bytes for the student.
        boundaries: Optional grade boundaries dict.

    Returns:
        The generated PDF document as bytes.
    """
    if boundaries is None:
        boundaries = get_default_boundaries()

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    name = "N/A"
    for key in ("name", "Name", "Student_Name", "Student Name"):
        if key in student_row.index:
            name = _safe_str(student_row[key])
            break

    student_id = "N/A"
    for key in ("student_id", "ID", "Student_ID", "Roll", "Roll_No"):
        if key in student_row.index:
            student_id = _safe_str(student_row[key])
            break

    student_class = "N/A"
    for key in ("class", "Class", "Section"):
        if key in student_row.index:
            student_class = _safe_str(student_row[key])
            break

    # ---- Header ------------------------------------------------------------
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, "Student Report Card", align="C", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(
        0,
        7,
        f"Generated on: {datetime.now().strftime('%B %d, %Y')}",
        align="C",
        ln=True,
    )
    pdf.ln(6)

    # ---- Student info ------------------------------------------------------
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, "Student Information", ln=True)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(50, 7, "Name:", border=0)
    pdf.cell(0, 7, name, ln=True)
    pdf.cell(50, 7, "Student ID:", border=0)
    pdf.cell(0, 7, student_id, ln=True)
    pdf.cell(50, 7, "Class / Section:", border=0)
    pdf.cell(0, 7, student_class, ln=True)
    pdf.ln(6)

    # ---- Subject marks table -----------------------------------------------
    scores: list[float] = []
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, "Subject-wise Performance", ln=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(55, 8, "Subject", border=1, align="C")
    pdf.cell(25, 8, "Marks", border=1, align="C")
    pdf.cell(25, 8, "Grade", border=1, align="C")
    pdf.cell(35, 8, "Class Avg", border=1, align="C")
    pdf.cell(35, 8, "Difference", border=1, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 10)
    subj_scores: dict[str, float] = {}

    for subj in subject_cols:
        raw = student_row.get(subj)
        try:
            mark = float(raw)
            if np.isnan(mark):
                mark = None
        except (TypeError, ValueError):
            mark = None

        cavg = class_avg.get(subj)
        try:
            cavg_f = float(cavg)
            if np.isnan(cavg_f):
                cavg_f = None
        except (TypeError, ValueError):
            cavg_f = None

        pdf.cell(55, 7, _safe_str(subj.title())[:28], border=1)
        if mark is not None:
            scores.append(mark)
            subj_scores[subj] = mark
            pdf.cell(25, 7, f"{mark:.1f}", border=1, align="C")
            pdf.cell(25, 7, get_grade(mark, boundaries), border=1, align="C")
        else:
            pdf.cell(25, 7, "N/A", border=1, align="C")
            pdf.cell(25, 7, "N/A", border=1, align="C")

        if cavg_f is not None:
            pdf.cell(35, 7, f"{cavg_f:.1f}", border=1, align="C")
            if mark is not None:
                diff = mark - cavg_f
                pdf.cell(35, 7, f"{diff:+.1f}", border=1, align="C")
            else:
                pdf.cell(35, 7, "N/A", border=1, align="C")
        else:
            pdf.cell(35, 7, "N/A", border=1, align="C")
            pdf.cell(35, 7, "N/A", border=1, align="C")
        pdf.ln()

    pdf.ln(6)

    # ---- Overall summary ---------------------------------------------------
    avg_score = float(np.nanmean(scores)) if scores else 0.0
    overall_grade = get_grade(avg_score, boundaries) if scores else "N/A"

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, "Overall Summary", ln=True)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(50, 7, "Average Score:", border=0)
    pdf.cell(0, 7, f"{avg_score:.2f}", ln=True)
    pdf.cell(50, 7, "Overall Grade:", border=0)
    pdf.cell(0, 7, overall_grade, ln=True)
    pdf.ln(6)

    # ---- Strengths & Weaknesses --------------------------------------------
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, "Strengths & Weaknesses", ln=True)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 11)

    if subj_scores:
        best_subj = max(subj_scores, key=subj_scores.get)
        weak_subj = min(subj_scores, key=subj_scores.get)
        pdf.cell(
            0,
            7,
            f"Strongest Subject: {best_subj.title()} ({subj_scores[best_subj]:.1f})",
            ln=True,
        )
        pdf.cell(
            0,
            7,
            f"Weakest Subject:   {weak_subj.title()} ({subj_scores[weak_subj]:.1f})",
            ln=True,
        )
    else:
        pdf.cell(0, 7, "No valid subject scores available.", ln=True)
    pdf.ln(6)

    # ---- Radar chart -------------------------------------------------------
    if radar_image:
        if pdf.get_y() > 160:
            pdf.add_page()
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 9, "Performance Radar", ln=True)
        pdf.ln(2)
        _embed_image_bytes(pdf, radar_image, x=30, y=pdf.get_y(), w=145)

    return bytes(pdf.output())
