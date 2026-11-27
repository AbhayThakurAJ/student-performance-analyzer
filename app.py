"""Student Performance Analyzer — Interactive Streamlit Dashboard."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src import analysis, charts, cleaning, grading, reports

# ---------------------------------------------------------------------------
# Page configuration & styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Student Performance Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* KPI Metric Cards (light & dark mode compatible) */
[data-testid="stMetric"] {
    background-color: rgba(128, 128, 128, 0.06);
    border: 1px solid rgba(128, 128, 128, 0.20);
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.10);
}
[data-testid="stMetricLabel"] {
    font-weight: 600;
    font-size: 0.9rem;
}
[data-testid="stMetricValue"] {
    font-weight: 700;
}

/* Grade badges */
.grade-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.85rem;
    color: #ffffff;
    text-align: center;
}
.grade-A { background-color: #22C55E; }
.grade-B { background-color: #84CC16; color: #0f172a; }
.grade-C { background-color: #EAB308; color: #0f172a; }
.grade-D { background-color: #F97316; }
.grade-F { background-color: #EF4444; }
.grade-NA { background-color: #94A3B8; }

/* Info cards */
.student-card {
    background-color: rgba(79, 70, 229, 0.06);
    border: 1px solid rgba(79, 70, 229, 0.25);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

SUBJECT_COLS = cleaning.SUBJECT_COLS
SAMPLE_DATA_PATH = Path(__file__).parent / "data" / "sample_students.csv"


# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_csv_bytes(raw_bytes: bytes) -> pd.DataFrame:
    """Read raw CSV bytes into a DataFrame preserving blank strings for cleaning."""
    return pd.read_csv(pd.io.common.BytesIO(raw_bytes), keep_default_na=True)


@st.cache_data(show_spinner=False)
def load_sample_csv(path_str: str) -> pd.DataFrame:
    """Load the bundled sample CSV dataset."""
    return pd.read_csv(path_str, keep_default_na=True)


@st.cache_data(show_spinner=False)
def run_cleaning(raw_df: pd.DataFrame, missing_strategy: str) -> tuple[pd.DataFrame, dict]:
    """Run the cached cleaning pipeline on raw data."""
    return cleaning.clean_data(raw_df, missing_strategy=missing_strategy)


@st.cache_data(show_spinner=False)
def compute_graded_df(
    cleaned_df: pd.DataFrame,
    boundaries: dict[str, int],
) -> pd.DataFrame:
    """Assign subject grades and overall average grades."""
    graded = grading.assign_grades(cleaned_df, SUBJECT_COLS, boundaries=boundaries)
    return analysis.student_strengths_weaknesses(graded)


def _grade_style_cell(val: object) -> str:
    """Return CSS background/text color style for a grade cell."""
    grade = str(val)
    if grade in ("A", "B"):
        return "background-color: rgba(34, 197, 94, 0.22); color: #166534; font-weight: 700;"
    if grade in ("C", "D"):
        return "background-color: rgba(234, 179, 8, 0.24); color: #854d0e; font-weight: 700;"
    if grade == "F":
        return "background-color: rgba(239, 68, 68, 0.24); color: #991b1b; font-weight: 700;"
    return ""


def _style_dataframe(df: pd.DataFrame, grade_cols: list[str]):
    """Apply color coding to grade columns in a DataFrame."""
    existing = [c for c in grade_cols if c in df.columns]
    styler = df.style
    if existing:
        if hasattr(styler, "map"):
            styler = styler.map(_grade_style_cell, subset=existing)
        else:
            styler = styler.applymap(_grade_style_cell, subset=existing)
    return styler


def _grade_badge_html(grade: str) -> str:
    """Render an HTML pill badge for a letter grade."""
    cls_suffix = grade if grade in ("A", "B", "C", "D", "F") else "NA"
    return f'<span class="grade-badge grade-{cls_suffix}">{grade}</span>'


def _try_chart_png(fig, timeout: float = 8.0) -> bytes | None:
    """Export a Plotly figure to PNG bytes via kaleido with a timeout guard."""
    import threading

    result: list[bytes | None] = [None]

    def _worker() -> None:
        try:
            result[0] = fig.to_image(format="png", width=900, height=500, scale=2)
        except Exception:
            result[0] = None

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    return result[0]


# ---------------------------------------------------------------------------
# Sidebar: Data Upload, Grading Settings, Filters
# ---------------------------------------------------------------------------

def render_sidebar() -> tuple[pd.DataFrame | None, str, dict[str, int], int, int]:
    """Render sidebar controls and return raw DataFrame + settings."""
    st.sidebar.title("📊 Controls")

    # 1. Data Source
    st.sidebar.header("1. Data Source & Cleaning")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Student CSV",
        type=["csv"],
        help="Required columns: student_id, name, class, math, science, english, history, computer, hindi",
    )

    if "use_sample" not in st.session_state:
        st.session_state["use_sample"] = True

    if uploaded_file is not None:
        st.session_state["use_sample"] = False
    elif st.sidebar.button("🔄 Use Sample Dataset", use_container_width=True):
        st.session_state["use_sample"] = True

    missing_strategy = st.sidebar.radio(
        "Missing / Invalid Value Handling",
        options=["median", "drop"],
        format_func=lambda s: "Fill with subject median" if s == "median" else "Drop incomplete rows",
        index=0,
    )

    raw_df: pd.DataFrame | None = None
    if uploaded_file is not None:
        try:
            raw_bytes = uploaded_file.getvalue()
            if not raw_bytes or not raw_bytes.strip():
                st.error("The uploaded CSV file is empty. Please upload a valid file or use the sample dataset.")
                return None, missing_strategy, grading.get_default_boundaries(), 40, 2
            raw_df = load_csv_bytes(raw_bytes)
        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV file contains no data or headers.")
            return None, missing_strategy, grading.get_default_boundaries(), 40, 2
        except Exception as exc:
            st.error(f"Could not read the uploaded CSV file: {exc}")
            return None, missing_strategy, grading.get_default_boundaries(), 40, 2
    elif st.session_state.get("use_sample", True):
        if SAMPLE_DATA_PATH.exists():
            raw_df = load_sample_csv(str(SAMPLE_DATA_PATH))
        else:
            st.error("Sample dataset not found at `data/sample_students.csv`.")
            return None, missing_strategy, grading.get_default_boundaries(), 40, 2

    # 2. Grading Settings
    st.sidebar.header("2. Grading & At-Risk")
    with st.sidebar.expander("⚙️ Grading Settings", expanded=False):
        defaults = grading.get_default_boundaries()
        a_cut = st.slider("Grade A minimum (%)", min_value=50, max_value=100, value=defaults["A"])
        b_cut = st.slider("Grade B minimum (%)", min_value=30, max_value=a_cut - 1, value=min(defaults["B"], a_cut - 1))
        c_cut = st.slider("Grade C minimum (%)", min_value=15, max_value=b_cut - 1, value=min(defaults["C"], b_cut - 1))
        d_cut = st.slider("Grade D minimum (%)", min_value=1, max_value=c_cut - 1, value=min(defaults["D"], c_cut - 1))
        pass_mark = st.slider("Pass Mark (%)", min_value=1, max_value=90, value=40)
        min_failing = st.slider("At-Risk: Min failing subjects", min_value=1, max_value=len(SUBJECT_COLS), value=2)

        boundaries = {"A": a_cut, "B": b_cut, "C": c_cut, "D": d_cut, "F": 0}
        st.session_state["boundaries"] = boundaries
        st.session_state["pass_mark"] = pass_mark
        st.session_state["min_failing"] = min_failing

    return raw_df, missing_strategy, boundaries, pass_mark, min_failing


def apply_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Render sidebar filter controls and return filtered DataFrame + chosen subject."""
    st.sidebar.header("3. Filters")
    with st.sidebar.expander("🔍 Data Filters", expanded=True):
        all_classes = sorted(df["class"].dropna().astype(str).unique().tolist())
        selected_classes = st.multiselect(
            "Class / Section",
            options=all_classes,
            default=all_classes,
        )

        selected_subject = st.selectbox(
            "Focus Subject",
            options=SUBJECT_COLS,
            format_func=lambda s: s.title(),
        )

        score_range = st.slider(
            "Average Score Range",
            min_value=0,
            max_value=100,
            value=(0, 100),
        )

        search_query = st.text_input(
            "Search Student (Name or ID)",
            placeholder="e.g., Aarav or S001",
        ).strip()

    filtered = df.copy()
    if selected_classes:
        filtered = filtered[filtered["class"].astype(str).isin(selected_classes)]
    else:
        filtered = filtered.iloc[0:0]

    if not filtered.empty:
        filtered = filtered[
            (filtered["average"] >= score_range[0]) & (filtered["average"] <= score_range[1])
        ]

    if search_query and not filtered.empty:
        q = search_query.lower()
        name_match = filtered["name"].astype(str).str.lower().str.contains(q, na=False)
        id_match = filtered["student_id"].astype(str).str.lower().str.contains(q, na=False)
        filtered = filtered[name_match | id_match]

    return filtered, selected_subject


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------

def render_overview_tab(
    filtered_df: pd.DataFrame,
    raw_count: int,
    clean_report: dict,
    pass_mark: int,
    min_failing: int,
    selected_subject: str,
) -> None:
    """Render the Overview tab."""
    kpis = analysis.get_kpi_metrics(filtered_df, pass_mark=pass_mark, min_failing=min_failing)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", f"{kpis['total_students']}")
    col2.metric("Class Average", f"{kpis['class_average']:.1f}%")
    col3.metric("Overall Pass Rate", f"{kpis['overall_pass_rate']:.1f}%")
    col4.metric("At-Risk Students", f"{kpis['at_risk_count']}")

    # Data quality summary
    total_invalid = sum(clean_report.get("invalid_marks", {}).values())
    total_missing = sum(clean_report.get("missing_values", {}).values())
    dup_removed = clean_report.get("duplicates_removed", 0)
    with st.expander("🧹 Data Quality Summary", expanded=False):
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Raw Rows Loaded", raw_count)
        q2.metric("Duplicates Removed", dup_removed)
        q3.metric("Invalid Marks Detected", total_invalid)
        q4.metric("Missing Marks Handled", total_missing)

        detail_df = pd.DataFrame(
            {
                "Subject": [s.title() for s in SUBJECT_COLS],
                "Invalid Marks Fixed": [clean_report.get("invalid_marks", {}).get(s, 0) for s in SUBJECT_COLS],
                "Missing Marks Handled": [clean_report.get("missing_values", {}).get(s, 0) for s in SUBJECT_COLS],
            }
        )
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

    if filtered_df.empty:
        st.info("No students match the current filters. Adjust the sidebar filters to view charts.")
        return

    stats_df = analysis.subject_statistics(filtered_df, pass_mark=pass_mark)

    c_left, c_right = st.columns(2)
    with c_left:
        st.plotly_chart(
            charts.subject_average_bar(stats_df),
            use_container_width=True,
            key="overview_subject_avg_bar",
        )
    with c_right:
        st.plotly_chart(
            charts.pass_fail_bar(stats_df),
            use_container_width=True,
            key="overview_pass_fail_bar",
        )

    c_hist, c_box = st.columns(2)
    with c_hist:
        hist_subject = st.selectbox(
            "Histogram Subject",
            options=SUBJECT_COLS,
            index=SUBJECT_COLS.index(selected_subject),
            format_func=lambda s: s.title(),
            key="overview_hist_subj",
        )
        st.plotly_chart(
            charts.score_distribution_histogram(filtered_df, hist_subject),
            use_container_width=True,
            key="overview_hist_chart",
        )
    with c_box:
        st.plotly_chart(
            charts.box_plot_all_subjects(filtered_df, SUBJECT_COLS),
            use_container_width=True,
            key="overview_box_chart",
        )

    at_risk_df = analysis.detect_at_risk(filtered_df, threshold=pass_mark, min_failing=min_failing)
    st.subheader(f"⚠️ At-Risk Students (< {pass_mark}% in {min_failing}+ subjects)")
    if at_risk_df.empty:
        st.success("No students are currently flagged as at-risk under these criteria.")
    else:
        display_risk = at_risk_df[
            ["student_id", "name", "class", "average", "average_grade", "failing_subjects"]
        ].copy()
        display_risk["average"] = display_risk["average"].round(2)
        display_risk["failing_subjects"] = display_risk["failing_subjects"].apply(
            lambda lst: ", ".join(s.title() for s in lst)
        )
        st.dataframe(
            _style_dataframe(display_risk, ["average_grade"]),
            use_container_width=True,
            hide_index=True,
        )


def render_subject_tab(
    filtered_df: pd.DataFrame,
    pass_mark: int,
    selected_subject: str,
) -> None:
    """Render the Subject Analysis tab."""
    if filtered_df.empty:
        st.info("No student records available for Subject Analysis under current filters.")
        return

    subj = st.selectbox(
        "Select Subject to Analyze",
        options=SUBJECT_COLS,
        index=SUBJECT_COLS.index(selected_subject),
        format_func=lambda s: s.title(),
        key="subject_tab_select",
    )

    stats_df = analysis.subject_statistics(filtered_df, pass_mark=pass_mark)
    row = stats_df.loc[subj]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Mean", f"{row['mean']:.1f}")
    m2.metric("Median", f"{row['median']:.1f}")
    m3.metric("Std Dev", f"{row['std']:.1f}")
    m4.metric("Min / Max", f"{row['min']:.0f} / {row['max']:.0f}")
    m5.metric("Pass Rate", f"{row['pass_percentage']:.1f}%")

    c1, c2 = st.columns(2)
    with c1:
        grade_counts = grading.grade_distribution(filtered_df, subj)
        st.plotly_chart(
            charts.grade_distribution_pie(grade_counts, subj),
            use_container_width=True,
            key="subject_grade_pie",
        )
    with c2:
        st.plotly_chart(
            charts.score_distribution_histogram(filtered_df, subj),
            use_container_width=True,
            key="subject_hist_chart",
        )

    st.subheader("All Subjects Statistical Summary")
    formatted_stats = stats_df.copy()
    formatted_stats.index = [s.title() for s in formatted_stats.index]
    formatted_stats = formatted_stats.round(2)
    st.dataframe(formatted_stats, use_container_width=True)

    st.subheader("Subject Correlation Analysis")
    corr = analysis.correlation_matrix(filtered_df)
    cc1, cc2 = st.columns([3, 2])
    with cc1:
        st.plotly_chart(
            charts.correlation_heatmap(corr),
            use_container_width=True,
            key="subject_corr_heatmap",
        )
    with cc2:
        corr_display = corr.round(2).copy()
        corr_display.index = [s.title() for s in corr_display.index]
        corr_display.columns = [s.title() for s in corr_display.columns]
        st.dataframe(corr_display, use_container_width=True)


def render_student_tab(
    filtered_df: pd.DataFrame,
    pass_mark: int,
    min_failing: int,
) -> None:
    """Render the Student Analysis tab."""
    if filtered_df.empty:
        st.info("No students match the current filters. Adjust sidebar filters to inspect a student.")
        return

    rankings_df = analysis.student_rankings(filtered_df, subject=None)
    rank_map = dict(zip(rankings_df["student_id"], rankings_df["rank"]))

    student_options = filtered_df["student_id"].astype(str).tolist()
    label_map = {
        str(r["student_id"]): f"{r['name']} ({r['student_id']} • {r['class']})"
        for _, r in filtered_df.iterrows()
    }

    selected_id = st.selectbox(
        "Select Student",
        options=student_options,
        format_func=lambda sid: label_map.get(sid, sid),
    )

    student_row = filtered_df[filtered_df["student_id"].astype(str) == selected_id].iloc[0]
    student_rank = rank_map.get(student_row["student_id"], "-")

    # Profile summary metrics
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Student", f"{student_row['name']}")
    s2.metric("Class • ID", f"{student_row['class']} • {student_row['student_id']}")
    s3.metric("Overall Average", f"{student_row['average']:.2f}%")
    s4.metric("Rank (Filtered)", f"#{student_rank} / {len(filtered_df)}")

    overall_grade = str(student_row["average_grade"])
    failing_count = sum(1 for s in SUBJECT_COLS if student_row[s] < pass_mark)
    status_badge = (
        '<span class="grade-badge grade-F">At-Risk</span>'
        if failing_count >= min_failing
        else '<span class="grade-badge grade-A">On Track</span>'
    )
    st.markdown(
        f"**Overall Grade:** {_grade_badge_html(overall_grade)} &nbsp;&nbsp; **Status:** {status_badge}",
        unsafe_allow_html=True,
    )

    class_avg = {s: float(filtered_df[s].mean()) for s in SUBJECT_COLS}
    student_scores = {s: float(student_row[s]) for s in SUBJECT_COLS}

    left_col, right_col = st.columns([1, 1])
    with left_col:
        st.subheader("Subject-wise Scorecard")
        scorecard = pd.DataFrame(
            {
                "Subject": [s.title() for s in SUBJECT_COLS],
                "Marks": [round(float(student_row[s]), 1) for s in SUBJECT_COLS],
                "Grade": [student_row[f"{s}_grade"] for s in SUBJECT_COLS],
                "Class Avg": [round(class_avg[s], 1) for s in SUBJECT_COLS],
                "Diff": [round(float(student_row[s]) - class_avg[s], 1) for s in SUBJECT_COLS],
                "Status": [grading.get_pass_fail(float(student_row[s]), pass_mark) for s in SUBJECT_COLS],
            }
        )
        st.dataframe(
            _style_dataframe(scorecard, ["Grade"]),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Strengths & Weaknesses")
        st.success(
            f"**Strongest Subject:** {str(student_row['strongest_subject']).title()} "
            f"({student_row['strongest_score']:.1f}%)"
        )
        st.warning(
            f"**Weakest Subject:** {str(student_row['weakest_subject']).title()} "
            f"({student_row['weakest_score']:.1f}%)"
        )

    with right_col:
        st.plotly_chart(
            charts.radar_chart(student_scores, class_avg, str(student_row["name"])),
            use_container_width=True,
            key="student_radar_chart",
        )


def render_rankings_tab(
    filtered_df: pd.DataFrame,
    boundaries: dict[str, int],
    pass_mark: int,
) -> None:
    """Render the Rankings tab."""
    if filtered_df.empty:
        st.info("No students match the current filters. Adjust sidebar filters to view rankings.")
        return

    col_mode, col_subj, col_n = st.columns([1, 1, 1])
    with col_mode:
        rank_mode = st.radio("Rank By", ["Overall Average", "By Subject"], horizontal=True)
    with col_subj:
        subj = st.selectbox(
            "Subject",
            options=SUBJECT_COLS,
            format_func=lambda s: s.title(),
            disabled=(rank_mode == "Overall Average"),
            key="rankings_subj",
        )
    with col_n:
        top_n = st.slider("Top / Bottom N", min_value=3, max_value=20, value=min(10, max(3, len(filtered_df))))

    target_subj = None if rank_mode == "Overall Average" else subj
    ranked = analysis.student_rankings(filtered_df, subject=target_subj)
    ranked["grade"] = ranked["score"].apply(lambda s: grading.get_grade(s, boundaries))
    ranked["score"] = ranked["score"].round(2)

    st.subheader("Leaderboard")
    st.dataframe(
        _style_dataframe(ranked, ["grade"]),
        use_container_width=True,
        hide_index=True,
    )

    st.plotly_chart(
        charts.top_bottom_performers(filtered_df, n=top_n),
        use_container_width=True,
        key="rankings_top_bottom_chart",
    )

    st.subheader("Class / Section Summary")
    cls_sum = analysis.class_summary(filtered_df, pass_mark=pass_mark).round(2)
    st.dataframe(cls_sum, use_container_width=True)


def render_reports_tab(
    filtered_df: pd.DataFrame,
    boundaries: dict[str, int],
    pass_mark: int,
    min_failing: int,
) -> None:
    """Render the Reports tab with CSV and PDF downloads."""
    if filtered_df.empty:
        st.info("No student records available to export. Adjust filters or load data.")
        return

    stats_df = analysis.subject_statistics(filtered_df, pass_mark=pass_mark)

    st.subheader("📥 CSV Exports")
    c1, c2 = st.columns(2)
    with c1:
        cleaned_csv = reports.export_cleaned_csv(filtered_df)
        st.download_button(
            label="⬇️ Download Cleaned Student Data (CSV)",
            data=cleaned_csv,
            file_name="cleaned_students.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        stats_csv = reports.export_analysis_csv(stats_df)
        st.download_button(
            label="⬇️ Download Subject Analysis Summary (CSV)",
            data=stats_csv,
            file_name="subject_analysis_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.divider()
    st.subheader("📄 Class Performance PDF Report")
    if st.button("Generate Class PDF Report", type="primary"):
        with st.spinner("Generating Class PDF Report with charts..."):
            chart_imgs: dict[str, bytes] = {}
            avg_png = _try_chart_png(charts.subject_average_bar(stats_df))
            if avg_png:
                chart_imgs["Average Score per Subject"] = avg_png
            pf_png = _try_chart_png(charts.pass_fail_bar(stats_df))
            if pf_png:
                chart_imgs["Pass vs Fail by Subject"] = pf_png
            corr_png = _try_chart_png(charts.correlation_heatmap(analysis.correlation_matrix(filtered_df)))
            if corr_png:
                chart_imgs["Subject Correlation Matrix"] = corr_png

            pdf_bytes = reports.generate_class_report_pdf(
                filtered_df,
                stats_df,
                chart_images=chart_imgs or None,
                boundaries=boundaries,
                pass_mark=pass_mark,
                min_failing=min_failing,
            )
            st.session_state["class_pdf_bytes"] = pdf_bytes

    if "class_pdf_bytes" in st.session_state:
        st.download_button(
            label="⬇️ Download Class Report (PDF)",
            data=st.session_state["class_pdf_bytes"],
            file_name="class_performance_report.pdf",
            mime="application/pdf",
        )

    st.divider()
    st.subheader("🎓 Individual Student Report Card (PDF)")
    student_options = filtered_df["student_id"].astype(str).tolist()
    label_map = {
        str(r["student_id"]): f"{r['name']} ({r['student_id']} • {r['class']})"
        for _, r in filtered_df.iterrows()
    }
    chosen_sid = st.selectbox(
        "Select Student for Report Card",
        options=student_options,
        format_func=lambda sid: label_map.get(sid, sid),
        key="report_student_select",
    )

    if st.button("Generate Student Report Card PDF"):
        with st.spinner("Generating Student Report Card PDF..."):
            student_row = filtered_df[filtered_df["student_id"].astype(str) == chosen_sid].iloc[0]
            class_avg = {s: float(filtered_df[s].mean()) for s in SUBJECT_COLS}
            student_scores = {s: float(student_row[s]) for s in SUBJECT_COLS}
            radar_fig = charts.radar_chart(student_scores, class_avg, str(student_row["name"]))
            radar_png = _try_chart_png(radar_fig)

            student_pdf = reports.generate_student_report_pdf(
                student_row=student_row,
                subject_cols=SUBJECT_COLS,
                class_avg=class_avg,
                radar_image=radar_png,
                boundaries=boundaries,
            )
            st.session_state["student_pdf_bytes"] = student_pdf
            st.session_state["student_pdf_name"] = f"report_card_{chosen_sid}.pdf"

    if "student_pdf_bytes" in st.session_state:
        st.download_button(
            label=f"⬇️ Download {st.session_state.get('student_pdf_name', 'student_report.pdf')}",
            data=st.session_state["student_pdf_bytes"],
            file_name=st.session_state.get("student_pdf_name", "student_report.pdf"),
            mime="application/pdf",
        )


# ---------------------------------------------------------------------------
# Main application entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the Streamlit Student Performance Analyzer application."""
    st.title("📊 Student Performance Analyzer")
    st.caption(
        "Analyze academic performance across subjects, detect at-risk students, "
        "configure grading scales, and export automated PDF/CSV reports."
    )

    raw_df, missing_strategy, boundaries, pass_mark, min_failing = render_sidebar()
    if raw_df is None:
        st.info("Upload a student CSV file in the sidebar or click **Use Sample Dataset** to begin.")
        return

    if raw_df.empty:
        st.warning("The loaded dataset has 0 rows. Please upload a non-empty CSV file.")
        return

    is_valid, missing_cols = cleaning.validate_columns(raw_df)
    if not is_valid:
        st.error(
            f"Missing required column(s): {', '.join(missing_cols)}. "
            f"Expected columns: {', '.join(cleaning.REQUIRED_COLS)}."
        )
        return

    with st.spinner("Cleaning and analyzing student records..."):
        cleaned_df, clean_report = run_cleaning(raw_df, missing_strategy)
        if cleaned_df.empty:
            st.warning(
                "All rows were removed during cleaning (e.g., due to missing values with the 'drop' strategy). "
                "Try switching to 'Fill with subject median' in the sidebar."
            )
            return
        graded_df = compute_graded_df(cleaned_df, boundaries)

    filtered_df, selected_subject = apply_filters(graded_df)

    tab_overview, tab_subject, tab_student, tab_rankings, tab_reports = st.tabs(
        [
            "📋 Overview",
            "📚 Subject Analysis",
            "👤 Student Analysis",
            "🏆 Rankings",
            "📄 Reports",
        ]
    )

    with tab_overview:
        render_overview_tab(
            filtered_df=filtered_df,
            raw_count=len(raw_df),
            clean_report=clean_report,
            pass_mark=pass_mark,
            min_failing=min_failing,
            selected_subject=selected_subject,
        )

    with tab_subject:
        render_subject_tab(
            filtered_df=filtered_df,
            pass_mark=pass_mark,
            selected_subject=selected_subject,
        )

    with tab_student:
        render_student_tab(
            filtered_df=filtered_df,
            pass_mark=pass_mark,
            min_failing=min_failing,
        )

    with tab_rankings:
        render_rankings_tab(
            filtered_df=filtered_df,
            boundaries=boundaries,
            pass_mark=pass_mark,
        )

    with tab_reports:
        render_reports_tab(
            filtered_df=filtered_df,
            boundaries=boundaries,
            pass_mark=pass_mark,
            min_failing=min_failing,
        )


if __name__ == "__main__":
    main()
