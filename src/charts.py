"""Plotly visualization module for Student Performance Analyzer.

Provides chart-generating functions that return plotly.graph_objects.Figure
objects, styled with a consistent color palette and layout suitable for
both light and dark mode rendering.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.grading import get_grade_color

# ---------------------------------------------------------------------------
# Shared palette & layout helpers
# ---------------------------------------------------------------------------

COLORS: list[str] = ["#4F46E5", "#7C3AED", "#EC4899", "#F59E0B", "#10B981", "#06B6D4"]

_COMMON_LAYOUT: dict = dict(
    template="plotly_white",
    font=dict(family="Inter, sans-serif"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=56, r=24, b=48, l=48),
)


def _apply_common_layout(fig: go.Figure, **overrides) -> go.Figure:
    """Merge the common layout defaults with any per-chart overrides."""
    fig.update_layout(**{**_COMMON_LAYOUT, **overrides})
    return fig


def _extract_subjects(stats_df: pd.DataFrame) -> pd.Series:
    """Return subject labels whether 'subject' is a column or the index."""
    if "subject" in stats_df.columns:
        return stats_df["subject"].astype(str).str.title()
    return pd.Series(stats_df.index.astype(str).str.title(), index=stats_df.index)


# ---------------------------------------------------------------------------
# 1. Subject average bar chart
# ---------------------------------------------------------------------------

def subject_average_bar(stats_df: pd.DataFrame) -> go.Figure:
    """Bar chart of average score per subject.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Must contain ``mean`` and either a ``subject`` column or subject index.

    Returns
    -------
    go.Figure
    """
    if stats_df.empty:
        fig = go.Figure()
        return _apply_common_layout(fig, title="Average Score per Subject")

    subjects = _extract_subjects(stats_df)
    n = len(stats_df)
    palette = (COLORS * ((n // len(COLORS)) + 1))[:n]

    fig = go.Figure(
        go.Bar(
            x=subjects,
            y=stats_df["mean"],
            marker_color=palette,
            text=stats_df["mean"].round(1),
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Average: %{y:.1f}<extra></extra>",
        )
    )
    _apply_common_layout(
        fig,
        title="Average Score per Subject",
        xaxis_title="Subject",
        yaxis_title="Average Score",
        yaxis=dict(range=[0, 108]),
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Score distribution histogram
# ---------------------------------------------------------------------------

def score_distribution_histogram(df: pd.DataFrame, subject: str) -> go.Figure:
    """Histogram of scores for a given subject with a vertical mean line.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a column matching *subject*.
    subject : str
        Column name whose distribution to plot.

    Returns
    -------
    go.Figure
    """
    scores = df[subject].dropna() if subject in df.columns else pd.Series(dtype=float)

    fig = go.Figure()
    if scores.empty:
        return _apply_common_layout(
            fig,
            title=f"Score Distribution — {subject.title()}",
            xaxis_title="Score",
            yaxis_title="Number of Students",
        )

    mean_val = float(scores.mean())
    fig.add_trace(
        go.Histogram(
            x=scores,
            xbins=dict(start=0, end=100, size=10),
            marker_color=COLORS[0],
            opacity=0.85,
            name=subject.title(),
            hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
        )
    )

    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color=COLORS[2],
        line_width=2,
        annotation_text=f"Mean: {mean_val:.1f}",
        annotation_position="top right",
        annotation_font_color=COLORS[2],
    )

    _apply_common_layout(
        fig,
        title=f"Score Distribution — {subject.title()}",
        xaxis_title="Score",
        yaxis_title="Number of Students",
        bargap=0.05,
        xaxis=dict(range=[0, 100]),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Box plot for all subjects
# ---------------------------------------------------------------------------

def box_plot_all_subjects(df: pd.DataFrame, subject_cols: list[str]) -> go.Figure:
    """Side-by-side box plots comparing score distributions across subjects.

    Parameters
    ----------
    df : pd.DataFrame
        Contains one column per subject listed in *subject_cols*.
    subject_cols : list[str]
        Column names to include in the comparison.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    for idx, col in enumerate(subject_cols):
        if col not in df.columns:
            continue
        fig.add_trace(
            go.Box(
                y=df[col].dropna(),
                name=col.title(),
                marker_color=COLORS[idx % len(COLORS)],
                boxmean=True,
                hoverinfo="y+name",
            )
        )

    _apply_common_layout(
        fig,
        title="Score Distribution by Subject",
        yaxis_title="Score",
        yaxis=dict(range=[0, 105]),
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Correlation heatmap
# ---------------------------------------------------------------------------

def correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Annotated heatmap of a correlation matrix.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Square DataFrame of pairwise correlation coefficients.

    Returns
    -------
    go.Figure
    """
    if corr_matrix.empty:
        return _apply_common_layout(go.Figure(), title="Subject Correlation Matrix")

    labels = [str(c).title() for c in corr_matrix.columns.tolist()]
    z = corr_matrix.fillna(0.0).values
    text = np.around(z, decimals=2).astype(str)

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=text,
            texttemplate="%{text}",
            hovertemplate="(%{x}, %{y})<br>r = %{z:.2f}<extra></extra>",
            colorbar=dict(title="r"),
        )
    )

    _apply_common_layout(
        fig,
        title="Subject Correlation Matrix",
        xaxis=dict(tickangle=-30),
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Radar / spider chart
# ---------------------------------------------------------------------------

def radar_chart(
    student_scores: dict[str, float],
    class_avg: dict[str, float],
    student_name: str,
) -> go.Figure:
    """Radar chart comparing a single student's scores to the class average.

    Parameters
    ----------
    student_scores : dict[str, float]
        Mapping of subject -> score for the student.
    class_avg : dict[str, float]
        Mapping of subject -> class-average score.
    student_name : str
        Display name used in the chart legend and title.

    Returns
    -------
    go.Figure
    """
    subjects = list(student_scores.keys())
    if not subjects:
        return _apply_common_layout(go.Figure(), title=f"{student_name} vs Class Average")

    labels = [s.title() for s in subjects]
    student_vals = [float(student_scores[s]) if pd.notna(student_scores[s]) else 0.0 for s in subjects]
    avg_vals = [float(class_avg.get(s, 0.0)) if pd.notna(class_avg.get(s, 0.0)) else 0.0 for s in subjects]

    labels_closed = labels + [labels[0]]
    student_closed = student_vals + [student_vals[0]]
    avg_closed = avg_vals + [avg_vals[0]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=student_closed,
            theta=labels_closed,
            fill="toself",
            name=student_name,
            line_color=COLORS[0],
            fillcolor="rgba(79,70,229,0.25)",
            hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=avg_closed,
            theta=labels_closed,
            fill="toself",
            name="Class Average",
            line_color=COLORS[3],
            fillcolor="rgba(245,158,11,0.15)",
            hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
        )
    )

    _apply_common_layout(
        fig,
        title=f"{student_name} vs Class Average",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100]),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# 6. Top & bottom performers
# ---------------------------------------------------------------------------

def top_bottom_performers(df: pd.DataFrame, n: int = 10) -> go.Figure:
    """Horizontal bar chart of the top *n* and bottom *n* students by average.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``student_name`` (or ``name``) and ``average``.
    n : int, optional
        Number of students to show from each end (default 10).

    Returns
    -------
    go.Figure
    """
    if df.empty or "average" not in df.columns:
        return _apply_common_layout(go.Figure(), title=f"Top {n} & Bottom {n} Performers")

    name_col = "student_name" if "student_name" in df.columns else "name"
    work = df.copy()
    if "student_id" in work.columns:
        work["_label"] = work[name_col].astype(str) + " (" + work["student_id"].astype(str) + ")"
    else:
        work["_label"] = work[name_col].astype(str)

    sorted_df = work.sort_values("average", ascending=False).reset_index(drop=True)
    top = sorted_df.head(n)
    bottom = sorted_df.iloc[max(len(sorted_df) - n, len(top)):]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=top["_label"],
            x=top["average"],
            orientation="h",
            name=f"Top {len(top)}",
            marker_color="#10B981",
            text=top["average"].round(1),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Average: %{x:.1f}<extra></extra>",
        )
    )

    if not bottom.empty:
        fig.add_trace(
            go.Bar(
                y=bottom["_label"],
                x=bottom["average"],
                orientation="h",
                name=f"Bottom {len(bottom)}",
                marker_color="#EF4444",
                text=bottom["average"].round(1),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Average: %{x:.1f}<extra></extra>",
            )
        )

    total_bars = len(top) + len(bottom)
    _apply_common_layout(
        fig,
        title=f"Top {len(top)} & Bottom {len(bottom)} Performers" if not bottom.empty else f"Top {len(top)} Performers",
        xaxis_title="Average Score",
        yaxis_title="Student",
        xaxis=dict(range=[0, 110]),
        height=max(360, total_bars * 28 + 100),
    )
    fig.update_yaxes(autorange="reversed")
    return fig


# ---------------------------------------------------------------------------
# 7. Grade distribution pie / donut
# ---------------------------------------------------------------------------

def grade_distribution_pie(grade_counts: dict[str, int], subject: str) -> go.Figure:
    """Donut chart of grade distribution for a subject.

    Parameters
    ----------
    grade_counts : dict[str, int]
        Mapping of grade label (e.g. ``"A"``) to student count.
    subject : str
        Subject name used in the chart title.

    Returns
    -------
    go.Figure
    """
    ordered_grades = ["A", "B", "C", "D", "F", "N/A"]
    labels = [g for g in ordered_grades if grade_counts.get(g, 0) > 0]
    for g, cnt in grade_counts.items():
        if g not in labels and cnt > 0:
            labels.append(g)

    if not labels:
        return _apply_common_layout(go.Figure(), title=f"Grade Distribution — {subject.title()}")

    values = [grade_counts[g] for g in labels]
    slice_colors = [get_grade_color(g) for g in labels]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.45,
            marker=dict(colors=slice_colors),
            sort=False,
            textinfo="label+percent",
            hovertemplate="<b>Grade %{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
        )
    )

    _apply_common_layout(
        fig,
        title=f"Grade Distribution — {subject.title()}",
    )
    return fig


# ---------------------------------------------------------------------------
# 8. Pass / Fail grouped bar chart
# ---------------------------------------------------------------------------

def pass_fail_bar(stats_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart showing pass vs fail counts per subject.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Must contain ``pass_count``, ``fail_count``, and either ``subject``
        column or subject index.

    Returns
    -------
    go.Figure
    """
    if stats_df.empty:
        return _apply_common_layout(go.Figure(), title="Pass vs Fail by Subject")

    subjects = _extract_subjects(stats_df)
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=subjects,
            y=stats_df["pass_count"],
            name="Pass",
            marker_color="#10B981",
            text=stats_df["pass_count"],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Pass: %{y}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            x=subjects,
            y=stats_df["fail_count"],
            name="Fail",
            marker_color="#EF4444",
            text=stats_df["fail_count"],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Fail: %{y}<extra></extra>",
        )
    )

    _apply_common_layout(
        fig,
        title="Pass vs Fail by Subject",
        xaxis_title="Subject",
        yaxis_title="Number of Students",
        barmode="group",
    )
    return fig
