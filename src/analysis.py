"""Statistical analysis module for student performance data."""

import numpy as np
import pandas as pd

SUBJECT_COLS: list[str] = ["math", "science", "english", "history", "computer", "hindi"]


def subject_statistics(df: pd.DataFrame, pass_mark: int = 40) -> pd.DataFrame:
    """Compute descriptive statistics for each subject.

    For every subject column, calculates mean, median, standard deviation,
    min, max, pass count (score >= pass_mark), fail count, and pass percentage.

    Args:
        df: DataFrame containing at least the subject columns.
        pass_mark: Minimum score required to pass a subject (default 40).

    Returns:
        A DataFrame with subjects as rows and statistics as columns.
    """
    records: list[dict] = []
    for subject in SUBJECT_COLS:
        if subject not in df.columns:
            continue
        col = df[subject].dropna()
        total = len(col)
        pass_count = int((col >= pass_mark).sum())
        fail_count = int((col < pass_mark).sum())
        std_val = float(col.std(ddof=1)) if total > 1 else 0.0
        records.append(
            {
                "subject": subject,
                "mean": float(col.mean()) if total else 0.0,
                "median": float(col.median()) if total else 0.0,
                "std": 0.0 if np.isnan(std_val) else std_val,
                "min": float(col.min()) if total else 0.0,
                "max": float(col.max()) if total else 0.0,
                "pass_count": pass_count,
                "fail_count": fail_count,
                "pass_percentage": (pass_count / total * 100) if total else 0.0,
            }
        )
    if not records:
        return pd.DataFrame(
            columns=[
                "mean",
                "median",
                "std",
                "min",
                "max",
                "pass_count",
                "fail_count",
                "pass_percentage",
            ]
        )
    return pd.DataFrame(records).set_index("subject")


def student_rankings(
    df: pd.DataFrame, subject: str | None = None
) -> pd.DataFrame:
    """Rank students by score.

    If *subject* is ``None`` the ranking is based on the average across all
    subjects; otherwise it is based on the given subject column.

    Args:
        df: DataFrame with student_id, name, and subject columns.
        subject: Optional subject name to rank by.

    Returns:
        A DataFrame with columns rank, student_id, name, (class if present),
        and score, sorted by score descending.
    """
    if df.empty:
        cols = ["rank", "student_id", "name", "class", "score"] if "class" in df.columns else ["rank", "student_id", "name", "score"]
        return pd.DataFrame(columns=cols)

    if subject is None:
        score = df[SUBJECT_COLS].mean(axis=1)
    else:
        score = df[subject]

    data: dict = {
        "student_id": df["student_id"],
        "name": df["name"],
    }
    if "class" in df.columns:
        data["class"] = df["class"]
    data["score"] = score

    result = pd.DataFrame(data)
    result = result.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)
    result["rank"] = result["score"].rank(method="min", ascending=False).fillna(len(result)).astype(int)
    ordered_cols = ["rank", "student_id", "name"] + (["class"] if "class" in result.columns else []) + ["score"]
    return result[ordered_cols]


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return the Pearson correlation matrix of subject scores.

    Args:
        df: DataFrame containing the subject columns.

    Returns:
        A square DataFrame of pairwise correlation coefficients.
    """
    available = [c for c in SUBJECT_COLS if c in df.columns]
    if df.empty or not available:
        return pd.DataFrame(index=available, columns=available, dtype=float)
    return df[available].corr()


def student_strengths_weaknesses(df: pd.DataFrame) -> pd.DataFrame:
    """Identify the strongest and weakest subject for each student.

    Adds four columns to the input DataFrame:
    ``strongest_subject``, ``strongest_score``,
    ``weakest_subject``, and ``weakest_score``.

    Args:
        df: DataFrame with subject columns.

    Returns:
        The augmented DataFrame with strength / weakness columns appended.
    """
    result = df.copy()
    if result.empty:
        result["strongest_subject"] = pd.Series(dtype=str)
        result["strongest_score"] = pd.Series(dtype=float)
        result["weakest_subject"] = pd.Series(dtype=str)
        result["weakest_score"] = pd.Series(dtype=float)
        return result

    subjects = result[SUBJECT_COLS]
    result["strongest_subject"] = subjects.idxmax(axis=1)
    result["strongest_score"] = subjects.max(axis=1)
    result["weakest_subject"] = subjects.idxmin(axis=1)
    result["weakest_score"] = subjects.min(axis=1)
    return result


def detect_at_risk(
    df: pd.DataFrame, threshold: int = 40, min_failing: int = 2
) -> pd.DataFrame:
    """Detect students who are at risk of academic failure.

    A student is considered at-risk when they score below *threshold* in
    at least *min_failing* subjects.

    Args:
        df: DataFrame with subject columns.
        threshold: Score below which a subject is considered failed.
        min_failing: Minimum number of failing subjects to flag a student.

    Returns:
        A filtered DataFrame of at-risk students with an additional
        ``failing_subjects`` column listing the names of failed subjects.
    """
    if df.empty:
        result = df.copy()
        result["failing_subjects"] = pd.Series(dtype=object)
        return result

    failing_mask = df[SUBJECT_COLS] < threshold

    def _failing_names(row: pd.Series) -> list[str]:
        return [subj for subj in SUBJECT_COLS if bool(row[subj])]

    failing_subjects = failing_mask.apply(_failing_names, axis=1)
    failing_counts = failing_mask.sum(axis=1)

    at_risk_idx = failing_counts[failing_counts >= min_failing].index
    result = df.loc[at_risk_idx].copy()
    result["failing_subjects"] = failing_subjects.loc[at_risk_idx]
    return result


def class_summary(df: pd.DataFrame, pass_mark: int = 40) -> pd.DataFrame:
    """Summarise performance grouped by class.

    For each class computes student count, average score across all
    subjects, overall pass rate (percentage of subject scores >= pass_mark),
    and the name of the top-scoring student.

    Args:
        df: DataFrame with a ``class`` column, subject columns, and ``name``.
        pass_mark: Score threshold for passing a subject (default 40).

    Returns:
        A summary DataFrame indexed by class.
    """
    if df.empty or "class" not in df.columns:
        return pd.DataFrame(
            columns=["student_count", "avg_score", "pass_rate", "top_student"]
        )

    work_df = df.copy()
    work_df["_avg_score"] = work_df[SUBJECT_COLS].mean(axis=1)

    rows: list[dict] = []
    for cls_name, group in work_df.groupby("class", sort=True):
        student_count = len(group)
        avg_score = float(group["_avg_score"].mean())
        scores = group[SUBJECT_COLS].values.flatten()
        valid_scores = scores[~np.isnan(scores)]
        pass_rate = (
            float((valid_scores >= pass_mark).sum() / len(valid_scores) * 100)
            if len(valid_scores)
            else 0.0
        )
        top_idx = group["_avg_score"].idxmax()
        top_student = str(group.loc[top_idx, "name"]) if pd.notna(top_idx) else "N/A"
        rows.append(
            {
                "class": cls_name,
                "student_count": student_count,
                "avg_score": avg_score,
                "pass_rate": pass_rate,
                "top_student": top_student,
            }
        )

    return pd.DataFrame(rows).set_index("class")


def get_kpi_metrics(
    df: pd.DataFrame, pass_mark: int = 40, min_failing: int = 2
) -> dict:
    """Return high-level KPI metrics for the dataset.

    Args:
        df: DataFrame with subject columns.
        pass_mark: Score threshold for passing a subject.
        min_failing: Minimum failing subjects to flag a student as at-risk.

    Returns:
        A dict with keys ``total_students``, ``class_average``,
        ``overall_pass_rate``, and ``at_risk_count``.
    """
    total_students = len(df)
    if total_students == 0:
        return {
            "total_students": 0,
            "class_average": 0.0,
            "overall_pass_rate": 0.0,
            "at_risk_count": 0,
        }

    all_scores = df[SUBJECT_COLS].values.flatten()
    valid_scores = all_scores[~np.isnan(all_scores)]
    class_average = float(valid_scores.mean()) if len(valid_scores) else 0.0
    overall_pass_rate = (
        float((valid_scores >= pass_mark).sum() / len(valid_scores) * 100)
        if len(valid_scores)
        else 0.0
    )
    at_risk_count = len(detect_at_risk(df, threshold=pass_mark, min_failing=min_failing))
    return {
        "total_students": total_students,
        "class_average": class_average,
        "overall_pass_rate": overall_pass_rate,
        "at_risk_count": at_risk_count,
    }
