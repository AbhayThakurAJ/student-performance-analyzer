"""Configurable grading module for student performance analysis."""

import math

import numpy as np
import pandas as pd


def get_default_boundaries() -> dict[str, int]:
    """Return the default grade boundaries.

    Returns:
        A dictionary mapping each letter grade to its minimum score threshold.
    """
    return {"A": 90, "B": 75, "C": 60, "D": 40, "F": 0}


def get_grade(score: float, boundaries: dict[str, int] | None = None) -> str:
    """Determine the letter grade for a given score.

    Args:
        score: A numeric score in the range 0–100.
        boundaries: Optional dictionary mapping grade letters to their minimum
            score thresholds.  If *None*, the default boundaries are used.

    Returns:
        The corresponding letter grade (``"A"``, ``"B"``, ``"C"``, ``"D"``,
        ``"F"``), or ``"N/A"`` if *score* is NaN.
    """
    # Handle NaN (works for both float('nan') and numpy NaN)
    if isinstance(score, float) and math.isnan(score):
        return "N/A"
    try:
        if np.isnan(score):
            return "N/A"
    except (TypeError, ValueError):
        pass

    if boundaries is None:
        boundaries = get_default_boundaries()

    # Sort grades by threshold descending so the first match wins.
    for grade, threshold in sorted(boundaries.items(), key=lambda x: x[1], reverse=True):
        if score >= threshold:
            return grade

    return "F"


def get_grade_color(grade: str) -> str:
    """Return a hex colour string associated with a letter grade.

    Args:
        grade: A letter grade (``"A"``–``"F"``) or ``"N/A"``.

    Returns:
        A hex colour code string.
    """
    colors: dict[str, str] = {
        "A": "#22C55E",
        "B": "#84CC16",
        "C": "#EAB308",
        "D": "#F97316",
        "F": "#EF4444",
        "N/A": "#94A3B8",
    }
    return colors.get(grade, "#94A3B8")


def assign_grades(
    df: pd.DataFrame,
    subject_cols: list[str],
    boundaries: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Add grade columns for each subject and an overall average.

    For every column name in *subject_cols*, a new ``<subject>_grade`` column is
    created containing the letter grade derived from the numeric score.  An
    ``average`` column (mean of all *subject_cols*) and its corresponding
    ``average_grade`` column are also appended.

    Args:
        df: DataFrame containing at least the columns listed in *subject_cols*.
        subject_cols: List of column names holding numeric scores.
        boundaries: Optional custom grade boundaries.

    Returns:
        A new DataFrame with the grade and average columns added.
    """
    result = df.copy()
    if result.empty:
        for col in subject_cols:
            result[f"{col}_grade"] = pd.Series(dtype=str)
        result["average"] = pd.Series(dtype=float)
        result["average_grade"] = pd.Series(dtype=str)
        return result

    for col in subject_cols:
        result[f"{col}_grade"] = result[col].apply(lambda s: get_grade(s, boundaries))

    result["average"] = result[subject_cols].mean(axis=1)
    result["average_grade"] = result["average"].apply(lambda s: get_grade(s, boundaries))

    return result


def get_pass_fail(score: float, pass_mark: int = 40) -> str:
    """Classify a score as *Pass* or *Fail*.

    Args:
        score: The numeric score to evaluate.
        pass_mark: The minimum score required to pass (default ``40``).

    Returns:
        ``"Pass"`` if *score* >= *pass_mark*, otherwise ``"Fail"``.
    """
    return "Pass" if score >= pass_mark else "Fail"


def grade_distribution(df: pd.DataFrame, subject: str) -> dict[str, int]:
    """Count the number of students in each grade for a subject.

    Args:
        df: DataFrame that contains a ``<subject>_grade`` column.
        subject: The subject name whose ``_grade`` column will be tallied.

    Returns:
        A dictionary mapping each grade letter to its student count.
    """
    grade_col = f"{subject}_grade"
    return df[grade_col].value_counts().to_dict()
