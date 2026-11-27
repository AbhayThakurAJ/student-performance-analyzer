"""Data cleaning module for Student Performance Analyzer."""

import pandas as pd
import numpy as np

SUBJECT_COLS: list[str] = ["math", "science", "english", "history", "computer", "hindi"]
REQUIRED_COLS: list[str] = ["student_id", "name", "class"] + SUBJECT_COLS


def validate_columns(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Check that all required columns exist in the DataFrame.

    Args:
        df: Input DataFrame to validate.

    Returns:
        A tuple of (is_valid, missing_columns). Returns (True, []) if all
        required columns are present, or (False, list_of_missing) otherwise.
    """
    missing: list[str] = [col for col in REQUIRED_COLS if col not in df.columns]
    return (len(missing) == 0, missing)


def clean_marks(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Convert subject columns to numeric and replace invalid marks with NaN.

    Invalid marks are non-numeric non-blank values, negative values, or
    values greater than 100.

    Args:
        df: Input DataFrame with subject columns.

    Returns:
        A tuple of (cleaned_df, stats_dict) where stats_dict maps each subject
        column to the count of invalid values that were set to NaN.
    """
    df = df.copy()
    stats: dict[str, int] = {}

    for col in SUBJECT_COLS:
        if col not in df.columns:
            continue

        raw_series = df[col]
        is_non_blank = raw_series.notna() & (raw_series.astype(str).str.strip() != "")
        numeric_col = pd.to_numeric(raw_series, errors="coerce")

        non_numeric_mask = is_non_blank & numeric_col.isna()
        out_of_range_mask = (numeric_col < 0) | (numeric_col > 100)
        invalid_mask = non_numeric_mask | out_of_range_mask

        stats[col] = int(invalid_mask.sum())
        numeric_col.loc[out_of_range_mask] = np.nan
        df[col] = numeric_col

    return (df, stats)


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove duplicate rows based on student_id, keeping the first occurrence.

    Args:
        df: Input DataFrame with a student_id column.

    Returns:
        A tuple of (deduped_df, count_removed).
    """
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates(subset="student_id", keep="first")
    count_removed = before - len(df)
    return (df, count_removed)


def handle_missing_values(
    df: pd.DataFrame, strategy: str = "median"
) -> tuple[pd.DataFrame, dict]:
    """Fill or drop missing values in subject columns.

    Args:
        df: Input DataFrame with subject columns.
        strategy: Either "median" (fill NaN with column median) or "drop"
                  (drop rows with any NaN in subject columns).

    Returns:
        A tuple of (filled_df, stats_dict). When strategy is "median",
        stats_dict maps each column to the count of values filled. When
        strategy is "drop", stats_dict maps each column to the count of
        rows that had NaN in that column before dropping.

    Raises:
        ValueError: If strategy is not "median" or "drop".
    """
    df = df.copy()
    stats: dict[str, int] = {}

    if strategy == "median":
        for col in SUBJECT_COLS:
            if col not in df.columns:
                continue
            missing_count = int(df[col].isna().sum())
            stats[col] = missing_count
            if missing_count > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

    elif strategy == "drop":
        for col in SUBJECT_COLS:
            if col not in df.columns:
                continue
            stats[col] = int(df[col].isna().sum())
        df = df.dropna(subset=[c for c in SUBJECT_COLS if c in df.columns])

    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'median' or 'drop'.")

    return (df, stats)


def clean_data(
    df: pd.DataFrame, missing_strategy: str = "median"
) -> tuple[pd.DataFrame, dict]:
    """Orchestrate all cleaning steps in order.

    Steps:
        1. Validate columns
        2. Remove duplicate student_id rows
        3. Clean marks (coerce to numeric, invalidate out-of-range)
        4. Handle missing values

    Args:
        df: Raw input DataFrame.
        missing_strategy: Strategy for handling missing values ("median" or "drop").

    Returns:
        A tuple of (clean_df, full_report) where full_report contains stats
        from every cleaning step.

    Raises:
        ValueError: If required columns are missing from df.
    """
    report: dict = {}

    # Step 1 – validate columns
    is_valid, missing_cols = validate_columns(df)
    report["validation"] = {"is_valid": is_valid, "missing_columns": missing_cols}
    if not is_valid:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Step 2 – remove duplicates
    df, dup_count = remove_duplicates(df)
    report["duplicates_removed"] = dup_count

    # Step 3 – clean marks
    df, marks_stats = clean_marks(df)
    report["invalid_marks"] = marks_stats

    # Step 4 – handle missing values
    df, missing_stats = handle_missing_values(df, strategy=missing_strategy)
    report["missing_values"] = missing_stats

    return (df, report)
