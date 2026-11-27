"""Unit tests for src/cleaning.py."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cleaning import (
    REQUIRED_COLS,
    SUBJECT_COLS,
    clean_data,
    clean_marks,
    handle_missing_values,
    remove_duplicates,
    validate_columns,
)


@pytest.fixture
def valid_df() -> pd.DataFrame:
    """Return a small valid DataFrame matching the expected schema."""
    return pd.DataFrame(
        {
            "student_id": ["S001", "S002", "S003"],
            "name": ["Aarav", "Diya", "Kabir"],
            "class": ["10-A", "10-B", "10-A"],
            "math": [85, 70, 90],
            "science": [78, 82, 88],
            "english": [92, 65, 75],
            "history": [74, 80, 70],
            "computer": [95, 88, 91],
            "hindi": [80, 76, 84],
        }
    )


def test_validate_columns_valid(valid_df: pd.DataFrame) -> None:
    is_valid, missing = validate_columns(valid_df)
    assert is_valid is True
    assert missing == []


def test_validate_columns_missing(valid_df: pd.DataFrame) -> None:
    incomplete = valid_df.drop(columns=["math", "hindi"])
    is_valid, missing = validate_columns(incomplete)
    assert is_valid is False
    assert set(missing) == {"math", "hindi"}


def test_clean_marks_invalid_negative(valid_df: pd.DataFrame) -> None:
    df = valid_df.copy()
    df.loc[0, "math"] = -10
    cleaned, stats = clean_marks(df)
    assert np.isnan(cleaned.loc[0, "math"])
    assert stats["math"] == 1


def test_clean_marks_invalid_above_100(valid_df: pd.DataFrame) -> None:
    df = valid_df.copy()
    df.loc[1, "science"] = 105
    cleaned, stats = clean_marks(df)
    assert np.isnan(cleaned.loc[1, "science"])
    assert stats["science"] == 1


def test_clean_marks_non_numeric(valid_df: pd.DataFrame) -> None:
    df = valid_df.copy()
    df["english"] = df["english"].astype(object)
    df.loc[2, "english"] = "absent"
    cleaned, stats = clean_marks(df)
    assert np.isnan(cleaned.loc[2, "english"])
    assert stats["english"] == 1


def test_remove_duplicates(valid_df: pd.DataFrame) -> None:
    df_with_dups = pd.concat([valid_df, valid_df.iloc[[0, 1]]], ignore_index=True)
    deduped, count = remove_duplicates(df_with_dups)
    assert count == 2
    assert len(deduped) == 3
    assert deduped["student_id"].tolist() == ["S001", "S002", "S003"]


def test_handle_missing_median(valid_df: pd.DataFrame) -> None:
    df = valid_df.copy()
    df.loc[1, "math"] = np.nan  # remaining values: 85, 90 -> median 87.5
    filled, stats = handle_missing_values(df, strategy="median")
    assert stats["math"] == 1
    assert filled.loc[1, "math"] == pytest.approx(87.5)
    assert not filled[SUBJECT_COLS].isna().any().any()


def test_handle_missing_drop(valid_df: pd.DataFrame) -> None:
    df = valid_df.copy()
    df.loc[1, "history"] = np.nan
    dropped, stats = handle_missing_values(df, strategy="drop")
    assert stats["history"] == 1
    assert len(dropped) == 2
    assert "S002" not in dropped["student_id"].values


def test_handle_missing_invalid_strategy(valid_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        handle_missing_values(valid_df, strategy="mean")


def test_clean_data_full_pipeline(valid_df: pd.DataFrame) -> None:
    df = valid_df.copy()
    df["math"] = df["math"].astype(object)
    df.loc[0, "math"] = -5
    df.loc[1, "science"] = 120
    df.loc[2, "hindi"] = np.nan
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    cleaned, report = clean_data(df, missing_strategy="median")
    assert report["validation"]["is_valid"] is True
    assert report["duplicates_removed"] == 1
    assert report["invalid_marks"]["math"] == 1
    assert report["invalid_marks"]["science"] == 1
    assert len(cleaned) == 3
    assert not cleaned[SUBJECT_COLS].isna().any().any()


def test_empty_dataframe() -> None:
    empty_df = pd.DataFrame(columns=REQUIRED_COLS)
    cleaned, report = clean_data(empty_df, missing_strategy="median")
    assert cleaned.empty
    assert report["duplicates_removed"] == 0


def test_single_student(valid_df: pd.DataFrame) -> None:
    single = valid_df.iloc[[0]].copy()
    cleaned, report = clean_data(single, missing_strategy="median")
    assert len(cleaned) == 1
    assert report["duplicates_removed"] == 0
    assert cleaned.iloc[0]["math"] == 85
