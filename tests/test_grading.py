"""Unit tests for src/grading.py."""

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.grading import (
    assign_grades,
    get_default_boundaries,
    get_grade,
    get_grade_color,
    get_pass_fail,
    grade_distribution,
)


def test_default_boundaries() -> None:
    assert get_default_boundaries() == {"A": 90, "B": 75, "C": 60, "D": 40, "F": 0}


def test_grade_A() -> None:
    assert get_grade(95) == "A"


def test_grade_B() -> None:
    assert get_grade(80) == "B"


def test_grade_C() -> None:
    assert get_grade(65) == "C"


def test_grade_D() -> None:
    assert get_grade(45) == "D"


def test_grade_F() -> None:
    assert get_grade(30) == "F"


@pytest.mark.parametrize(
    "score,expected",
    [
        (90, "A"),
        (75, "B"),
        (60, "C"),
        (40, "D"),
        (0, "F"),
    ],
)
def test_grade_boundary_exact(score: float, expected: str) -> None:
    assert get_grade(score) == expected


def test_grade_nan() -> None:
    assert get_grade(float("nan")) == "N/A"
    assert get_grade(np.nan) == "N/A"
    assert math.isnan(float("nan"))


def test_custom_boundaries() -> None:
    custom = {"A": 85, "B": 70, "C": 55, "D": 35, "F": 0}
    assert get_grade(86, custom) == "A"
    assert get_grade(72, custom) == "B"
    assert get_grade(55, custom) == "C"
    assert get_grade(36, custom) == "D"
    assert get_grade(30, custom) == "F"


def test_grade_color() -> None:
    assert get_grade_color("A") == "#22C55E"
    assert get_grade_color("B") == "#84CC16"
    assert get_grade_color("C") == "#EAB308"
    assert get_grade_color("D") == "#F97316"
    assert get_grade_color("F") == "#EF4444"
    assert get_grade_color("N/A") == "#94A3B8"


def test_assign_grades() -> None:
    df = pd.DataFrame(
        {
            "student_id": ["S001", "S002"],
            "name": ["Aarav", "Diya"],
            "math": [92, 35],
            "science": [80, 60],
        }
    )
    graded = assign_grades(df, ["math", "science"])
    assert "math_grade" in graded.columns
    assert "science_grade" in graded.columns
    assert "average" in graded.columns
    assert "average_grade" in graded.columns
    assert graded.loc[0, "math_grade"] == "A"
    assert graded.loc[1, "math_grade"] == "F"
    assert graded.loc[0, "average"] == pytest.approx(86.0)
    assert graded.loc[0, "average_grade"] == "B"


def test_pass_fail() -> None:
    assert get_pass_fail(40, pass_mark=40) == "Pass"
    assert get_pass_fail(39.9, pass_mark=40) == "Fail"
    assert get_pass_fail(50, pass_mark=55) == "Fail"
    assert get_pass_fail(60, pass_mark=55) == "Pass"


def test_grade_distribution() -> None:
    df = pd.DataFrame(
        {
            "math": [95, 91, 80, 65, 30],
        }
    )
    graded = assign_grades(df, ["math"])
    dist = grade_distribution(graded, "math")
    assert dist == {"A": 2, "B": 1, "C": 1, "F": 1}
