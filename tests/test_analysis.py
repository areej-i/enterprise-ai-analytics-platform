# `test_analysis.py`
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)

import pandas as pd
from analysis_engine import AnalysisEngine
from schemas import ExecutionRequest, Intent, Operation

def create_test_dataframe():
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "David"],
        "salary": [50000, 60000, 70000, 80000],
        "department": ["IT", "HR", "IT", "HR"],
    })


def test_mean():
    df = create_test_dataframe()
    engine = AnalysisEngine(df)

    request = ExecutionRequest(
        intent=Intent.analysis,
        operation=Operation.mean,
        value_column="salary"
    )

    result = engine.execute(request)

    assert result["result"] == 65000


def test_sum():
    df = create_test_dataframe()
    engine = AnalysisEngine(df)

    request = ExecutionRequest(
        intent=Intent.analysis,
        operation=Operation.sum,
        value_column="salary"
    )

    result = engine.execute(request)

    assert result["result"] == 260000


def test_max():
    df = create_test_dataframe()
    engine = AnalysisEngine(df)

    request = ExecutionRequest(
        intent=Intent.analysis,
        operation=Operation.max,
        value_column="salary"
    )

    result = engine.execute(request)

    assert result["result"] == 80000


def test_min():
    df = create_test_dataframe()
    engine = AnalysisEngine(df)

    request = ExecutionRequest(
        intent=Intent.analysis,
        operation=Operation.min,
        value_column="salary"
    )

    result = engine.execute(request)

    assert result["result"] == 50000


def test_count():
    df = create_test_dataframe()
    engine = AnalysisEngine(df)

    request = ExecutionRequest(
        intent=Intent.analysis,
        operation=Operation.count
    )

    result = engine.execute(request)

    assert result["result"] == 4


def test_invalid_column():
    df = create_test_dataframe()
    engine = AnalysisEngine(df)

    request = ExecutionRequest(
        intent=Intent.analysis,
        operation=Operation.mean,
        value_column="age"
    )

    result = engine.execute(request)

    assert "error" in result

