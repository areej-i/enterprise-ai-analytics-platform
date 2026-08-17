from enum import Enum
from typing import Literal

from pydantic import BaseModel

class Intent(str, Enum):
    analysis = "analysis"
    explanation = "explanation"
    machine_learning = "machine_learning"
    rag = "rag"

class MLTask(str, Enum):
    classification = "classification"
    regression = "regression"
    clustering = "clustering"

class ChartType(str, Enum):
    bar = "bar"
    line = "line"
    scatter = "scatter"
    histogram = "histogram"
    box = "box"

class Operation(str, Enum):
    mean = "mean"
    sum = "sum"
    max = "max"
    min = "min"
    median = "median"
    count = "count"
    missing_values = "missing_values"
    unique_values = "unique_values"

class ExecutionRequest(BaseModel):
    intent:Intent
    operation: Operation | None = None
    value_column: str | None = None
    group_by: str | None = None

    x_column: str | None = None
    y_column: str | None = None
    chart_type: ChartType | None = None

    task: MLTask | None = None  # classification, regression, clustering
    target_column: str | None = None  # e.g. "salary" or "churn"
    