# from unicodedata import numeric

import pandas as pd
import matplotlib.pyplot as plt
import uuid
import os

class AnalysisEngine:

    def __init__(self, dataframe):
        self.df = dataframe

    # Performs calculations on the dataframe.
    def aggregate_data(self, operation, group_by=None, value_column=None):
        if group_by:
            if operation == "count":
                return self.df.groupby(group_by).size()

            if value_column is None:
                return {"error": "A value column is required."}

            return (
                self.df
                .groupby(group_by)[value_column]
                .agg(operation)
            )

        if operation == "count":
            return len(self.df)

        if value_column is None:
            return {"error": "A value column is required."}

        return self.df[value_column].agg(operation)
    

    def execute(self, request):
        # Handle operation aliases by converting Enum -> string if necessary
        operation = (
            request.operation.value
            if hasattr(request.operation, "value")
            else request.operation
        )

        # Handle charts that do not require aggregation
        if request.chart_type in {"scatter", "histogram", "box"}:
            return self.generate_chart(
                chart_type=request.chart_type,
                x_column=request.x_column,
                y_column=request.y_column
            )
        
        if operation is None:
            return {"error": "No operation was provided."}
            
        allowed_operations = {
            "mean",
            "sum",
            "max",
            "min",
            "median",
            "count",
            "missing_values",
            "unique_values"
        }

        x_column = request.x_column
        y_column = request.y_column
        
        # Handle missing value analysis
        if operation == "missing_values":
            return {
                "result": self.df.isnull().sum().to_dict()
            }
        
        # Validate requested columns
        for column in [
            x_column,
            y_column,
            request.value_column,
            request.group_by
        ]:
            if column and column not in self.df.columns:
                return {
                    "error": f"Column '{column}' does not exist"}
        
        if operation == "unique_values":
            return {
                "result": self.df[x_column].unique().tolist()
            }
        
        # Simple dataset count
        if operation == "count" and not x_column and not y_column:
            return {"result": len(self.df)}
        
        if operation not in allowed_operations:
            return {
                "error": f"I don't know how to perform '{request.operation}' yet."
            }
        
        result = self.aggregate_data(
            operation=request.operation,
            group_by=request.group_by,
            value_column=request.value_column
        )

        if request.chart_type in {"bar", "line"}:
            return self.generateAggChart(
                chart_type=request.chart_type,
                data=result,
                group_by=request.group_by,
                value_column=request.value_column
            )

        if isinstance(result, dict):
            return result

        # Convert Pandas Series into a dictionary
        return {
            "result": result.to_dict() if hasattr(result, "to_dict") else result
        }

    """
    Creates visualizations.

    Bar and line charts use aggregated data.
    Scatter, histogram and box plots use
    the raw dataframe.
    """
    def generateAggChart(
        self,
        chart_type,
        data,
        group_by,
        value_column
    ):

        plt.figure(figsize=(10,6))

        if isinstance(data, dict):
            return data

        if not isinstance(data, pd.Series):
            return {
                "error": f"{chart_type.capitalize()} charts require multiple values."
            }

        data.plot(kind=chart_type)

        plt.xlabel(group_by)
        plt.ylabel(value_column)

        filename = f"{uuid.uuid4()}.png"
        filepath = os.path.join("charts", filename)

        os.makedirs("charts", exist_ok=True)

        plt.savefig(filepath)
        plt.close()

        return {
            "chart_path": filepath,
            "chart_type": chart_type,
            "x_column": group_by,
            "y_column": value_column
        }
        
            
    def generate_chart(
        self,
        chart_type,
        x_column,
        y_column
    ):
        # Validate request columns
        for column in [x_column, y_column]:
            if column and column not in self.df.columns:
                return {
                    "error": f"Column '{column}' does not exist"
                }

        plt.figure(figsize=(10, 6))

        if chart_type == "scatter":
            if not x_column or not y_column:
                return {
                    "error": "Scatter plots require both x_column and y_column."
                }

            if not pd.api.types.is_numeric_dtype(self.df[x_column]):
                return {
                    "error": "Scatter plots require numeric x_column."
                }

            if not pd.api.types.is_numeric_dtype(self.df[y_column]):
                return {
                    "error": "Scatter plots require numeric y_column."
                }

            self.df.plot(
                x=x_column,
                y=y_column,
                kind="scatter")

        elif chart_type == "histogram":
            if not x_column:
                return {
                    "error": "Histogram requires x_column."
                }
        
            if not pd.api.types.is_numeric_dtype(self.df[x_column]):
                return {
                    "error": "Histograms require numeric data."
                }
            
            self.df[x_column].plot(kind="hist")

        elif chart_type == "box":
            if not x_column:
                return {
                    "error": "Box plots require x_column."
                }
            if not pd.api.types.is_numeric_dtype(self.df[x_column]):
                return {
                    "error": "Box plots require numeric data."
                }
            self.df.boxplot(column=x_column)

        else:
            return {"error": f"Unsupported chart type: {chart_type}"}

        # final formatting
        plt.title(f"{chart_type.capitalize()} Chart")
        plt.xlabel(x_column)
        if y_column:
            plt.ylabel(y_column)

        # Save the plot to a file
        filename = f"{uuid.uuid4()}.png"
        filepath = os.path.join("charts", filename)
        os.makedirs("charts", exist_ok=True)
        plt.savefig(filepath)
        plt.close()

        return {
            "chart_path": filepath,
            "chart_type": chart_type,
            "x_column": x_column,
            "y_column": y_column
        }