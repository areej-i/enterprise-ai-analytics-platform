import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
    silhouette_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib
from datetime import datetime
import os

class MachineLearningEngine:

    def __init__(self, dataframe):
        self.df = dataframe

    def execute(self, request):

        allowed_tasks = {
            "classification",
            "regression",
            "clustering"
        }

        task = request.task.lower()

        if task not in allowed_tasks:
            return {
                "error": f"Unsupported machine learning task: {task}"
            }

        if task == "classification":
            return self.classification(request)

        elif task == "regression":
            return self.regression(request)

        elif task == "clustering":
            return self.clustering(request)

    def classification(self, request):

        target = request.target_column

        if target not in self.df.columns:
            return {"error": f"Target column '{target}' does not exist."}

        X = self.df.drop(columns=[target])
        y = self.df[target]

        categorical = X.select_dtypes(include=["object", "category"]).columns
        numeric = X.select_dtypes(include=["number"]).columns

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median"))
                    ]),
                    numeric,
                ),
                (
                    "cat",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore"))
                    ]),
                    categorical,
                ),
            ]
        )

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42))
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        model.fit(X_train, y_train)

        # Save model
        model_name = f"models/{target}_classifier.pkl"
        joblib.dump(model, model_name)

        # Get trained classifier
        classifier = model.named_steps["classifier"]

        predictions = model.predict(X_test)

        # Get feature names after preprocessing
        feature_names = model.named_steps[
            "preprocessor"
        ].get_feature_names_out()

        # Match feature names with importance scores
        importance = dict(
            sorted(
                zip(
                    feature_names,
                    classifier.feature_importances_
                ),
                key=lambda x: x[1],
                reverse=True
            )
        )

        # Keep top 10 features
        top_features = dict(
            list(importance.items())[:10]
        )

        # Clean the features
        clean_features = {}

        for feature, score in top_features.items():
            clean_feature = (
                feature
                .replace("num__", "")
                .replace("cat__", "")
            )

            clean_features[clean_feature] = float(score)

        return {
            "task": "classification",
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            ),
            "recall": recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            ),
            "f1": f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            ),
            "model_path": model_name,
            "feature_importance": clean_features
        }

    def regression(self, request):

        target = request.target_column

        if target not in self.df.columns:
            return {"error": f"Target column '{target}' does not exist."}

        X = self.df.drop(columns=[target])
        y = self.df[target]

        categorical = X.select_dtypes(include=["object", "category"]).columns
        numeric = X.select_dtypes(include=["number"]).columns

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median"))
                    ]),
                    numeric,
                ),
                (
                    "cat",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore"))
                    ]),
                    categorical,
                ),
            ]
        )

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(random_state=42))
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        model.fit(X_train, y_train)

        model_name = f"models/{target}_regressor.pkl"
        joblib.dump(model, model_name)

        predictions = model.predict(X_test)

        # Feature importance
        regressor = model.named_steps["regressor"]

        feature_names = model.named_steps[
            "preprocessor"
        ].get_feature_names_out()

        importance = dict(
            sorted(
                zip(
                    feature_names,
                    regressor.feature_importances_
                ),
                key=lambda x: x[1],
                reverse=True
            )
        )

        top_features = dict(
            list(importance.items())[:10]
        )

                # Clean the features
        clean_features = {}

        for feature, score in top_features.items():
            clean_feature = (
                feature
                .replace("num__", "")
                .replace("cat__", "")
            )

            clean_features[clean_feature] = float(score)

        return {
            "task": "regression",
            "mae": mean_absolute_error(y_test, predictions),
            "rmse": root_mean_squared_error(
                y_test,
                predictions,
            ),
            "r2": r2_score(y_test, predictions),
            "model_path": model_name,
            "feature_importance": clean_features
        }

    def clustering(self, request):
        X = self.df.copy()

        categorical = X.select_dtypes(include=["object", "category"]).columns
        numeric = X.select_dtypes(include=["number"]).columns

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median"))
                    ]),
                    numeric,
                ),
                (
                    "cat",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore"))
                    ]),
                    categorical,
                ),
            ]
        )

        X_processed = preprocessor.fit_transform(X)

        model = KMeans(
            n_clusters=3,
            random_state=42,
            n_init="auto",
        )

        labels = model.fit_predict(X_processed)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"models/kmeans_{timestamp}.pkl"

        joblib.dump(model, model_name)

        score = silhouette_score(
            X_processed,
            labels,
        )

        return {
            "task": "clustering",
            "clusters": int(len(set(labels))),
            "silhouette_score": score,
            "model_path": model_name
        }