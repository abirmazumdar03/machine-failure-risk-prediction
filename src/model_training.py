import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data_processing import RANDOM_STATE


def build_preprocessor(X):
    """Create preprocessing that is fitted only inside model pipelines."""
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_features = X.select_dtypes(include=["number"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numerical_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def build_models(preprocessor, random_state=RANDOM_STATE):
    """Return the simple model set required for the project."""
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    DecisionTreeClassifier(
                        class_weight="balanced",
                        max_depth=5,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    RandomForestClassifier(
                        class_weight="balanced",
                        n_estimators=200,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(model, X, y, threshold=0.5):
    """Calculate classification metrics at a chosen decision threshold."""
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return {
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(y, predictions, zero_division=0),
        "Recall": recall_score(y, predictions, zero_division=0),
        "F1": f1_score(y, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y, probabilities),
    }


def fit_and_evaluate_models(models, X_train, y_train, X_validation, y_validation):
    """Fit each model and return validation metrics side by side."""
    rows = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_validation, y_validation)
        rows.append({"Model": model_name, **metrics})

    return pd.DataFrame(rows)


def confusion_matrix_dataframe(model, X, y, threshold=0.5):
    """Return a labelled confusion matrix for easier notebook display."""
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y, predictions, labels=[0, 1])

    return pd.DataFrame(
        matrix,
        index=["Actual: No Failure", "Actual: Failure"],
        columns=["Predicted: No Failure", "Predicted: Failure"],
    )


def threshold_analysis(model, X, y, thresholds=(0.2, 0.3, 0.4, 0.5, 0.6)):
    """Show how precision, recall, and F1 move as the threshold changes."""
    rows = []
    probabilities = model.predict_proba(X)[:, 1]

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "Threshold": threshold,
                "Precision": precision_score(y, predictions, zero_division=0),
                "Recall": recall_score(y, predictions, zero_division=0),
                "F1": f1_score(y, predictions, zero_division=0),
            }
        )

    return pd.DataFrame(rows)


def get_feature_names(fitted_pipeline):
    """Read transformed feature names from a fitted pipeline."""
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    return preprocessor.get_feature_names_out()


def feature_importance_dataframe(fitted_pipeline):
    """Return feature importance or coefficient information for supported models."""
    model = fitted_pipeline.named_steps["model"]
    feature_names = get_feature_names(fitted_pipeline)

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        column_name = "Importance"
    elif hasattr(model, "coef_"):
        values = model.coef_[0]
        column_name = "Coefficient"
    else:
        raise ValueError("The selected model does not expose simple importance values.")

    importance = pd.DataFrame(
        {
            "Feature": feature_names,
            column_name: values,
            "Absolute Value": np.abs(values),
        }
    )

    return importance.sort_values("Absolute Value", ascending=False).reset_index(drop=True)
