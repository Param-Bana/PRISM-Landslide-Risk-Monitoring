
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "PRISM_RF_training_complete_environmental.csv"
MODEL_PATH = BASE_DIR / "prism_landslide_rf_pipeline_updated.joblib"
METRICS_PATH = BASE_DIR / "prism_model_metrics_updated.json"


NUMERIC_FEATURES = [
    "elevation_m",
    "slope_deg",
    "aspect_deg",
    "curvature",
    "rainfall_24h_mm",
    "rainfall_72h_mm",
    "rainfall_7d_mm",
    "ndvi",
    "distance_to_river_m",
]

CATEGORICAL_FEATURES = [
    "soil_type",
    "land_cover",
    "state",
]

TARGET = "landslide_label"


def build_model():
    """Build the exact Random Forest pipeline used by the prototype."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(
                strategy="median",
                add_indicator=True
            )),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found:\n{DATA_PATH}\n\n"
            "Keep PRISM_RF_training_complete_environmental.csv "
            "in the same folder as this script."
        )

    df = pd.read_csv(DATA_PATH, low_memory=False)

    required_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following required columns are missing:\n"
            + "\n".join(missing_columns)
        )

    # Clean target.
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].isin([0, 1])].copy()
    df[TARGET] = df[TARGET].astype(int)

    # Clean categorical values.
    for column in CATEGORICAL_FEATURES:
        df[column] = df[column].astype("string").str.strip()
        df.loc[
            df[column].isin(["", "nan", "None", "NaN"]),
            column
        ] = pd.NA

    # The supplied training file already contains rows with complete
    # environmental predictors. Do not manufacture environmental values.
    df = df.dropna(
        subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES
    ).copy()

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    print("=" * 70)
    print("PRISM RANDOM FOREST TRAINING")
    print("=" * 70)
    print(f"Training rows : {len(df)}")
    print(f"Negative (0)  : {(y == 0).sum()}")
    print(f"Positive (1)  : {(y == 1).sum()}")
    print()
    print("Categorical features:")
    print("  - soil_type")
    print("  - land_cover")
    print("  - state")
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )

    model = build_model()

    print("Training Random Forest...")
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    metrics = {
        "algorithm": "Random Forest Classifier",
        "n_estimators": 500,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": 42,
        "training_rows": int(len(df)),
        "test_rows": int(len(X_test)),
        "positive_rows": int(y.sum()),
        "negative_rows": int((y == 0).sum()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
        "brier_score": float(
            brier_score_loss(y_test, probabilities)
        ),
        "test_split": 0.25,
        "test_random_state": 42,
        "warning": (
            "Prototype holdout metrics only. Final PRISM validation "
            "should use spatial and temporal validation."
        ),
    }

    joblib.dump(model, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("MODEL PERFORMANCE")
    print("-" * 70)
    print(f"Accuracy           : {metrics['accuracy']:.4f}")
    print(f"Precision          : {metrics['precision']:.4f}")
    print(f"Recall             : {metrics['recall']:.4f}")
    print(f"F1 Score           : {metrics['f1']:.4f}")
    print(f"ROC-AUC            : {metrics['roc_auc']:.4f}")
    print(f"Average Precision  : {metrics['average_precision']:.4f}")
    print(f"Brier Score        : {metrics['brier_score']:.4f}")
    print()

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print()

    print("Classification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["No Landslide", "Landslide"],
        zero_division=0,
    ))

    print(f"Saved model  : {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
