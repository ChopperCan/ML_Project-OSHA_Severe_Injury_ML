import pandas as pd
import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# --------------------------------------------------
# Paths (relative – GitHub safe)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # src/
PROJECT_DIR = os.path.dirname(BASE_DIR)                 # OSHA_Project/
DATA_DIR = os.path.join(PROJECT_DIR, "data")

TRAIN_CSV = os.path.join(DATA_DIR, "TrainingSet_2015_2020.csv")
VAL_CSV   = os.path.join(DATA_DIR, "ValidationSet_2021_2022.csv")
TEST_CSV  = os.path.join(DATA_DIR, "TestSet_2023_2025.csv")

TARGET = "Permanent_Injury"
DROP_COLUMNS = ["Permanent_Injury", "Hospitalized"]


def main():

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    df_train = pd.read_csv(TRAIN_CSV)
    df_val   = pd.read_csv(VAL_CSV)
    df_test  = pd.read_csv(TEST_CSV)

    X_train = df_train.drop(columns=DROP_COLUMNS)
    y_train = df_train[TARGET]

    X_val = df_val.drop(columns=DROP_COLUMNS)
    y_val = df_val[TARGET]

    X_test = df_test.drop(columns=DROP_COLUMNS)
    y_test = df_test[TARGET]

    # --------------------------------------------------
    # Features (ORIGINAL)
    # --------------------------------------------------
    categorical_features = [
        "Primary NAICS",
        "NatureTitle",
        "Part of Body Title",
        "EventTitle",
        "SourceTitle"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    model = LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        class_weight="balanced"
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", model)
    ])

    # --------------------------------------------------
    # Train
    # --------------------------------------------------
    pipeline.fit(X_train, y_train)

    # --------------------------------------------------
    # Evaluation (ORIGINAL – threshold yok)
    # --------------------------------------------------
    def evaluate(X, y, name):
        y_pred = pipeline.predict(X)
        y_prob = pipeline.predict_proba(X)[:, 1]

        print(f"\n===== {name.upper()} RESULTS =====")
        print("Accuracy:", accuracy_score(y, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
        print("Classification Report:\n", classification_report(y, y_pred))
        print("ROC-AUC:", roc_auc_score(y, y_prob))

        return y_pred, y_prob

    val_pred, val_prob = evaluate(X_val, y_val, "Validation")
    test_pred, test_prob = evaluate(X_test, y_test, "Test")

    # --------------------------------------------------
    # Save outputs
    # --------------------------------------------------
    df_val["Predicted_Class"] = val_pred
    df_val["Risk_Score_Permanent_Injury"] = val_prob
    df_val.to_csv("ValidationSet_Predictions.csv", index=False)

    df_test["Predicted_Class"] = test_pred
    df_test["Risk_Score_Permanent_Injury"] = test_prob
    df_test.to_csv("TestSet_Predictions.csv", index=False)

    joblib.dump(pipeline, "LogisticRegression_Model.joblib")

    print("\nModel and prediction files saved successfully.")


if __name__ == "__main__":
    main()
