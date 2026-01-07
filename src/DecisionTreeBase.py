import pandas as pd
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


# ===================== PATHS =====================

TRAIN_CSV = "data/TrainingSet_2015_2020.csv"
VAL_CSV   = "data/ValidationSet_2021_2022.csv"
TEST_CSV  = "data/TestSet_2023_2025.csv"

MODEL_PATH = "models/DecisionTree_Model.joblib"

VAL_PRED_CSV  = "data/DecisionTree_Validation_Predictions.csv"
TEST_PRED_CSV = "data/DecisionTree_Test_Predictions.csv"


# ===================== MAIN =====================

def main():
    df_train = pd.read_csv(TRAIN_CSV)
    df_val   = pd.read_csv(VAL_CSV)
    df_test  = pd.read_csv(TEST_CSV)

    TARGET = "Permanent_Injury"
    drop_columns = ["Permanent_Injury", "Hospitalized"]

    X_train = df_train.drop(columns=drop_columns)
    y_train = df_train[TARGET]

    X_val = df_val.drop(columns=drop_columns)
    y_val = df_val[TARGET]

    X_test = df_test.drop(columns=drop_columns)
    y_test = df_test[TARGET]

    categorical_features = [
        "Primary NAICS",
        "NatureTitle",
        "Part of Body Title",
        "EventTitle",
        "SourceTitle"
    ]

    numerical_features = []

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numerical", StandardScaler(), numerical_features)
        ]
    )

    decision_tree = DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=100,
        class_weight="balanced",
        random_state=42
    )

    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("classifier", decision_tree)
    ])

    pipeline.fit(X_train, y_train)

    def evaluate(X, y, name):
        y_pred = pipeline.predict(X)
        y_prob = pipeline.predict_proba(X)[:, 1]

        print(f"\n===== {name.upper()} RESULTS =====")
        print("Accuracy:", accuracy_score(y, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
        print("Classification Report:\n", classification_report(y, y_pred))
        print("ROC-AUC Score:", roc_auc_score(y, y_prob))

        return y_pred, y_prob

    val_pred, val_prob = evaluate(X_val, y_val, "Validation")
    df_val_out = df_val.copy()
    df_val_out["Predicted_Class"] = val_pred
    df_val_out["Risk_Score_Permanent_Injury"] = val_prob
    df_val_out.to_csv(VAL_PRED_CSV, index=False)

    test_pred, test_prob = evaluate(X_test, y_test, "Test")
    df_test_out = df_test.copy()
    df_test_out["Predicted_Class"] = test_pred
    df_test_out["Risk_Score_Permanent_Injury"] = test_prob
    df_test_out.to_csv(TEST_PRED_CSV, index=False)

    joblib.dump(pipeline, MODEL_PATH)

    print("\nDecision Tree model and outputs saved successfully.")


if __name__ == "__main__":
    main()
