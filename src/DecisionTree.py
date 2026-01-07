import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix


# ===================== PATHS =====================

TRAIN_CSV = "data/TrainingSet_2015_2020.csv"
VAL_CSV   = "data/ValidationSet_2021_2022.csv"
TEST_CSV  = "data/TestSet_2023_2025.csv"

HIGH_RISK_OUT = "data/high_risk_validation_cases.csv"
FN_OUT        = "data/missed_permanent_injuries.csv"


# ===================== MAIN =====================

def main():

    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    TARGET = "Permanent_Injury"

    categorical_features = [
        "Primary NAICS",
        "NatureTitle",
        "Part of Body Title",
        "EventTitle",
        "SourceTitle"
    ]

    # ONE-HOT ENCODING
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    X_train = encoder.fit_transform(train_df[categorical_features])
    X_val   = encoder.transform(val_df[categorical_features])
    X_test  = encoder.transform(test_df[categorical_features])

    feature_names = encoder.get_feature_names_out(categorical_features)

    y_train = train_df[TARGET]
    y_val   = val_df[TARGET]
    y_test  = test_df[TARGET]

    # LOGISTIC REGRESSION (RISK SCORING)
    log_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1
    )

    log_model.fit(X_train, y_train)

    # HIGH-RISK DEFINITION
    THRESHOLD = 0.60

    val_probs = log_model.predict_proba(X_val)[:, 1]
    val_df["logistic_risk"] = val_probs

    high_risk_df = val_df[val_df["logistic_risk"] >= THRESHOLD]

    print(f"\nHigh-risk validation cases: {len(high_risk_df)}")

    # DECISION TREE — EXPLAIN HIGH-RISK CASES
    X_hr = encoder.transform(high_risk_df[categorical_features])
    y_hr = high_risk_df[TARGET]

    tree_explainer = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42
    )

    tree_explainer.fit(X_hr, y_hr)

    rules = export_text(
        tree_explainer,
        feature_names=list(feature_names)
    )

    print("\n================ DECISION TREE RULES =================\n")
    print("Rules explaining HIGH-RISK cases (Logistic-based):\n")
    print(rules)

    # FALSE NEGATIVES — MISSED PERMANENT INJURIES
    test_probs = log_model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= THRESHOLD).astype(int)

    cm = confusion_matrix(y_test, test_preds)
    print("\nConfusion Matrix (Test Set):\n", cm)

    fn_mask = (y_test == 1) & (test_preds == 0)
    fn_df = test_df[fn_mask]

    print(f"\nFalse Negatives (missed permanent injuries): {len(fn_df)}")

    # DECISION TREE — FN PROFILE
    if len(fn_df) >= 50:
        X_fn = encoder.transform(fn_df[categorical_features])
        y_fn = fn_df[TARGET]

        fn_tree = DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=30,
            class_weight="balanced",
            random_state=42
        )

        fn_tree.fit(X_fn, y_fn)

        fn_rules = export_text(
            fn_tree,
            feature_names=list(feature_names)
        )

        print("\n=========== RULES FOR MISSED (FN) CASES ===========\n")
        print(fn_rules)
    else:
        print("\nNot enough False Negative cases for a reliable tree.")

    # SAVE OUTPUT FILES
    high_risk_df.to_csv(HIGH_RISK_OUT, index=False)
    fn_df.to_csv(FN_OUT, index=False)

    print("\nSaved output files:")
    print(f" - {HIGH_RISK_OUT}")
    print(f" - {FN_OUT}")


if __name__ == "__main__":
    main()
