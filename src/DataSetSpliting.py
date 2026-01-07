import pandas as pd

DATA_PATH = "data/OSHA_3Digit_BaseData_cleaned.csv"

TRAIN_CSV = "data/TrainingSet_2015_2020.csv"
VAL_CSV   = "data/ValidationSet_2021_2022.csv"
TEST_CSV  = "data/TestSet_2023_2025.csv"

TRAIN_XLSX = "data/TrainingSet_2015_2020.xlsx"
VAL_XLSX   = "data/ValidationSet_2021_2022.xlsx"
TEST_XLSX  = "data/TestSet_2023_2025.xlsx"


def main(data_path=DATA_PATH):
    df = pd.read_csv(data_path)

    if "Year" not in df.columns:
        raise KeyError("Required column 'Year' not found in dataset")

    # --------------------------------------------------
    # Temporal splits (ORIGINAL)
    # --------------------------------------------------
    train_df = df[(df["Year"] >= 2015) & (df["Year"] <= 2020)]
    val_df   = df[(df["Year"] >= 2021) & (df["Year"] <= 2022)]
    test_df  = df[(df["Year"] >= 2023) & (df["Year"] <= 2025)]

    # --------------------------------------------------
    # Sanity checks
    # --------------------------------------------------
    print("\nYears included in each split:")
    print("TRAIN:", sorted(train_df["Year"].unique()))
    print("VALIDATION:", sorted(val_df["Year"].unique()))
    print("TEST:", sorted(test_df["Year"].unique()))

    print("\nSample counts:")
    print("Training:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))

    # --------------------------------------------------
    # Save outputs
    # --------------------------------------------------
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    train_df.to_excel(TRAIN_XLSX, index=False)
    val_df.to_excel(VAL_XLSX, index=False)
    test_df.to_excel(TEST_XLSX, index=False)

    print("\nTemporal split completed successfully.")
    print("Files saved in 'data/' directory.")


if __name__ == "__main__":
    main()
