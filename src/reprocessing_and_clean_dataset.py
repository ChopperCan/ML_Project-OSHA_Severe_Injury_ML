import pandas as pd

INPUT_CSV = "data/January2015toApril2025.csv"
OUTPUT_CSV = "data/OSHA_3Digit_BaseData_cleaned.csv"

MIN_SAMPLES_NAICS = 20


def reprocess_osha_data(input_path=INPUT_CSV, output_path=OUTPUT_CSV):

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    df = pd.read_csv(input_path, quotechar='"', low_memory=False)

    # --------------------------------------------------
    # Text cleaning
    # --------------------------------------------------
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.replace(r"\n|\r", " ", regex=True)

    # --------------------------------------------------
    # Event-level target creation
    # --------------------------------------------------
    df["Permanent_Injury"] = (
        (df["Amputation"] > 0) |
        (df["Loss of Eye"] > 0)
    ).astype(int)

    df["Hospitalized"] = (
        (df["Permanent_Injury"] == 0) &
        (df["Hospitalized"] > 0)
    ).astype(int)

    # --------------------------------------------------
    # NAICS preprocessing (FINAL & CORRECT)
    # --------------------------------------------------
    df["Primary NAICS"] = (
        df["Primary NAICS"]
        .astype(str)
        .str.extract(r"(\d{3})")
        .astype(float)
        .astype("Int64")
    )

    # NAICS merge rules
    naics_merge_map = {
        521: 523,
        525: 523,
        519: 518
    }

    df["Primary NAICS"] = df["Primary NAICS"].replace(naics_merge_map)

    # Remove invalid NAICS
    df = df[df["Primary NAICS"].notna()]
    df = df[df["Primary NAICS"] != 999]

    # Remove rare NAICS AFTER merging
    naics_counts = df["Primary NAICS"].value_counts()
    rare_naics = naics_counts[naics_counts < MIN_SAMPLES_NAICS].index
    df = df[~df["Primary NAICS"].isin(rare_naics)]

    # --------------------------------------------------
    # Date features (used only for splitting)
    # --------------------------------------------------
    df["EventDate"] = pd.to_datetime(df["EventDate"], errors="coerce")
    df["Year"] = df["EventDate"].dt.year
    df["Month"] = df["EventDate"].dt.month

    # --------------------------------------------------
    # Final feature set
    # --------------------------------------------------
    selected_columns = [
        "Primary NAICS",
        "NatureTitle",
        "Part of Body Title",
        "EventTitle",
        "SourceTitle",
        "State",
        "Year",
        "Month",
        "Hospitalized",
        "Permanent_Injury"
    ]

    df = df[selected_columns].dropna()

    # --------------------------------------------------
    # Save processed dataset
    # --------------------------------------------------
    df.to_csv(output_path, index=False, encoding="utf-8")

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------
    print("Reprocessing completed successfully.")
    print("Input file :", input_path)
    print("Output file:", output_path)
    print("Final shape:", df.shape)
    print("NAICS distribution:")
    print(df["Primary NAICS"].value_counts().sort_index())
    print("Permanent injury rate:", df["Permanent_Injury"].mean())
    print("Hospitalized rate:", df["Hospitalized"].mean())


if __name__ == "__main__":
    reprocess_osha_data()
