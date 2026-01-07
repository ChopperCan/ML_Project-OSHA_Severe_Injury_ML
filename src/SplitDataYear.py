import pandas as pd

DATA_PATH = "data/OSHA_3Digit_BaseData_cleaned.csv"
OUTPUT_CSV = "data/yearly_sample_counts.csv"
OUTPUT_XLSX = "data/yearly_sample_counts.xlsx"


def main(data_path=DATA_PATH):
    df = pd.read_csv(data_path)

    if "Year" not in df.columns:
        raise KeyError("Year column not found in dataset")

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    yearly_counts = (
        df["Year"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    yearly_counts.columns = ["Year", "SampleCount"]

    yearly_counts.to_csv(OUTPUT_CSV, index=False)
    yearly_counts.to_excel(OUTPUT_XLSX, index=False)

    print("Yearly sample counts created successfully.")
    print(yearly_counts)
    print(f"Output files: {OUTPUT_CSV}, {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
