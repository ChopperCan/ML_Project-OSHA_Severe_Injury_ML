import pandas as pd
import os

# Tüm satırları ekranda göster
pd.set_option("display.max_rows", None)

def naics_counts():
    file_name = input("CSV dosya adını girin (örn: veri.csv): ").strip()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, file_name)

    if not os.path.isfile(file_path):
        print("❌ Dosya bulunamadı:", file_path)
        return

    df = pd.read_csv(file_path)

    if "Primary NAICS" not in df.columns:
        print("❌ 'Primary NAICS' kolonu bulunamadı.")
        return

    counts = (
        df["Primary NAICS"]
        .value_counts()
        .sort_index()
    )

    total_samples = counts.sum()

    print("\n📊 TÜM NAICS Kodları ve Örnek Sayıları")
    print("------------------------------------")
    print(counts.to_string())

    print("\n------------------------------------")
    print(f"TOPLAM ÖRNEK SAYISI: {total_samples}")


if __name__ == "__main__":
    naics_counts()
