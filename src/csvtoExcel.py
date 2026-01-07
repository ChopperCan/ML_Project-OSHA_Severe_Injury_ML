import pandas as pd
import os

def csv_to_excel_same_folder():
    # Kullanıcıdan sadece dosya adı al
    csv_name = input("CSV dosya adını girin (örn: veri.csv): ").strip()

    # Programın bulunduğu klasör
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, csv_name)

    if not os.path.isfile(csv_path):
        print("❌ Dosya bulunamadı. Program ile aynı klasörde olmalı.")
        return

    if not csv_name.lower().endswith(".csv"):
        print("❌ Dosya uzantısı .csv olmalı.")
        return

    try:
        # CSV oku
        df = pd.read_csv(csv_path)

        # Excel dosya yolu
        excel_path = csv_path.replace(".csv", ".xlsx")

        # Excel'e yaz
        df.to_excel(excel_path, index=False)

        print(f"✅ Dönüştürme tamamlandı: {os.path.basename(excel_path)}")

    except Exception as e:
        print("❌ Hata oluştu:", e)


if __name__ == "__main__":
    csv_to_excel_same_folder()
