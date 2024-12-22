from hotel_1_data_cleaner import optimize_dataframes
import pandas as pd

def main():
    # Bemeneti könyvtár megadása
    input_dir = '/config/workspace/verseny_dataklub_morgens/data/raw/hotel_1'

    # Adatok tisztítása és betöltése
    dfs = optimize_dataframes(input_dir)

    # Datepicker DataFrame kiemelése
    datepicker_df = dfs['datepicker_daily_visitors']

    # CPC tartalmú utm_source_and_medium értékek szűrése
    cpc_sources = datepicker_df[datepicker_df['utm_source_and_medium'].str.contains('cpc', case=False, na=False)]

    # Egyedi értékek kiírása
    unique_cpc_sources = sorted(cpc_sources['utm_source_and_medium'].unique())

    print("\nCPC-t tartalmazó utm_source_and_medium értékek:")
    print("=" * 50)
    for source in unique_cpc_sources:
        print(source)

    # Előfordulási gyakoriság
    print("\nElőfordulási gyakoriság:")
    print("=" * 50)
    print(cpc_sources['utm_source_and_medium'].value_counts())

if __name__ == "__main__":
    main()
