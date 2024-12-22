import pandas as pd
import os
from hotel_3_data_cleaner import optimize_dataframes

def main():
    # Bemeneti könyvtár megadása
    input_dir = '/config/workspace/verseny_dataklub_morgens/data/raw/hotel_3'

    # Adatok tisztítása és betöltése
    dfs = optimize_dataframes(input_dir)

    # DataFramek külön változókba mentése és információk kiírása
    dataframes = {
        'booking_data': dfs['booking_data'],
        'daily_occupancy': dfs['daily_occupancy'],
        'daily_ppc_budget': dfs['daily_ppc_budget'],
        'datepicker_daily_visitors': dfs['datepicker_daily_visitors'],
        'search_log': dfs['search_log'],
        'search_log_room_child': dfs['search_log_room_child'],
        'search_log_room': dfs['search_log_room'],
        'search_log_room_offer': dfs['search_log_room_offer'],
        'search_log_session': dfs['search_log_session'],
        'upsell_data': dfs['upsell_data'],
        'website_daily_users': dfs['website_daily_users']
    }

    # Minden DataFrame információinak kiírása
    for name, df in dataframes.items():
        print(f"\n{'='*50}")
        print(f"{name} oszloptípusok:")
        print(df.dtypes)
        print(f"\n{name} első 5 sora:")
        print(df.head())
        print(f"Sorok száma: {len(df)}")
        print('='*50)

    # Alapvető statisztikák
    print("\nAlapvető statisztikák:")
    print(f"Booking adatok száma: {len(dataframes['booking_data'])}")
    print(f"Napi foglaltság átlaga: {dataframes['daily_occupancy']['fill_rate'].mean():.2f}")

    # További statisztikák
    print("\nTovábbi statisztikák:")
    print(f"Összes keresés száma: {len(dataframes['search_log'])}")
    print(f"Átlagos tartózkodási idő (éjszaka): {dataframes['search_log']['nights'].mean():.2f}")
    print(f"Átlagos felnőttek száma/foglalás: {dataframes['search_log']['adults'].mean():.2f}")

if __name__ == "__main__":
    main()
