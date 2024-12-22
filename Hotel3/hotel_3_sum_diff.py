import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from hotel_3_data_cleaner import optimize_dataframes

def create_diff_plot(booking_data, search_log, currency, save_name):
    """Eltérések diagram készítése egy adott devizára"""
    plt.figure(figsize=(12, 6))

    # Összekötjük a booking_data és search_log adatokat
    merged_data = pd.merge(
        booking_data,
        search_log[['id', 'utc_datetime', 'currency']],
        left_on='search_log_id',
        right_on='id',
        how='left'
    )

    # Szűrés devizára
    currency_data = merged_data[merged_data['currency'] == currency]

    # Külön színek a pozitív és negatív eltéréseknek
    positive_diff = currency_data[currency_data['diff'] > 0]
    negative_diff = currency_data[currency_data['diff'] < 0]

    # Pozitív eltérések (kék)
    plt.scatter(pd.to_datetime(positive_diff['utc_datetime']), 
               positive_diff['diff'],
               alpha=0.6, color='blue', label='Többlet')

    # Negatív eltérések (piros)
    plt.scatter(pd.to_datetime(negative_diff['utc_datetime']), 
               negative_diff['diff'],
               alpha=0.6, color='red', label='Hiány')

    plt.title(f'HOTEL 3 - Eltérések időbeli eloszlása ({currency})', fontsize=14, pad=20)
    plt.xlabel('Dátum', fontsize=12)
    plt.ylabel(f'Eltérés ({currency})', fontsize=12)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.4, linestyle='--')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f'elteresek_idosora_{save_name}.png')
    plt.show()

def main():
    # Adatok betöltése
    input_dir = '/config/workspace/verseny_dataklub_morgens/data/raw/hotel_3'
    dfs = optimize_dataframes(input_dir)
    booking_data = dfs['booking_data']
    search_log = dfs['search_log']

    # Számítások elvégzése
    booking_data['sum'] = (
        booking_data['rooms_total_price'] +
        booking_data['upsell_total_price'] -
        booking_data['vouchers_total_price'] -
        booking_data['loyalty_discount_total'] -
        booking_data['redeemed_loyalty_points_total']
    )
    booking_data['diff'] = booking_data['sum'] - booking_data['total_price_final']
    booking_data['abs_diff'] = abs(booking_data['diff'])

    # Összekötjük az adatokat
    merged_data = pd.merge(
        booking_data,
        search_log[['id', 'utc_datetime', 'currency']],
        left_on='search_log_id',
        right_on='id',
        how='left'
    )

    # Devizánkénti elemzés
    for currency in search_log['currency'].unique():
        currency_data = merged_data[merged_data['currency'] == currency]
        diff_count = (currency_data['diff'] != 0).sum()

        print(f"\n{'='*50}")
        print(f"\nDeviza: {currency}")
        print(f"Eltérések száma: {diff_count}")

        if diff_count > 0:
            print("\nTop 5 legnagyobb abszolút eltérés:")
            top_5 = currency_data.nlargest(5, 'abs_diff')[
                ['search_log_id', 'sum', 'total_price_final', 'diff', 'currency']
            ]
            print(top_5)

            create_diff_plot(booking_data, search_log, currency, currency.lower())

    # Foglalási statisztikák
    print("\n" + "="*50)
    print("\nFoglalási statisztikák:")

    # Foglalások száma és konverziós ráta
    total_bookings = search_log['conversion'].sum()
    total_searches = len(search_log)

    print(f"\nÖsszes keresés száma: {total_searches:,}")
    print(f"Összes foglalás száma: {int(total_bookings):,}")
    print(f"Konverziós ráta: {(total_bookings/total_searches)*100:.2f}%")

    # Foglalások devizánkénti bontásban
    bookings_by_currency = search_log[search_log['conversion'] == 1]['currency'].value_counts()
    print("\nFoglalások devizánként:")
    print(bookings_by_currency)

    # Foglalások nyelvi bontásban
    bookings_by_language = search_log[search_log['conversion'] == 1]['lang_code'].value_counts()
    print("\nFoglalások nyelvenként:")
    print(bookings_by_language)

    # Átlagos foglalási érték devizánként
    avg_booking_value = search_log[search_log['conversion'] == 1].groupby('currency')['total_price_final'].mean()
    print("\nÁtlagos foglalási érték devizánként:")
    print(avg_booking_value)

if __name__ == "__main__":
    main()