from hotel_2_data_cleaner import optimize_dataframes
import pandas as pd

# Adatok betöltése
dataframes = optimize_dataframes('/config/workspace/verseny_dataklub_morgens/data/raw/hotel_2')

# Adatok kinyerése
search_log = dataframes['search_log']
website_users = dataframes['website_daily_users']
datepicker_users = dataframes['datepicker_daily_visitors']

# Összesített számok
total_visitors = website_users['user_count'].sum()
total_datepicker = datepicker_users['user_count'].sum()
total_searches = len(search_log)
total_bookings = len(search_log[search_log['conversion'] == 1])

print("\nTELJES FUNNEL:")
print("-" * 50)
print(f"1. Összes látogató: {total_visitors:,}")
print(f"2. Dátumválasztó használat: {total_datepicker:,}")
print(f"3. Összes keresés: {total_searches:,}")
print(f"4. Összes foglalás: {total_bookings:,}")

print("\nKONVERZIÓS RÁTÁK:")
print("-" * 50)
print(f"Látogató → Dátumválasztó: {(total_datepicker/total_visitors*100):.2f}%")
print(f"Dátumválasztó → Keresés: {(total_searches/total_datepicker*100):.2f}%")
print(f"Keresés → Foglalás: {(total_bookings/total_searches*100):.2f}%")
print(f"Teljes konverzió (Látogató → Foglalás): {(total_bookings/total_visitors*100):.2f}%")

# Devizánkénti bontás
print("\nDEVIZÁNKÉNTI BONTÁS:")
print("-" * 50)
for currency in search_log['currency'].unique():
    currency_searches = len(search_log[search_log['currency'] == currency])
    currency_bookings = len(search_log[(search_log['currency'] == currency) & (search_log['conversion'] == 1)])

    print(f"\n{currency}:")
    print(f"Keresések száma: {currency_searches:,}")
    print(f"Foglalások száma: {currency_bookings:,}")
    print(f"Konverziós ráta (keresés → foglalás): {(currency_bookings/currency_searches*100):.2f}%")
    print(f"Részesedés a keresésekből: {(currency_searches/total_searches*100):.2f}%")
    print(f"Részesedés a foglalásokból: {(currency_bookings/total_bookings*100):.2f}%")