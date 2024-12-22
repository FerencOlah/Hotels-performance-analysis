import pandas as pd
import os
from hotel_1_data_cleaner import optimize_dataframes

def analyze_hotel_bookings(input_dir):
    """
    Komplex elemzés a hotel foglalásokról:
    1. Látogatások száma foglalás előtt
    2. Többszörös foglalók részletes elemzése

    Args:
        input_dir (str): Az input adatok könyvtára
    """
    # Adatok betöltése
    data = optimize_dataframes(input_dir)

    # Szükséges DataFramek kinyerése
    search_log = data['search_log']
    search_log_session = data['search_log_session']

    # Összekötjük a search_log és search_log_session táblákat
    merged_df = pd.merge(
        search_log,
        search_log_session,
        left_on='search_log_session_id',
        right_on='id',
        suffixes=('_search', '_session')
    )

    # 1. RÉSZ: LÁTOGATÁSOK SZÁMA FOGLALÁS ELŐTT
    print("\n=== LÁTOGATÁSOK ELEMZÉSE FOGLALÁS ELŐTT ===")

    # Rendezzük időrend szerint
    merged_df = merged_df.sort_values(['uuid', 'utc_datetime'])

    # Csak azokat a felhasználókat nézzük, akik végül foglaltak
    converted_users = merged_df[merged_df['conversion'] == 1]['uuid'].unique()

    # Számoljuk meg látogatásokat a konverzióig
    visits_before_booking = []

    for user in converted_users:
        user_sessions = merged_df[merged_df['uuid'] == user]
        conversion_session = user_sessions[user_sessions['conversion'] == 1].iloc[0]
        previous_visits = len(user_sessions[
            (user_sessions['utc_datetime'] < conversion_session['utc_datetime'])
        ])
        visits_before_booking.append({
            'uuid': user,
            'visits_before_booking': previous_visits
        })

    results_df = pd.DataFrame(visits_before_booking)

    # Alapstatisztikák kiírása
    stats = {
        'Átlagos látogatások száma': results_df['visits_before_booking'].mean(),
        'Medián látogatások száma': results_df['visits_before_booking'].median(),
        'Minimum látogatások száma': results_df['visits_before_booking'].min(),
        'Maximum látogatások száma': results_df['visits_before_booking'].max(),
        'Foglalók száma': len(converted_users),
        'Összes foglalás': len(merged_df[merged_df['conversion'] == 1])
    }

    print("\nÖsszes látogató statisztika:")
    print("-" * 40)
    for key, value in stats.items():
        print(f"{key}: {value:.1f}")

    # Devizánkénti statisztikák
    for currency in merged_df['currency'].unique():
        currency_conversions = merged_df[
            (merged_df['conversion'] == 1) & 
            (merged_df['currency'] == currency)
        ]

        currency_users = currency_conversions['uuid'].unique()
        currency_results = results_df[results_df['uuid'].isin(currency_users)]

        currency_stats = {
            'Átlagos látogatások száma': currency_results['visits_before_booking'].mean(),
            'Medián látogatások száma': currency_results['visits_before_booking'].median(),
            'Minimum látogatások száma': currency_results['visits_before_booking'].min(),
            'Maximum látogatások száma': currency_results['visits_before_booking'].max(),
            'Foglalók száma': len(currency_users),
            'Összes foglalás': len(currency_conversions),
            'Átlagos foglalási érték': currency_conversions['total_price_final'].mean(),
            'Összes bevétel': currency_conversions['total_price_final'].sum(),
            'Összes felnőtt': currency_conversions['adults'].sum(),
            'Összes gyerek': currency_conversions['children'].sum()
        }

        print(f"\n{currency} foglalások statisztikái:")
        print("-" * 40)
        for key, value in currency_stats.items():
            if key in ['Átlagos foglalási érték', 'Összes bevétel']:
                print(f"{key}: {value:,.0f} {currency}")
            else:
                print(f"{key}: {value:.1f}")

        # Látogatások eloszlása devizánként
        currency_distribution = currency_results['visits_before_booking'].value_counts().sort_index()
        print(f"\n{currency} látogatások eloszlása:")
        for visits, count in currency_distribution.items():
            percentage = (count/len(currency_results))*100
            print(f"{visits} látogatás: {count} felhasználó ({percentage:.1f}%)")

    # 2. RÉSZ: TÖBBSZÖRÖS FOGLALÓK ELEMZÉSE
    print("\n=== TÖBBSZÖRÖS FOGLALÓK ELEMZÉSE ===")

    # Csak a konvertált (foglalt) felhasználókat nézzük
    bookings = merged_df[merged_df['conversion'] == 1]

    # Számoljuk meg felhasználónként a foglalások számát
    booking_counts = bookings.groupby('uuid').size().reset_index(name='booking_count')
    repeat_bookers = booking_counts[booking_counts['booking_count'] > 1]

    # Nézzük meg a többszörös foglalók forrásait
    repeat_booker_details = pd.merge(
        bookings[bookings['uuid'].isin(repeat_bookers['uuid'])],
        repeat_bookers,
        on='uuid'
    )

    # Források és devizák szerinti elemzés
    source_currency_analysis = repeat_booker_details.groupby(
        ['utm_source', 'utm_medium', 'currency'],
        observed=True
    ).agg({
        'uuid': ['count', 'nunique'],
        'total_price_final': ['sum', 'mean'],
        'adults': 'sum',
        'children': 'sum'
    }).reset_index()

    # Oszlopnevek egyszerűsítése
    source_currency_analysis.columns = ['Forrás', 'Médium', 'Deviza', 'Összes_foglalás', 
                                      'Egyedi_foglalók', 'Teljes_bevétel', 
                                      'Átlagos_foglalási_érték', 'Összes_felnőtt', 
                                      'Összes_gyerek']

    print("\nForrások és devizák szerinti részletes elemzés:")
    print("-" * 40)
    print(f"Összes visszatérő foglaló: {len(repeat_bookers)} fő")

    # Devizánkénti statisztikák
    for currency in source_currency_analysis['Deviza'].unique():
        currency_data = source_currency_analysis[source_currency_analysis['Deviza'] == currency]
        if currency_data['Összes_foglalás'].sum() > 0:  # Csak akkor írjuk ki, ha van foglalás
            print(f"\nDeviza: {currency}")
            print(f"Összes foglalás: {currency_data['Összes_foglalás'].sum():.0f} db")
            print(f"Teljes bevétel: {currency_data['Teljes_bevétel'].sum():,.0f} {currency}")
            összes_vendég = currency_data['Összes_felnőtt'].sum() + currency_data['Összes_gyerek'].sum()
            print(f"Összes vendég: {összes_vendég:.0f} fő")
            print(f"  - Ebből felnőtt: {currency_data['Összes_felnőtt'].sum():.0f} fő")
            print(f"  - Ebből gyerek: {currency_data['Összes_gyerek'].sum():.0f} fő")

    # Foglalási gyakoriság elemzése
    booking_frequency = repeat_bookers['booking_count'].value_counts().sort_index()

    print("\nFoglalási gyakoriság:")
    print("-" * 40)
    for foglalás_szám, felhasználók in booking_frequency.items():
        print(f"{foglalás_szám}x foglalt: {felhasználók} felhasználó")

    # Top foglalók részletesebb elemzése
    top_bookers = repeat_bookers.nlargest(5, 'booking_count')
    top_booker_details = repeat_booker_details[repeat_booker_details['uuid'].isin(top_bookers['uuid'])]

    print("\nTop 5 legtöbbet foglaló felhasználó részletes adatai:")
    print("-" * 40)
    for uuid in top_bookers['uuid']:
        user_bookings = top_booker_details[top_booker_details['uuid'] == uuid]
        összes_vendég = user_bookings['adults'].sum() + user_bookings['children'].sum()

        print(f"\nFelhasználó ID: {uuid}")
        print(f"Foglalások száma: {len(user_bookings)} alkalom")
        print(f"Összes vendég: {összes_vendég} fő")
        print(f"  - Felnőttek: {user_bookings['adults'].sum()} fő")
        print(f"  - Gyerekek: {user_bookings['children'].sum()} fő")

        # Devizánkénti költés
        for currency in user_bookings['currency'].unique():
            currency_bookings = user_bookings[user_bookings['currency'] == currency]
            print(f"Összes költés ({currency}): {currency_bookings['total_price_final'].sum():,.0f} {currency}")
            print(f"Átlagos költés/foglalás ({currency}): {currency_bookings['total_price_final'].mean():,.0f} {currency}")

        # Források megjelenítése csak a nem 0 értékűeknél
        források = user_bookings['utm_source'].value_counts().to_dict()
        források = {k: v for k, v in források.items() if v > 0}
        print(f"Források: {források}")

if __name__ == "__main__":
    input_dir = '/config/workspace/verseny_dataklub_morgens/data/raw/hotel_1'
    analyze_hotel_bookings(input_dir)
