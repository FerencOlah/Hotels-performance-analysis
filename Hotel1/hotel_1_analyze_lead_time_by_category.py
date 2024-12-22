from hotel_1_data_cleaner import optimize_dataframes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Adatok beolvasása és optimalizálása
dataframes = optimize_dataframes('/config/workspace/verseny_dataklub_morgens/data/raw/hotel_1')

# DataFrames kinyerése a dictionary-ből
booking_data = dataframes['booking_data']
search_log = dataframes['search_log']
search_log_room = dataframes['search_log_room']
search_log_room_child = dataframes['search_log_room_child']

def detailed_categorize_booking(row, search_log_room, search_log_room_child):
    if row['children'] == 0:
        if row['adults'] == 1:
            return 'Egyedülálló'
        elif row['adults'] == 2:
            return 'Pár'
        elif row['adults'] == 3:
            return 'Három felnőtt'
        elif row['adults'] == 4:
            return 'Négy felnőtt'
        else:
            return 'Nagy csoport (5+ felnőtt)'
    else:
        children_info = search_log_room_child[
            search_log_room_child['search_log_room_id'].isin(
                search_log_room[search_log_room['search_log_id'] == row.name]['id']
            )
        ]

        has_toddler = any(children_info['age'] <= 2) if not children_info.empty else False
        num_children = row['children']

        if has_toddler:
            return f'Család kisgyerekkel (0-2 év)'
        elif num_children == 1:
            return 'Család 1 gyerekkel'
        elif num_children == 2:
            return 'Család 2 gyerekkel'
        else:
            return f'Család 3+ gyerekkel'

def analyze_lead_time_by_category(booking_data, search_log, search_log_room, search_log_room_child):
    # Csak a tényleges foglalásokat nézzük
    actual_bookings = search_log[search_log['id'].isin(booking_data['search_log_id'])]

    # Lead time kiszámítása
    actual_bookings['lead_time'] = (actual_bookings['arrival'] - actual_bookings['utc_datetime']).dt.days

    # Kategóriák hozzáadása
    actual_bookings['guest_category'] = actual_bookings.apply(
        lambda row: detailed_categorize_booking(row, search_log_room, search_log_room_child), 
        axis=1
    )

    # Alap statisztikák számítása
    stats = actual_bookings.groupby('guest_category').agg({
        'lead_time': ['count', 'mean', 'median', 'std', 'min', 'max']
    }).round(1)

    # Vizualizáció
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=actual_bookings, x='guest_category', y='lead_time', showfliers=False)
    plt.xticks(rotation=45, ha='right')
    plt.title('Lead Time Eloszlás Vendégkategóriánként')
    plt.xlabel('Vendégkategória')
    plt.ylabel('Lead Time (napok)')
    plt.tight_layout()

    # Percentilisek számítása
    percentiles = actual_bookings.groupby('guest_category')['lead_time'].agg([
        ('25%', lambda x: np.percentile(x, 25)),
        ('50%', lambda x: np.percentile(x, 50)),
        ('75%', lambda x: np.percentile(x, 75))
    ]).round(1)

    return {
        'basic_stats': stats,
        'percentiles': percentiles,
        'detailed_data': actual_bookings[['guest_category', 'lead_time', 'arrival', 'utc_datetime']]
    }

# Függvény használata
results = analyze_lead_time_by_category(booking_data, search_log, search_log_room, search_log_room_child)

# Eredmények megjelenítése
print("\nLead Time Alapstatisztikák Kategóriánként:")
print("=========================================")
print(results['basic_stats'])

print("\nLead Time Percentilisek Kategóriánként:")
print("======================================")
print(results['percentiles'])

# Átlagos lead time-ok sorrendben
avg_lead_times = results['detailed_data'].groupby('guest_category')['lead_time'].mean().sort_values(ascending=False)
print("\nÁtlagos Lead Time Kategóriánként (csökkenő sorrendben):")
print("====================================================")
print(avg_lead_times)

# További hasznos statisztikák
print("\nKategóriánkénti foglalások száma:")
print("================================")
bookings_by_category = results['detailed_data']['guest_category'].value_counts()
print(bookings_by_category)

# Havi bontás kategóriánként
results['detailed_data']['booking_month'] = results['detailed_data']['utc_datetime'].dt.month
monthly_stats = results['detailed_data'].groupby(['guest_category', 'booking_month'])['lead_time'].mean().unstack()
print("\nÁtlagos Lead Time havonként és kategóriánként:")
print("============================================")
print(monthly_stats.round(1))

# Plot mentése
plt.savefig('lead_time_by_category.png')
plt.close()

#######################################

def analyze_lead_time_by_source_and_category(booking_data, search_log, search_log_room, search_log_room_child, search_log_session, datepicker_daily_visitors):
    # Csak a tényleges foglalásokat nézzük
    actual_bookings = search_log[search_log['id'].isin(booking_data['search_log_id'])].copy()

    # Lead time kiszámítása
    actual_bookings['lead_time'] = (actual_bookings['arrival'] - actual_bookings['utc_datetime']).dt.days

    # Kategóriák hozzáadása
    actual_bookings['guest_category'] = actual_bookings.apply(
        lambda row: detailed_categorize_booking(row, search_log_room, search_log_room_child), 
        axis=1
    )

    # Marketing források hozzáadása search_log_session-ön keresztül
    # Először készítsük el az utm_source_and_medium oszlopot a search_log_session táblában
    search_log_session = search_log_session.copy()
    search_log_session['utm_source_and_medium'] = search_log_session['utm_source'].astype(str) + ' / ' + search_log_session['utm_medium'].astype(str)

    actual_bookings = actual_bookings.merge(
        search_log_session[['id', 'utm_source_and_medium']], 
        left_on='search_log_session_id', 
        right_on='id', 
        suffixes=('', '_session')
    )

    # Csak a megadott PPC források szűrése
    ppc_sources = ['google / cpc', 'facebook / cpc', 'instagram / cpc', 'bing / cpc']
    ppc_bookings = actual_bookings[actual_bookings['utm_source_and_medium'].isin(ppc_sources)]

    # Ha nincs elegendő adat, adjunk vissza üres eredményt
    if len(ppc_bookings) == 0:
        print("Figyelmeztetés: Nem található megfelelő PPC forrásból származó foglalás!")
        return None

    # Alap statisztikák számítása forrás és kategória szerint
    stats = ppc_bookings.groupby(['utm_source_and_medium', 'guest_category']).agg({
        'lead_time': ['count', 'mean', 'median', 'std', 'min', 'max']
    }).round(1)

    # Vizualizáció
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=ppc_bookings, x='guest_category', y='lead_time', hue='utm_source_and_medium', showfliers=False)
    plt.xticks(rotation=45, ha='right')
    plt.title('Lead Time Eloszlás Vendégkategóriánként és Marketing Forrásonként')
    plt.xlabel('Vendégkategória')
    plt.ylabel('Lead Time (napok)')
    plt.legend(title='Marketing Forrás', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Havi bontás hozzáadása
    ppc_bookings['booking_month'] = ppc_bookings['utc_datetime'].dt.month
    monthly_stats = ppc_bookings.groupby(['utm_source_and_medium', 'guest_category', 'booking_month'])['lead_time'].mean().unstack()

    return {
        'basic_stats': stats,
        'monthly_stats': monthly_stats.round(1),
        'detailed_data': ppc_bookings[['guest_category', 'lead_time', 'utm_source_and_medium', 'arrival', 'utc_datetime']]
    }

# Mindkét függvény használata
results_by_category = analyze_lead_time_by_category(booking_data, search_log, search_log_room, search_log_room_child)
results_by_source = analyze_lead_time_by_source_and_category(
    booking_data, 
    search_log, 
    search_log_room, 
    search_log_room_child,
    dataframes['search_log_session'],
    dataframes['datepicker_daily_visitors']
)

# Eredmények megjelenítése (csak ha van adat)
if results_by_source is not None:
    print("\nLead Time Statisztikák Marketing Forrás és Kategória szerint:")
    print("========================================================")
    print(results_by_source['basic_stats'])

    print("\nHavi átlagos Lead Time Marketing Forrás és Kategória szerint:")
    print("=======================================================")
    print(results_by_source['monthly_stats'])

    # Források szerinti foglalások száma
    bookings_by_source = results_by_source['detailed_data']['utm_source_and_medium'].value_counts()
    print("\nFoglalások száma Marketing Forrásonként:")
    print("=====================================")
    print(bookings_by_source)

    # Plot mentése
    plt.savefig('lead_time_by_source_and_category.png')
    plt.close()