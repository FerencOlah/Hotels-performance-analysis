from hotel_1_data_cleaner import optimize_dataframes
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Adatok beolvasása
dataframes = optimize_dataframes('/config/workspace/verseny_dataklub_morgens/data/raw/hotel_1')
search_log = dataframes['search_log']
search_log_room = dataframes['search_log_room']
search_log_room_child = dataframes['search_log_room_child']

# Dátum konvertálása
search_log['arrival'] = pd.to_datetime(search_log['arrival'])

# Csak a sikeres foglalásokat nézzük (conversion = 1)
successful_bookings = search_log[search_log['conversion'] == 1].copy()

# Pénznemek szerinti szétválasztás
huf_bookings = successful_bookings[successful_bookings['currency'] == 'HUF']
eur_bookings = successful_bookings[successful_bookings['currency'] == 'EUR']

# Alapvető foglalási típusok létrehozása mindkét pénznemre
def create_booking_types(bookings):
    return pd.DataFrame({
        'adults': bookings['adults'],
        'children': bookings['children'],
        'currency': bookings['currency'],
        'nights': bookings['nights'],
        'total_price': bookings['total_price_final']
    })

huf_booking_types = create_booking_types(huf_bookings)
eur_booking_types = create_booking_types(eur_bookings)

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

# Kategóriák hozzáadása mindkét pénznemhez
huf_booking_types['detailed_category'] = huf_booking_types.apply(
    lambda row: detailed_categorize_booking(row, search_log_room, search_log_room_child), axis=1
)
eur_booking_types['detailed_category'] = eur_booking_types.apply(
    lambda row: detailed_categorize_booking(row, search_log_room, search_log_room_child), axis=1
)

def print_statistics(booking_types, currency):
    print(f"\n=== Részletes statisztikák ({currency}) ===")

    print(f"\nFoglalások megoszlása ({currency}):")
    category_counts = booking_types['detailed_category'].value_counts()
    print(category_counts)

    print(f"\nÁtlagos tartózkodási idő kategóriánként ({currency}):")
    avg_nights = booking_types.groupby('detailed_category')['nights'].mean().round(2)
    print(avg_nights)

    print(f"\nÁtlagos költés kategóriánként ({currency}):")
    avg_spending = booking_types.groupby('detailed_category')['total_price'].mean().round(2)
    print(avg_spending)

    return category_counts, avg_nights, avg_spending

# Statisztikák mindkét pénznemre
huf_stats = print_statistics(huf_booking_types, 'HUF')
eur_stats = print_statistics(eur_booking_types, 'EUR')

# Vizualizáció mindkét pénznemre
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))

# HUF foglalások
huf_stats[0].plot(kind='bar', ax=ax1, color='blue', alpha=0.6)
ax1.set_title('Foglalások megoszlása (HUF)')
ax1.set_xlabel('Kategória')
ax1.set_ylabel('Foglalások száma')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# EUR foglalások
eur_stats[0].plot(kind='bar', ax=ax2, color='green', alpha=0.6)
ax2.set_title('Foglalások megoszlása (EUR)')
ax2.set_xlabel('Kategória')
ax2.set_ylabel('Foglalások száma')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# HUF átlagos költés
huf_stats[2].plot(kind='bar', ax=ax3, color='blue', alpha=0.6)
ax3.set_title('Átlagos költés kategóriánként (HUF)')
ax3.set_xlabel('Kategória')
ax3.set_ylabel('Átlagos költés (HUF)')
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

# EUR átlagos költés
eur_stats[2].plot(kind='bar', ax=ax4, color='green', alpha=0.6)
ax4.set_title('Átlagos költés kategóriánként (EUR)')
ax4.set_xlabel('Kategória')
ax4.set_ylabel('Átlagos költés (EUR)')
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()

# További részletes elemzések pénznemenként
for currency, booking_types in [('HUF', huf_booking_types), ('EUR', eur_booking_types)]:
    print(f"\n=== További részletes statisztikák ({currency}) ===")

    # Egy éjszakára jutó átlagos költség
    per_night_cost = (booking_types.groupby('detailed_category')['total_price'].mean() / 
                     booking_types.groupby('detailed_category')['nights'].mean()).round(2)
    print(f"\nEgy éjszakára jutó átlagos költség ({currency}):")
    print(per_night_cost.sort_values(ascending=False))

    # Átlagos vendégszám
    booking_types['total_guests'] = booking_types['adults'] + booking_types['children']
    avg_guests = booking_types.groupby('detailed_category')['total_guests'].mean().round(2)
    print(f"\nÁtlagos vendégszám kategóriánként ({currency}):")
    print(avg_guests.sort_values(ascending=False))

    # Egy főre jutó átlagos költség
    per_person_cost = (booking_types.groupby('detailed_category')['total_price'].mean() / 
                      booking_types.groupby('detailed_category')['total_guests'].mean()).round(2)
    print(f"\nEgy főre jutó átlagos költség ({currency}):")
    print(per_person_cost.sort_values(ascending=False))

    print("\n=== Teljes bevétel elemzése ===")

print("\n=== Teljes bevétel elemzése ===")

# Bevétel elemzés függvény
def analyze_revenue(booking_types, currency):
    # Teljes bevétel számítása kategóriánként
    total_revenue = booking_types.groupby('detailed_category')['total_price'].agg([
        ('Teljes bevétel', 'sum'),
        ('Foglalások száma', 'count')
    ])
    
    # Százalékos megoszlás számítása
    total_revenue['Bevétel megoszlása (%)'] = (total_revenue['Teljes bevétel'] / total_revenue['Teljes bevétel'].sum() * 100).round(2)
    
    # Átlagos foglalási érték számítása
    total_revenue['Átlagos foglalási érték'] = (total_revenue['Teljes bevétel'] / total_revenue['Foglalások száma']).round(2)
    
    # Eredmények rendezése bevétel szerint
    return total_revenue.sort_values('Teljes bevétel', ascending=False)

# HUF elemzés
print("\nTeljes bevétel kategóriánként (HUF):")
huf_analysis = analyze_revenue(huf_booking_types, 'HUF')
print(huf_analysis.to_string(float_format=lambda x: '{:,.0f}'.format(x)))

# EUR elemzés
print("\nTeljes bevétel kategóriánként (EUR):")
eur_analysis = analyze_revenue(eur_booking_types, 'EUR')
print(eur_analysis.to_string(float_format=lambda x: '{:,.2f}'.format(x)))

# Vizualizáció
plt.figure(figsize=(15, 10))

# HUF bevétel megoszlás
plt.subplot(2, 1, 1)
huf_analysis['Bevétel megoszlása (%)'].plot(kind='bar', color='blue', alpha=0.6)
plt.title('Bevétel megoszlása kategóriánként (HUF)')
plt.xlabel('Kategória')
plt.ylabel('Bevétel aránya (%)')
plt.xticks(rotation=45, ha='right')

# EUR bevétel megoszlás
plt.subplot(2, 1, 2)
eur_analysis['Bevétel megoszlása (%)'].plot(kind='bar', color='green', alpha=0.6)
plt.title('Bevétel megoszlása kategóriánként (EUR)')
plt.xlabel('Kategória')
plt.ylabel('Bevétel aránya (%)')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

# Összesített statisztikák
for currency, analysis in [('HUF', huf_analysis), ('EUR', eur_analysis)]:
    print(f"\n=== Összesített statisztikák ({currency}) ===")
    print(f"Összes bevétel: {analysis['Teljes bevétel'].sum():,.2f} {currency}")
    print(f"Összes foglalás: {analysis['Foglalások száma'].sum():,.0f} db")
    print(f"\nTop 3 kategória bevétel szerint:")
    top_3 = analysis.nlargest(3, 'Teljes bevétel')
    for idx, row in top_3.iterrows():
        print(f"{idx}: {row['Teljes bevétel']:,.2f} {currency} ({row['Bevétel megoszlása (%)']}%)")
########################################################################################
# Részletes kampány elemzés devizánként
print("\n=== Kampány hatékonyság elemzése devizánként ===")
# Search log session adatok kinyerése a dataframes szótárból
search_log_session = dataframes['search_log_session']

# Detailed category hozzáadása a successful_bookings-hoz
successful_bookings['detailed_category'] = successful_bookings.apply(
    lambda row: detailed_categorize_booking(row, search_log_room, search_log_room_child), axis=1
)

# Successful bookings összekapcsolása a session adatokkal
successful_bookings_with_session = pd.merge(
    successful_bookings,
    search_log_session[['id', 'utm_campaign']],
    left_on='search_log_session_id',
    right_on='id',
    how='left'
)

# Hiányzó kampány értékek kezelése
successful_bookings_with_session['utm_campaign'].fillna('(not set)', inplace=True)

# Külön elemzés HUF és EUR foglalásokra
for currency in ['HUF', 'EUR']:
    bookings_by_currency = successful_bookings_with_session[
        successful_bookings_with_session['currency'] == currency
    ]

    # Kampányok elemzése az adott devizára
    campaign_analysis = bookings_by_currency.groupby(
        ['detailed_category', 'utm_campaign']
    ).size().unstack(fill_value=0)

    # Oszlopok szűrése - csak azok maradnak, ahol volt legalább 1 konverzió
    campaign_analysis = campaign_analysis.loc[:, campaign_analysis.sum() > 0]

    campaign_percentages = campaign_analysis.div(campaign_analysis.sum(axis=1), axis=0) * 100

    # Táblázatos megjelenítés
    print(f"\n=== Kampány elemzés - {currency} foglalások ===")

    # Abszolút számok táblázata
    print("\nFoglalások száma kampányonként és kategóriánként:")
    styled_analysis = campaign_analysis.style\
        .background_gradient(cmap='YlOrRd')\
        .format("{:.0f}")\
        .set_caption(f"Foglalások száma - {currency}")
    display(styled_analysis)

    # Százalékos megoszlás táblázata
    print("\nKampányok megoszlása kategóriánként (%):")
    styled_percentages = campaign_percentages.style\
        .background_gradient(cmap='YlOrRd')\
        .format("{:.1f}%")\
        .set_caption(f"Kampányok megoszlása (%) - {currency}")
    display(styled_percentages)

    # Top 5 legsikeresebb kampány
    print(f"\nTop 5 legsikeresebb kampány - {currency}:")
    top_campaigns = campaign_analysis.sum().sort_values(ascending=False).head()
    styled_top = pd.DataFrame({
        'Foglalások száma': top_campaigns,
        'Részarány (%)': (top_campaigns / top_campaigns.sum() * 100).round(1)
    }).style\
        .background_gradient(cmap='YlOrRd')\
        .format({'Foglalások száma': '{:.0f}', 'Részarány (%)': '{:.1f}%'})\
        .set_caption("Top 5 kampány")
    display(styled_top)

    # Vizualizáció (egyszerűsített)
    plt.figure(figsize=(15, 8))
    campaign_percentages.plot(
        kind='bar', 
        stacked=True,
        color=plt.cm.Set3(np.linspace(0, 1, len(campaign_percentages.columns)))
    )
    plt.title(f'Kampányok hatékonysága foglalási kategóriánként - {currency}')
    plt.xlabel('Kategória')
    plt.ylabel('Megoszlás (%)')
    plt.legend(title='Kampány', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Összefoglaló statisztikák
    print(f"\nÖsszefoglaló statisztikák - {currency}:")
    summary_stats = pd.DataFrame({
        'Összes foglalás': campaign_analysis.sum().sum(),
        'Aktív kampányok száma': len(campaign_analysis.columns),
        'Átlagos foglalás/kampány': campaign_analysis.sum().mean(),
        'Legnagyobb kampány részesedés (%)': campaign_analysis.sum().max() / campaign_analysis.sum().sum() * 100
    }, index=['Érték']).T

    styled_summary = summary_stats.style\
        .format({'Érték': '{:.1f}'})\
        .set_caption("Összefoglaló statisztikák")
    display(styled_summary)

# Részletes kampány elemzés devizánként
print("\n=== Kampány hatékonyság elemzése devizánként ===")

# Színek definiálása
colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', 
          '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd']

# Külön elemzés HUF és EUR foglalásokra
for currency in ['HUF', 'EUR']:
    bookings_by_currency = successful_bookings_with_session[
        successful_bookings_with_session['currency'] == currency
    ]

    # Kampányok elemzése az adott devizára
    campaign_analysis = bookings_by_currency.groupby(
        ['detailed_category', 'utm_campaign']
    ).size().unstack(fill_value=0)

    # Oszlopok szűrése - csak azok maradnak, ahol volt legalább 1 konverzió
    campaign_analysis = campaign_analysis.loc[:, campaign_analysis.sum() > 0]

    # Százalékok számítása
    campaign_percentages = campaign_analysis.div(campaign_analysis.sum(axis=1), axis=0) * 100

    # Táblázatos megjelenítés
    print(f"\n=== Kampány elemzés - {currency} foglalások ===")

    # Abszolút számok táblázata
    print("\nFoglalások száma kampányonként és kategóriánként:")
    styled_analysis = campaign_analysis.style\
        .background_gradient(cmap='YlOrRd')\
        .format("{:.0f}")\
        .set_caption(f"Foglalások száma - {currency}")
    display(styled_analysis)

    # Vizualizáció
    plt.figure(figsize=(12, 8))
    ax = campaign_percentages.plot(
        kind='bar',
        stacked=True,
        color=colors[:len(campaign_percentages.columns)],
        width=0.8
    )

    # Cím és tengelyek
    plt.title(f'Kampányok hatékonysága foglalási kategóriánként - {currency}',
              fontsize=14, pad=20)
    plt.xlabel('Kategória', fontsize=12)
    plt.ylabel('Megoszlás (%)', fontsize=12)

    # Y tengely 0-100 között

    plt.ylim(0, 100)

    # Rács hozzáadása
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # Legend módosítása
    plt.legend(title='Kampány',
              bbox_to_anchor=(1.05, 1),
              loc='upper left',
              fontsize=10,
              title_fontsize=12)

    # Tengelyek módosítása
    plt.xticks(rotation=30, ha='right', fontsize=10)
    plt.yticks(fontsize=10)

    # Értékek megjelenítése (opcionális)
    # for c in ax.containers:
    #     ax.bar_label(c, fmt='%.0f%%', label_type='center')

    # Margók igazítása
    plt.tight_layout()
    plt.show()

    # Top 5 legsikeresebb kampány
    print(f"\nTop 5 legsikeresebb kampány - {currency}:")
    top_campaigns = campaign_analysis.sum().sort_values(ascending=False).head()
    styled_top = pd.DataFrame({
        'Foglalások száma': top_campaigns,
        'Részarány (%)': (top_campaigns / top_campaigns.sum() * 100).round(1)
    }).style\
        .background_gradient(cmap='YlOrRd')\
        .format({'Foglalások száma': '{:.0f}', 'Részarány (%)': '{:.1f}%'})\
        .set_caption("Top 5 kampány")
    display(styled_top)

    # Összefoglaló statisztikák
    print(f"\nÖsszefoglaló statisztikák - {currency}:")
    summary_stats = pd.DataFrame({
        'Összes foglalás': campaign_analysis.sum().sum(),
        'Aktív kampányok száma': len(campaign_analysis.columns),
        'Átlagos foglalás/kampány': campaign_analysis.sum().mean(),
        'Legnagyobb kampány részesedés (%)': campaign_analysis.sum().max() / campaign_analysis.sum().sum() * 100
    }, index=['Érték']).T

    styled_summary = summary_stats.style\
        .format({'Érték': '{:.1f}'})\
        .set_caption("Összefoglaló statisztikák")
    display(styled_summary)
