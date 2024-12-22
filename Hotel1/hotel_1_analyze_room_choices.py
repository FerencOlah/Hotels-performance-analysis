import pandas as pd
import os
from hotel_1_data_cleaner import optimize_dataframes
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import display, HTML

# Stílus beállítások
pd.set_option('display.precision', 2)
pd.set_option('display.max_columns', None)

# Seaborn alapbeállítások
sns.set_theme()  # Ez helyettesíti a plt.style.use('seaborn') sort

# Színek definiálása
colors_map = {
    'google': '#1f77b4',
    'facebook': '#ff7f0e',
    'instagram': '#2ca02c',
    'bing': '#d62728'
}

def style_dataframe(df, caption=""):
    """
    Stílusos táblázat formázás
    """
    return df.style\
        .background_gradient(cmap='YlOrRd')\
        .set_caption(caption)\
        .format("{:.2f}")\
        .set_properties(**{
            'text-align': 'center',
            'border': '1px solid gray',
            'padding': '5px'
        })\
        .set_table_styles([
            {'selector': 'caption', 'props': [('caption-side', 'top'), 
                                            ('font-size', '16px'),
                                            ('font-weight', 'bold')]},
            {'selector': 'th', 'props': [('background-color', '#f0f0f0'),
                                       ('color', 'black'),
                                       ('font-weight', 'bold'),
                                       ('text-align', 'center')]},
            {'selector': 'td', 'props': [('text-align', 'center')]},
        ])

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

def analyze_room_choices(dataframes):
    """
    Elemzi a szoba választásokat különböző vendég kategóriák és devizák szerint.
    """
    search_log = dataframes['search_log']
    search_log_room = dataframes['search_log_room']
    search_log_room_child = dataframes['search_log_room_child']

    # Csak a sikeres foglalásokat nézzük (conversion = 1)
    successful_bookings = search_log[search_log['conversion'] == 1].copy()

    # Foglalások összekapcsolása a szoba információkkal
    booking_analysis = successful_bookings.merge(
        search_log_room[['search_log_id', 'picked_room']], 
        left_on='id', 
        right_on='search_log_id', 
        how='left'
    )

    # Kategóriák létrehozása
    booking_analysis['booking_category'] = booking_analysis.apply(
        lambda x: detailed_categorize_booking(x, search_log_room, search_log_room_child), 
        axis=1
    )

    results = {}

    # Devizánkénti elemzés (HUF és EUR)
    for currency in ['HUF', 'EUR']:
        currency_data = booking_analysis[booking_analysis['currency'] == currency]

        if len(currency_data) > 0:  # Csak akkor elemezzük, ha van adat az adott devizában
            # Szoba választások elemzése kategóriánként
            room_choices = pd.crosstab(
                currency_data['booking_category'], 
                currency_data['picked_room'],
                normalize='index'
            ) * 100

            # Abszolút számok
            room_counts = pd.crosstab(
                currency_data['booking_category'], 
                currency_data['picked_room']
            )

            # Vizualizációk devizánként
            plt.figure(figsize=(15, 8))
            room_choices.plot(kind='bar', stacked=True)
            plt.title(f'Szoba választások megoszlása kategóriánként ({currency})')
            plt.xlabel('Foglalási kategória')
            plt.ylabel('Százalék')
            plt.legend(title='Szobatípus', bbox_to_anchor=(1.05, 1))
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(12, 8))
            sns.heatmap(room_counts, annot=True, fmt='d', cmap='YlOrRd')
            plt.title(f'Szoba foglalások száma kategóriánként ({currency})')
            plt.tight_layout()
            plt.show()

            results[currency] = {
                'percentage': room_choices,
                'counts': room_counts
            }

    return results

def analyze_room_choices_by_ppc(search_log, search_log_room, search_log_session):
    """
    Elemzi a szoba választásokat PPC források szerint
    """
    # Foglalások összekapcsolása a szoba és session adatokkal
    bookings_with_rooms = pd.merge(
        search_log[search_log['conversion'] == 1],
        search_log_room[['search_log_id', 'picked_room']],
        left_on='id',
        right_on='search_log_id',
        how='left'
    )

    bookings_complete = pd.merge(
        bookings_with_rooms,
        search_log_session[['id', 'utm_source', 'utm_medium']],
        left_on='search_log_session_id',
        right_on='id',
        how='left'
    )

    # PPC foglalások szűrése
    ppc_bookings = bookings_complete[
        (bookings_complete['utm_medium'] == 'cpc') &
        (bookings_complete['utm_source'].isin(['google', 'facebook', 'instagram', 'bing']))
    ]

    results = {}

    # 1. Szoba választások megoszlása PPC forrásonként
    room_source_dist = pd.crosstab(
        ppc_bookings['picked_room'],
        ppc_bookings['utm_source'],
        normalize='columns'
    ) * 100

    # Abszolút számok
    room_source_counts = pd.crosstab(
        ppc_bookings['picked_room'],
        ppc_bookings['utm_source']
    )

    # Vizualizációk
    plt.figure(figsize=(15, 8))
    ax = room_source_dist.plot(kind='bar', color=[colors_map[col] for col in room_source_dist.columns])
    plt.title('Szoba választások megoszlása PPC forrásonként (%)')
    plt.xlabel('Szobatípus')
    plt.ylabel('Arány (%)')
    plt.legend(title='PPC Forrás')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))
    sns.heatmap(room_source_counts, annot=True, fmt='d', cmap='YlOrRd')
    plt.title('Szoba foglalások száma PPC forrásonként')
    plt.xlabel('PPC Forrás')
    plt.ylabel('Szobatípus')
    plt.tight_layout()
    plt.show()

    # 2. Devizánkénti bontás PPC forrásonként
    for currency in ['HUF', 'EUR']:
        currency_bookings = ppc_bookings[ppc_bookings['currency'] == currency]

        if len(currency_bookings) > 0:
            # Százalékos megoszlás
            currency_dist = pd.crosstab(
                currency_bookings['picked_room'],
                currency_bookings['utm_source'],
                normalize='columns'
            ) * 100

            # Abszolút számok
            currency_counts = pd.crosstab(
                currency_bookings['picked_room'],
                currency_bookings['utm_source']
            )

            # Átlagos foglalási érték
            avg_price = currency_bookings.pivot_table(
                values='total_price_final',
                index='picked_room',
                columns='utm_source',
                aggfunc='mean'
            )

            # Stílusos megjelenítés
            print(f"\nSzoba választások {currency} devizában PPC forrásonként (%):")
            styled_dist = style_dataframe(currency_dist.round(2), 
                                        f"Szoba választások megoszlása {currency} devizában (%)")
            display(styled_dist)

            print(f"\nSzoba választások {currency} devizában PPC forrásonként (db):")
            styled_counts = style_dataframe(currency_counts, 
                                          f"Szoba foglalások száma {currency} devizában")
            display(styled_counts)

            # Átlagos foglalási érték heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(avg_price, annot=True, fmt='.0f', cmap='YlOrRd')
            plt.title(f'Átlagos foglalási érték szobatípusonként és PPC forrásonként ({currency})')
            plt.xlabel('PPC Forrás')
            plt.ylabel('Szobatípus')
            plt.tight_layout()
            plt.show()

            results[currency] = {
                'distribution': currency_dist,
                'counts': currency_counts,
                'avg_price': avg_price
            }

    return results

def main():
    # Bemeneti könyvtár megadása
    input_dir = '/config/workspace/verseny_dataklub_morgens/data/raw/hotel_1'

    # Adatok tisztítása és betöltése
    dfs = optimize_dataframes(input_dir)

    # DataFramek külön változókba mentése
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

    print("\n=== Szoba választások elemzése kategóriák szerint ===")
    # Kategória elemzés futtatása
    category_results = analyze_room_choices(dataframes)

    print("\n=== Szoba választások elemzése PPC források szerint ===")
    # PPC elemzés futtatása
    ppc_results = analyze_room_choices_by_ppc(
        dataframes['search_log'],
        dataframes['search_log_room'],
        dataframes['search_log_session']
    )

if __name__ == "__main__":
    main()
