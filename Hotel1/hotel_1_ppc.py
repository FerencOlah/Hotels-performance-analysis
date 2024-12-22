from hotel_1_data_cleaner import optimize_dataframes
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML

# Adatok beolvasása
dataframes = optimize_dataframes('/config/workspace/verseny_dataklub_morgens/data/raw/hotel_1')
datepicker = dataframes['datepicker_daily_visitors']
ppc_budget = dataframes['daily_ppc_budget']
search_log = dataframes['search_log']
search_log_room = dataframes['search_log_room']
search_log_room_child = dataframes['search_log_room_child']
search_log_session = dataframes['search_log_session']

# Stílus beállítások
pd.set_option('display.precision', 2)
pd.set_option('display.max_columns', None)

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

def clean_and_display_revenue(df, currency):
    """
    Tisztítja és megjeleníti a bevételi táblázatot
    """
    # Csak azokat az oszlopokat tartjuk meg, ahol van érték
    sums = df.sum()
    non_zero_cols = sums[sums != 0].index

    # Szűrjük a DataFrame-et
    df_cleaned = df[non_zero_cols]

    # Formázás és megjelenítés
    styled_df = style_dataframe(df_cleaned, f"Bevételek {currency}-ban")
    display(styled_df)

    return df_cleaned

# PPC források szűrése
ppc_sources = ['google / cpc', 'facebook / cpc', 'instagram / cpc', 'bing / cpc']
datepicker_ppc = datepicker[datepicker['utm_source_and_medium'].isin(ppc_sources)]

# Napi aggregálás forrásonként
daily_ppc_traffic = datepicker_ppc.groupby(['date', 'utm_source_and_medium'])\
    .agg({
        'user_count': 'sum',
        'session_count': 'sum'
    }).reset_index()

# Dátumok konvertálása
daily_ppc_traffic['date'] = pd.to_datetime(daily_ppc_traffic['date'])
ppc_budget['date'] = pd.to_datetime(ppc_budget['date'])

# Egységes színek definiálása
colors = {
    'google / cpc': '#1f77b4',    # kék
    'facebook / cpc': '#ff7f0e',  # narancssárga
    'instagram / cpc': '#2ca02c', # zöld
    'bing / cpc': '#d62728',      # piros
    'Google': '#1f77b4',          # ugyanaz a kék
    'Bing': '#d62728',           # ugyanaz a piros
    'Meta (FB+IG)': '#2ca02c'    # ugyanaz a zöld
}

# Költségek és forgalom vizualizációja
plt.figure(figsize=(15, 10))

# Napi forgalom forrásonként
plt.subplot(2, 1, 1)
for source in ppc_sources:
    source_data = daily_ppc_traffic[daily_ppc_traffic['utm_source_and_medium'] == source]
    plt.plot(source_data['date'], source_data['user_count'], 
             label=source, marker='o', color=colors[source])

plt.title('Napi látogatók száma PPC forrásonként')
plt.xlabel('Dátum')
plt.ylabel('Látogatók száma')
plt.legend()
plt.grid(True)

# Napi költségek
plt.subplot(2, 1, 2)
plt.plot(ppc_budget['date'], ppc_budget['daily_google_spend'], 
         label='Google', marker='o', color=colors['Google'])
plt.plot(ppc_budget['date'], ppc_budget['daily_microsoft_spend'], 
         label='Bing', marker='o', color=colors['Bing'])
plt.plot(ppc_budget['date'], ppc_budget['daily_meta_spend'], 
         label='Meta (FB+IG)', marker='o', color=colors['Meta (FB+IG)'])

plt.title('Napi PPC költségek platformonként')
plt.xlabel('Dátum')
plt.ylabel('Költség (HUF)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

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

# Sikeres foglalások szűrése és kategorizálása
successful_bookings = search_log[search_log['conversion'] == 1].copy()
successful_bookings['family_category'] = successful_bookings.apply(
    lambda row: detailed_categorize_booking(row, search_log_room, search_log_room_child), axis=1
)

# Foglalások összekapcsolása a session adatokkal
bookings_with_source = pd.merge(
    successful_bookings,
    search_log_session[['id', 'utm_source', 'utm_medium']],
    left_on='search_log_session_id',
    right_on='id',
    how='left'
)

# PPC foglalások szűrése
ppc_bookings = bookings_with_source[
    (bookings_with_source['utm_medium'] == 'cpc') &
    (bookings_with_source['utm_source'].isin(['google', 'facebook', 'instagram', 'bing']))
]

# Célcsoportés PPC források kereszttáblája
family_source_dist = pd.crosstab(
    ppc_bookings['family_category'],
    ppc_bookings['utm_source'],
    normalize='columns'
) * 100

# Célcsoport vizualizációja
plt.figure(figsize=(15, 8))
colors_map = {'google': colors['google / cpc'],
              'facebook': colors['facebook / cpc'],
              'instagram': colors['instagram / cpc'],
              'bing': colors['bing / cpc']}

ax = family_source_dist.plot(kind='bar', color=[colors_map[col] for col in family_source_dist.columns])
plt.title('PPC forrásból származó foglalások célcsoportonkénti eloszlása (%)')
plt.xlabel('Célcsoport')
plt.ylabel('Arány (%)')
plt.legend(title='PPC Forrás')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Konverziós arányok számítása - csak PPC források
ppc_mapping = {
    'google / cpc': 'google',
    'facebook / cpc': 'facebook',
    'instagram / cpc': 'instagram',
    'bing / cpc': 'bing'
}

# Látogatók számának összesítése PPC források szerint
visitors = daily_ppc_traffic.groupby('utm_source_and_medium')['user_count'].sum()
visitors.index = visitors.index.map(lambda x: ppc_mapping.get(x, x))

# Sikeres foglalások számának összesítése
bookings = ppc_bookings['utm_source'].value_counts()

# Konverziós arányok DataFrame létrehozása
conversions = pd.DataFrame({
    'Összes látogató': visitors,
    'Sikeres foglalások': bookings
}).fillna(0)

# Konverziós arány számítása
conversions['Konverziós arány (%)'] = (
    conversions['Sikeres foglalások'] / conversions['Összes látogató'] * 100
).round(3)

# Csak azokat a sorokat tartjuk meg, ahol van látogató vagy foglalás
conversions = conversions[
    (conversions['Összes látogató'] > 0) | 
    (conversions['Sikeres foglalások'] > 0)
]

# Stílusos megjelenítés
styled_conversions = style_dataframe(conversions, "Konverziós arányok forrásonként")
display(styled_conversions)

# Foglalások száma (abszolút értékben)
bookings_count = pd.crosstab(
    ppc_bookings['family_category'],
    ppc_bookings['utm_source']
)

styled_bookings_count = style_dataframe(bookings_count, "Foglalások száma PPC forrásonként (db)")
display(styled_bookings_count)

# Bevételek elemzése külön HUF és EUR
# HUF foglalások
huf_bookings = ppc_bookings[ppc_bookings['currency'] == 'HUF']
huf_revenue = pd.pivot_table(
    huf_bookings,
    values='total_price_final',
    index='family_category',
    columns='utm_source',
    aggfunc=['count', 'sum'],
    fill_value=0
)

# EUR foglalások
eur_bookings = ppc_bookings[ppc_bookings['currency'] == 'EUR']
eur_revenue = pd.pivot_table(
    eur_bookings,
    values='total_price_final',
    index='family_category',
    columns='utm_source',
    aggfunc=['count', 'sum'],
    fill_value=0
)

# Tisztított táblázatok megjelenítése
huf_revenue_cleaned = clean_and_display_revenue(huf_revenue, "HUF")
eur_revenue_cleaned = clean_and_display_revenue(eur_revenue, "EUR")

# Vizualizáció az abszolút számokkal
plt.figure(figsize=(15, 8))
ax = bookings_count.plot(kind='bar', color=[colors_map[col] for col in bookings_count.columns])
plt.title('Foglalások abszolút száma forrásonként és célcsoportonként')
plt.xlabel('Célcsoport')
plt.ylabel('Foglalások száma (db)')
plt.legend(title='PPC Forrás')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.show()
