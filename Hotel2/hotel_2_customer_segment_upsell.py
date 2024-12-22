from hotel_2_data_cleaner import optimize_dataframes
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns 

# Adatok beolvasása
dataframes = optimize_dataframes('/config/workspace/verseny_dataklub_morgens/data/raw/hotel_2')
search_log = dataframes['search_log']
search_log_room = dataframes['search_log_room']
search_log_room_child = dataframes['search_log_room_child']
booking_data = dataframes['booking_data']
upsell_data = dataframes['upsell_data']
datepicker_daily_visitors = dataframes['datepicker_daily_visitors']

# Dátum konvertálása
search_log['arrival'] = pd.to_datetime(search_log['arrival'])
datepicker_daily_visitors['date'] = pd.to_datetime(datepicker_daily_visitors['date'])

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
        'total_price': bookings['total_price_final'],
        'search_log_id': bookings['id']
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
        room_ids = search_log_room[search_log_room['search_log_id'] == row['search_log_id']]['id']
        children_info = search_log_room_child[search_log_room_child['search_log_room_id'].isin(room_ids)]

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

huf_booking_types['detailed_category'] = huf_booking_types.apply(lambda row: detailed_categorize_booking(row, search_log_room, search_log_room_child), axis=1)
eur_booking_types['detailed_category'] = eur_booking_types.apply(lambda row: detailed_categorize_booking(row, search_log_room, search_log_room_child), axis=1)

# Adatok összekapcsolása
huf_upsell_data = pd.merge(huf_booking_types, booking_data, on='search_log_id', how='inner')
huf_upsell_data = pd.merge(huf_upsell_data, upsell_data, on='search_log_id', how='inner')

eur_upsell_data = pd.merge(eur_booking_types, booking_data, on='search_log_id', how='inner')
eur_upsell_data = pd.merge(eur_upsell_data, upsell_data, on='search_log_id', how='inner')

# Összesített táblázat
def create_upsell_summary(upsell_data):
    upsell_counts = upsell_data.groupby('detailed_category')['name'].value_counts().unstack(fill_value=0)
    upsell_counts = upsell_counts.T
    upsell_counts['Összesen'] = upsell_counts.sum(axis=1)
    return upsell_counts

# Bevétel szerinti csoportosítás
def categorize_upsell_by_revenue(upsell_data):
    upsell_revenue = upsell_data.groupby('name')['sum_price'].sum()

    _, bins = pd.qcut(upsell_revenue, 3, retbins=True, duplicates='drop')
    labels = ["Alacsony bevétel", "Közepes bevétel", "Magas bevétel"]
    labels = labels[:len(bins)-1]

    revenue_categories = pd.cut(upsell_revenue, bins, labels=labels, include_lowest=True)

    revenue_df = pd.DataFrame({'Upsell termék': upsell_revenue.index, 'Bevételi kategória': revenue_categories.values, 'Bevétel': upsell_revenue.values})
    return revenue_df

# Táblázat formázása
def format_summary_table(df, currency):
    # Táblázat transzponálása és rendezése
    df = df.sort_values('Összesen', ascending=False)

    # Nulla értékek eltávolítása
    df = df[df['Összesen'] > 0]

    # Index átnevezése
    df.index.name = 'Upsell termék'

    # Formázott megjelenítés
    if currency == 'HUF':
        formatted_df = df.style\
            .format('{:,.0f}')\
            .set_caption('Upsell termékek összesítése (HUF)')\
            .background_gradient(cmap='YlOrRd')
    else:
        formatted_df = df.style\
            .format('{:,.2f}')\
            .set_caption('Upsell termékek összesítése (EUR)')\
            .background_gradient(cmap='YlOrRd')

    return formatted_df

# Bevételi kategóriák formázása
def format_revenue_table(df, currency):
    # Rendezés bevétel szerint
    df = df.sort_values('Bevétel', ascending=False)

    # Nulla értékek eltávolítása
    df = df[df['Bevétel'] > 0]

    # Formázott megjelenítés
    if currency == 'HUF':
        formatted_df = df.style\
            .format({'Bevétel': '{:,.0f}'}, na_rep="-")\
            .set_caption('Bevételi kategóriák (HUF)')\
            .background_gradient(subset=['Bevétel'], cmap='YlOrRd')
    else:
        formatted_df = df.style\
            .format({'Bevétel': '{:,.2f}'}, na_rep="-")\
            .set_caption('Bevételi kategóriák (EUR)')\
            .background_gradient(subset=['Bevétel'], cmap='YlOrRd')

    return formatted_df

# Interaktív vizualizáció Plotly-val
def plot_interactive_upsell(huf_data, eur_data):
    huf_grouped = huf_data.groupby(['detailed_category', 'name'])['sum_price'].sum().reset_index()
    eur_grouped = eur_data.groupby(['detailed_category', 'name'])['sum_price'].sum().reset_index()

    fig = make_subplots(rows=1, cols=2, subplot_titles=('HUF Upsell bevételek', 'EUR Upsell bevételek'))

    for name in huf_grouped['name'].unique():
        huf_subset = huf_grouped[huf_grouped['name'] == name]
        fig.add_trace(go.Bar(x=huf_subset['detailed_category'], y=huf_subset['sum_price'], name=name, showlegend=True), row=1, col=1)

    for name in eur_grouped['name'].unique():
        eur_subset = eur_grouped[eur_grouped['name'] == name]
        fig.add_trace(go.Bar(x=eur_subset['detailed_category'], y=eur_subset['sum_price'], name=name, showlegend=False), row=1, col=2)

    fig.update_layout(title_text='Upsell bevételek kategóriánként és pénznemenként')
    fig.show()

# Statikus vizualizáció Matplotlibbel
def plot_static_upsell(huf_data, eur_data):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    huf_data.groupby(['detailed_category', 'name'])['sum_price'].sum().unstack().plot(kind='bar', ax=axes[0], title='HUF Upsell bevételek')
    eur_data.groupby(['detailed_category', 'name'])['sum_price'].sum().unstack().plot(kind='bar', ax=axes[1], title='EUR Upsell bevételek')

    plt.tight_layout()
    plt.show()

# Függvények meghívása
huf_upsell_summary = create_upsell_summary(huf_upsell_data)
eur_upsell_summary = create_upsell_summary(eur_upsell_data)

print("\n=== HUF Upsell Összesített Táblázat ===")
display(format_summary_table(huf_upsell_summary, 'HUF'))

print("\n=== EUR Upsell Összesített Táblázat ===")
display(format_summary_table(eur_upsell_summary, 'EUR'))

huf_revenue_categories = categorize_upsell_by_revenue(huf_upsell_data)
eur_revenue_categories = categorize_upsell_by_revenue(eur_upsell_data)

print("\n=== HUF Bevétel szerinti csoportosítás ===")
display(format_revenue_table(huf_revenue_categories, 'HUF'))

print("\n=== EUR Bevétel szerinti csoportosítás ===")
display(format_revenue_table(eur_revenue_categories, 'EUR'))

plot_interactive_upsell(huf_upsell_data, eur_upsell_data)
