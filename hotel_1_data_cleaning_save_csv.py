import pandas as pd
import os
import zipfile
from datetime import datetime

# Célmappa létrehozása, ha még nem létezik
output_dir = '/config/workspace/verseny_dataklub_morgens/data/clean/hotel_1'
os.makedirs(output_dir, exist_ok=True)

# Adattípusok optimalizálása és hiányzó értékek kezelése
def optimize_dataframes():
    dataframes = {
        'booking_data_hotel_1': pd.read_csv('/config/workspace/verseny_dataklub_morgens/data/raw/hotel_1/booking_data_hotel_1.csv', delimiter=';'),
        'daily_occupancy_hotel_1': pd.read_csv('/config/workspace/verseny_dataklub_morgens/data/raw/hotel_1/daily_occupancy_hotel_1.csv', delimiter=';'),
        'daily_ppc_budget_hotel_1': pd.read_csv('/config/workspace/verseny_dataklub_morgens/data/raw/hotel_1/daily_ppc_budget_hotel_1.csv', delimiter=','),
        'datepicker_daily_visitors_hotel_1': pd.read_csv('/config/workspace/verseny_dataklub_morgens/data/raw/hotel_1/datepicker_daily_visitors_hotel_1.csv', delimiter=';'),
        'search_log_hotel_1': pd.read_csv('/config/workspace/verseny_dataklub_morgens/data/raw/hotel_1/search_log_hotel_1.csv', delimiter=';'),
        'search_log_room_child_hotel_1': pd.read_csv('/config/workspace/verseny_dataklub_morgens/data/raw/hotel_1/search_log_room_child_hotel_1.csv', delimiter=';'),
        'search_log_room_hotel_1': pd.read_csv('/config/workspace/verseny_dataklub_morgens/data/raw/hotel_1/search_log_room_hotel_1.csv', delimiter=';'),
        'search_log_room_offer_hotel_1': pd.read_csv('/config/workspace/verseny_dataklub_morgens/data/raw/hotel_1/search_log_room_offer_hotel_1.csv', delimiter=';'),
        'search_log_session_hotel_1': pd.read_csv('/config/workspace/verseny_dataklub_morgens/data/raw/hotel_1/search_log_session_hotel_1.csv', delimiter=';'),
        'upsell_data_hotel_1': pd.read_csv('/config/workspace/verseny_dataklub_morgens/data/raw/hotel_1/upsell_data_hotel_1.csv', delimiter=';'),
        'website_daily_users_hotel_1': pd.read_csv('/config/workspace/verseny_dataklub_morgens/data/raw/hotel_1/website_daily_users_hotel_1.csv', delimiter=';')
            }

    # Optimalizálások
    dataframes['booking_data_hotel_1'] = dataframes['booking_data_hotel_1'].astype({
        'search_log_id': 'int32',
        'total_price_final': 'float32',
        'rooms_total_price': 'float32',
        'upsell_total_price': 'float32',
        'vouchers_total_price': 'int32',
        'loyalty_discount_total': 'float32',
        'redeemed_loyalty_points_total': 'float32'
    })

    dataframes['daily_occupancy_hotel_1'] = dataframes['daily_occupancy_hotel_1'].astype({
        'recording_date': 'datetime64[ns]',
        'subject_date': 'datetime64[ns]',
        'fill_rate': 'float32'
    })

    dataframes['daily_ppc_budget_hotel_1'] = dataframes['daily_ppc_budget_hotel_1'].rename(columns={'Unnamed: 0': 'date'})
    dataframes['daily_ppc_budget_hotel_1'] = dataframes['daily_ppc_budget_hotel_1'].astype({
        'date': 'datetime64[ns]',
        'daily_google_spend': 'int32',
        'daily_microsoft_spend': 'int32',
        'daily_meta_spend': 'int32'
    })

    # Datepicker daily visitors feldolgozása
    split_data_datepicker = dataframes['datepicker_daily_visitors_hotel_1']['utm_source_and_medium'].str.split('/', expand=True, n=1)
    if split_data_datepicker.shape[1] == 1:
        # Ha csak egy oszlop van, adjunk hozzá egy másodikat
        split_data_datepicker[1] = None
    split_data_datepicker.columns = ['utm_source', 'utm_medium']

    # Ha nincs két rész, akkor unknown értéket adjunk
    split_data_datepicker['utm_source'] = split_data_datepicker['utm_source'].str.strip().fillna('unknown')
    split_data_datepicker['utm_medium'] = split_data_datepicker['utm_medium'].str.strip().fillna('unknown')

    # Hozzáadjuk az új oszlopokat az eredeti DataFrame-hez
    dataframes['datepicker_daily_visitors_hotel_1'] = pd.concat([dataframes['datepicker_daily_visitors_hotel_1'], split_data_datepicker], axis=1)

    # Adattípusok beállítása
    dataframes['datepicker_daily_visitors_hotel_1'] = dataframes['datepicker_daily_visitors_hotel_1'].astype({
        'date': 'datetime64[ns]',
        'utm_source_and_medium': 'category',
        'utm_campaign': 'category',
        'user_count': 'int32',
        'session_count': 'int32',
        'utm_source': 'category',
        'utm_medium': 'category'
    })

    dataframes['search_log_hotel_1'] = dataframes['search_log_hotel_1'].astype({
        'id': 'int32',
        'search_log_session_id': 'int32',
        'utc_datetime': 'datetime64[ns]',
        'lang_code': 'category',
        'currency': 'category',
        'arrival': 'datetime64[ns]',
        'departure': 'datetime64[ns]',
        'days': 'int32',
        'nights': 'int32',
        'adults': 'int32',
        'children': 'int32',
        'conversion': 'float32',
        'total_price_final': 'float32'
    })

    dataframes['search_log_room_child_hotel_1'] = dataframes['search_log_room_child_hotel_1'].astype({
        'id': 'int32',
        'search_log_room_id': 'int32',
        'age': 'int32',
        'baby_bed': 'int32'
    })

    dataframes['search_log_room_hotel_1'] = dataframes['search_log_room_hotel_1'].astype({
        'id': 'int32',
        'search_log_id': 'int32',
        'adults': 'int32',
        'children': 'int32',
        'picked_price': 'float32',
        'picked_room': 'category'
    })

    dataframes['search_log_room_offer_hotel_1'] = dataframes['search_log_room_offer_hotel_1'].astype({
        'id': 'int32',
        'search_log_id': 'int32',
        'search_log_room_id': 'int32',
        'room_code': 'category',
        'room_price_min': 'float32',
        'room_price_max': 'float32'
    })

    dataframes['search_log_session_hotel_1'] = dataframes['search_log_session_hotel_1'].astype({
        'id': 'int32',
        'uuid': 'string',
        'session_id': 'int32',
        'utm_source': 'category',
        'utm_medium': 'category',
        'utm_campaign': 'category'
    })

    dataframes['upsell_data_hotel_1'] = dataframes['upsell_data_hotel_1'].astype({
        'search_log_id': 'int32',
        'upsell_type': 'int32',
        'name': 'category',
        'unit_price': 'float32',
        'pieces': 'int32',
        'sum_price': 'float32'
    })

    # Új oszlopok hozzáadása a website_daily_users_hotel_1 dataframe-hez
    split_data = dataframes['website_daily_users_hotel_1']['utm_source_and_medium'].str.split('/', expand=True, n=1)
    if split_data.shape[1] == 1:
        # Ha csak egy oszlop van, adjunk hozzá egy másodikat
        split_data[1] = None
    split_data.columns = ['utm_source', 'utm_medium']

    # Ha nincs két rész, akkor unknown értéket adjunk
    split_data['utm_source'] = split_data['utm_source'].fillna('unknown')
    split_data['utm_medium'] = split_data['utm_medium'].fillna('unknown')

    dataframes['website_daily_users_hotel_1'] = pd.concat([dataframes['website_daily_users_hotel_1'], split_data], axis=1)
    dataframes['website_daily_users_hotel_1'] = dataframes['website_daily_users_hotel_1'].astype({
        'date': 'datetime64[ns]',
        'utm_source_and_medium': 'category',
        'utm_campaign': 'category',
        'user_count': 'int32',
        'session_count': 'int32',
        'utm_source': 'category',
        'utm_medium': 'category'
    })

    # Hiányzó értékek kezelése
    for name, df in dataframes.items():
        if 'category' in df.dtypes.values:
            for col in df.select_dtypes(include=['category']).columns:
                if 0 not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories([0])
        df.fillna(0, inplace=True)  # Példa hiányzó értékek kezelésére

    # Optimalizált adatok mentése csv formátumba
    for name, df in dataframes.items():
        df.to_csv(os.path.join(output_dir, f'{name}.csv'), index=False)

# Optimalizált adatok mentése
optimize_dataframes()

# Created/Modified files during execution:
for file_name in [
    "booking_data_hotel_1.csv",
    "daily_occupancy_hotel_1.csv",
    "daily_ppc_budget_hotel_1.csv",
    "datepicker_daily_visitors_hotel_1.csv",
    "search_log_hotel_1.csv",
    "search_log_room_child_hotel_1.csv",
    "search_log_room_hotel_1.csv",
    "search_log_room_offer_hotel_1.csv",
    "search_log_session_hotel_1.csv",
    "upsell_data_hotel_1.csv",
    "website_daily_users_hotel_1.csv"
]:
    print(file_name)

# ZIP fájl neve dátummal
zip_filename = os.path.join(output_dir, f'hotel_1_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip')

# CSV fájlok tömörítése ZIP-be
with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file_name in [
        "booking_data_hotel_1.csv",
        "daily_occupancy_hotel_1.csv",
        "daily_ppc_budget_hotel_1.csv",
        "datepicker_daily_visitors_hotel_1.csv",
        "search_log_hotel_1.csv",
        "search_log_room_child_hotel_1.csv",
        "search_log_room_hotel_1.csv",
        "search_log_room_offer_hotel_1.csv",
        "search_log_session_hotel_1.csv",
        "upsell_data_hotel_1.csv",
        "website_daily_users_hotel_1.csv"
    ]:
        file_path = os.path.join(output_dir, file_name)
        zipf.write(file_path, file_name)

print(f"\nZIP fájl létrehozva: {zip_filename}")