import pandas as pd
import os

def optimize_dataframes(input_dir):
    """
    Beolvassa és tisztítja az adatokat.

    Args:
        input_dir (str): Bemeneti könyvtár elérési útja

    Returns:
        dict: Tisztított DataFramek
    """
    # Adatok beolvasása külön DataFramekbe
    booking_data = pd.read_csv(f'{input_dir}/booking_data_hotel_3.csv', delimiter=';')
    daily_occupancy = pd.read_csv(f'{input_dir}/daily_occupancy_hotel_3.csv', delimiter=';')
    daily_ppc_budget = pd.read_csv(f'{input_dir}/daily_ppc_budget_hotel_3.csv', delimiter=',')
    datepicker_daily_visitors = pd.read_csv(f'{input_dir}/datepicker_daily_visitors_hotel_3.csv', delimiter=';', encoding='iso-8859-2')
    search_log = pd.read_csv(f'{input_dir}/search_log_hotel_3.csv', delimiter=';')
    search_log_room_child = pd.read_csv(f'{input_dir}/search_log_room_child_hotel_3.csv', delimiter=';')
    search_log_room = pd.read_csv(f'{input_dir}/search_log_room_hotel_3.csv', delimiter=';')
    search_log_room_offer = pd.read_csv(f'{input_dir}/search_log_room_offer_hotel_3.csv', delimiter=';')
    search_log_session = pd.read_csv(f'{input_dir}/search_log_session_hotel_3.csv', delimiter=';', encoding='iso-8859-2')
    upsell_data = pd.read_csv(f'{input_dir}/upsell_data_hotel_3.csv', delimiter=';')
    website_daily_users = pd.read_csv(f'{input_dir}/website_daily_users_hotel_3.csv', delimiter=';', encoding='iso-8859-2')

    # Booking data optimalizálása
    booking_data = booking_data.astype({
        'search_log_id': 'int32',
        'total_price_final': 'float32',
        'rooms_total_price': 'float32',
        'upsell_total_price': 'float32',
        'vouchers_total_price': 'int32',
        'loyalty_discount_total': 'float32',
        'redeemed_loyalty_points_total': 'float32'
    })

    # Daily occupancy optimalizálása
    daily_occupancy = daily_occupancy.astype({
        'recording_date': 'datetime64[ns]',
        'subject_date': 'datetime64[ns]',
        'fill_rate': 'float32'
    })

    # Daily PPC budget optimalizálása
    daily_ppc_budget = daily_ppc_budget.rename(columns={'Unnamed: 0': 'date'})
    daily_ppc_budget = daily_ppc_budget.astype({
        'date': 'datetime64[ns]',
        'daily_google_spend': 'int32',
        'daily_microsoft_spend': 'int32',
        'daily_meta_spend': 'int32'
    })

    # Datepicker daily visitors feldolgozása
    split_data_datepicker = datepicker_daily_visitors['utm_source_and_medium'].str.split('/', expand=True, n=1)
    if split_data_datepicker.shape[1] == 1:
        split_data_datepicker[1] = None
    split_data_datepicker.columns = ['utm_source', 'utm_medium']

    split_data_datepicker['utm_source'] = split_data_datepicker['utm_source'].str.strip().fillna('unknown')
    split_data_datepicker['utm_medium'] = split_data_datepicker['utm_medium'].str.strip().fillna('unknown')

    datepicker_daily_visitors = pd.concat([datepicker_daily_visitors, split_data_datepicker], axis=1)
    datepicker_daily_visitors = datepicker_daily_visitors.astype({
        'date': 'datetime64[ns]',
        'utm_source_and_medium': 'category',
        'utm_campaign': 'category',
        'user_count': 'int32',
        'session_count': 'int32',
        'utm_source': 'category',
        'utm_medium': 'category'
    })

    # Search log optimalizálása
    search_log = search_log.astype({
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

    # Search log room child optimalizálása
    search_log_room_child = search_log_room_child.astype({
        'id': 'int32',
        'search_log_room_id': 'int32',
        'age': 'int32',
        'baby_bed': 'int32'
    })

    # Search log room optimalizálása
    search_log_room = search_log_room.astype({
        'id': 'int32',
        'search_log_id': 'int32',
        'adults': 'int32',
        'children': 'int32',
        'picked_price': 'float32',
        'picked_room': 'category'
    })

    # Search log room offer optimalizálása
    search_log_room_offer = search_log_room_offer.astype({
        'id': 'int32',
        'search_log_id': 'int32',
        'search_log_room_id': 'int32',
        'room_code': 'category',
        'room_price_min': 'float32',
        'room_price_max': 'float32'
    })

    # Search log session optimalizálása
    search_log_session = search_log_session.astype({
        'id': 'int32',
        'uuid': 'string',
        'session_id': 'int32',
        'utm_source': 'category',
        'utm_medium': 'category',
        'utm_campaign': 'category'
    })

    # Upsell data optimalizálása
    upsell_data = upsell_data.astype({
        'search_log_id': 'int32',
        'upsell_type': 'int32',
        'name': 'category',
        'unit_price': 'float32',
        'pieces': 'int32',
        'sum_price': 'float32'
    })

    # Website daily users feldolgozása
    split_data = website_daily_users['utm_source_and_medium'].str.split('/', expand=True, n=1)
    if split_data.shape[1] == 1:
        split_data[1] = None
    split_data.columns = ['utm_source', 'utm_medium']

    split_data['utm_source'] = split_data['utm_source'].fillna('unknown')
    split_data['utm_medium'] = split_data['utm_medium'].fillna('unknown')

    website_daily_users = pd.concat([website_daily_users, split_data], axis=1)
    website_daily_users = website_daily_users.astype({
        'date': 'datetime64[ns]',
        'utm_source_and_medium': 'category',
        'utm_campaign': 'category',
        'user_count': 'int32',
        'session_count': 'int32',
        'utm_source': 'category',
        'utm_medium': 'category'
    })

    # Hiányzó értékek kezelése minden DataFrameben
    all_dataframes = [
        booking_data, daily_occupancy, daily_ppc_budget, 
        datepicker_daily_visitors, search_log, search_log_room_child,
        search_log_room, search_log_room_offer, search_log_session,
        upsell_data, website_daily_users
    ]

    for df in all_dataframes:
        if 'category' in df.dtypes.values:
            for col in df.select_dtypes(include=['category']).columns:
                if 0 not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories([0])
        df.fillna(0, inplace=True)

    return {
        'booking_data': booking_data,
        'daily_occupancy': daily_occupancy,
        'daily_ppc_budget': daily_ppc_budget,
        'datepicker_daily_visitors': datepicker_daily_visitors,
        'search_log': search_log,
        'search_log_room_child': search_log_room_child,
        'search_log_room': search_log_room,
        'search_log_room_offer': search_log_room_offer,
        'search_log_session': search_log_session,
        'upsell_data': upsell_data,
        'website_daily_users': website_daily_users
    }