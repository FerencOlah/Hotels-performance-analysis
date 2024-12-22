import pandas as pd
import plotly.graph_objects as go
from hotel_1_data_cleaner import optimize_dataframes

def create_basic_funnel(dfs):
    """Alap funnel számítások"""
    # 1. Összes weboldal látogató
    total_visitors = dfs['website_daily_users']['user_count'].sum()

    # 2. Dátumválasztó használat
    datepicker_users = dfs['datepicker_daily_visitors']['user_count'].sum()

    # 3. Keresések száma (egyedi search_log_session_id alapján)
    searches = len(dfs['search_log']['search_log_session_id'].unique())

    # 4. Foglalások száma
    bookings = len(dfs['booking_data'])

    # Funnel adatok összeállítása
    funnel_data = {
        'Stage': ['Website Visits', 'Datepicker Usage', 'Searches', 'Bookings'],
        'Users': [total_visitors, datepicker_users, searches, bookings]
    }

    return pd.DataFrame(funnel_data)

def plot_enhanced_funnel(dfs):
    """Fejlesztett funnel vizualizáció Plotly-val"""
    funnel_df = create_basic_funnel(dfs)
    
    # Konverziós ráták számítása
    conv_rates = []
    for i in range(1, len(funnel_df)):
        rate = (funnel_df['Users'].iloc[i] / funnel_df['Users'].iloc[i-1]) * 100
        conv_rates.append(f"{rate:.1f}%")
    conv_rates.insert(0, "100%")

    # Színek és egyéb vizuális elemek beállítása
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f1c40f']
    
    # Plotly funnel diagram létrehozása
    fig = go.Figure()

    # Funnel hozzáadása
    fig.add_trace(go.Funnel(
        name='Conversion Funnel',
        y=funnel_df['Stage'],
        x=funnel_df['Users'],
        textposition="auto",
        textinfo="value+percent initial",
        opacity=0.85,
        marker={
            "color": colors,
            "line": {"width": [4, 4, 4, 4], "color": ["white", "white", "white", "white"]}
        },
        connector={
            "line": {
                "color": "rgb(63, 63, 63)",
                "width": 1,
                "dash": "solid"
            }
        }
    ))

    # Layout beállítások
    fig.update_layout(
        title={
            'text': "Hotel Conversion Funnel Analysis",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24, 'color': '#2c3e50'}
        },
        font=dict(
            family="Arial, sans-serif",
            size=14,
            color="#2c3e50"
        ),
        width=1000,
        height=600,
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(t=100, l=120, r=120, b=80)
    )

    # Konverziós ráták hozzáadása annotációként
    for i, (value, stage) in enumerate(zip(funnel_df['Users'], funnel_df['Stage'])):
        fig.add_annotation(
            x=value,
            y=stage,
            text=f"Conv. Rate: {conv_rates[i]}",
            showarrow=False,
            align="left",
            xanchor="left",
            xshift=10,
            font=dict(
                size=12,
                color="#34495e"
            ),
            bgcolor="#ecf0f1",
            bordercolor="#bdc3c7",
            borderwidth=1,
            borderpad=4,
            opacity=0.8
        )

    return fig

def print_funnel_summary(dfs):
    """Funnel összefoglaló statisztikák"""
    funnel_df = create_basic_funnel(dfs)

    print("\nFunnel Analysis Summary")
    print("=" * 50)
    print(f"📊 Total Website Visitors: {funnel_df['Users'].iloc[0]:,.0f}")
    print(f"📅 Datepicker Users: {funnel_df['Users'].iloc[1]:,.0f}")
    print(f"🔍 Total Searches: {funnel_df['Users'].iloc[2]:,.0f}")
    print(f"✅ Total Bookings: {funnel_df['Users'].iloc[3]:,.0f}")
    
    print("\nConversion Rates Analysis")
    print("=" * 50)
    
    # Alap konverziós ráták
    for i in range(1, len(funnel_df)):
        stage_from = funnel_df['Stage'].iloc[i-1]
        stage_to = funnel_df['Stage'].iloc[i]
        conv_rate = (funnel_df['Users'].iloc[i] / funnel_df['Users'].iloc[i-1]) * 100
        print(f"🔄 {stage_from} → {stage_to}: {conv_rate:.1f}%")
    
    print("\nAdditional Conversion Rates")
    print("=" * 50)
    
    # Datepicker → Bookings
    datepicker_to_booking = (funnel_df['Users'].iloc[3] / funnel_df['Users'].iloc[1]) * 100
    print(f"🔄 Datepicker Usage → Bookings: {datepicker_to_booking:.1f}%")
    
    # Website Visits → Bookings (teljes tölcsér konverzió)
    total_conversion = (funnel_df['Users'].iloc[3] / funnel_df['Users'].iloc[0]) * 100
    print(f"🔄 Website Visits → Bookings: {total_conversion:.1f}%")

def main():
    # Adatok betöltése
    input_dir = '/config/workspace/verseny_dataklub_morgens/data/raw/hotel_1'
    print("📂 Loading and optimizing data...")
    dfs = optimize_dataframes(input_dir)

    # Függvények futtatása
    print("🎨 Generating enhanced funnel analysis...")
    fig = plot_enhanced_funnel(dfs)
    fig.show()
    print_funnel_summary(dfs)

if __name__ == "__main__":
    main()
