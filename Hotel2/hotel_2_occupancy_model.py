import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
import joblib
import warnings
warnings.filterwarnings('ignore')

# Adatok betöltése
from hotel_2_data_cleaner import optimize_dataframes
dfs = optimize_dataframes('/config/workspace/verseny_dataklub_morgens/data/raw/hotel_2')

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

def prepare_analysis_data():
    # Datepicker adatok aggregálása
    daily_datepicker = dataframes['datepicker_daily_visitors'].groupby('date').agg({
        'user_count': 'sum',
        'session_count': 'sum'
    }).reset_index()

    # Occupancy adatok előkészítése
    daily_occupancy = dataframes['daily_occupancy'][
        dataframes['daily_occupancy']['recording_date'] == dataframes['daily_occupancy']['subject_date']
    ]
    daily_occupancy = daily_occupancy[['subject_date', 'fill_rate']]
    daily_occupancy = daily_occupancy.rename(columns={'subject_date': 'date'})

    # PPC költések
    ppc = dataframes['daily_ppc_budget']
    ppc['total_ppc_spend'] = ppc['daily_google_spend'] + ppc['daily_microsoft_spend'] + ppc['daily_meta_spend']

    # Adatok összefűzése
    merged_data = daily_datepicker.merge(daily_occupancy, on='date', how='left')
    merged_data = merged_data.merge(ppc[['date', 'total_ppc_spend']], on='date', how='left')

    # Dátumok konvertálása
    merged_data['date'] = pd.to_datetime(merged_data['date'])

    # Időbeli jellemzők
    merged_data['weekday'] = merged_data['date'].dt.dayofweek
    merged_data['is_weekend'] = merged_data['weekday'].isin([5, 6]).astype(int)
    merged_data['month'] = merged_data['date'].dt.month
    merged_data['day_of_month'] = merged_data['date'].dt.day

    # Alapvető jellemzők
    merged_data['ppc_per_user'] = merged_data['total_ppc_spend'] / merged_data['user_count'].replace(0, 1)
    merged_data['conversion_rate'] = merged_data['session_count'] / merged_data['user_count'].replace(0, 1)

    # Mozgóátlagok
    windows = [3, 7, 14]
    for window in windows:
        merged_data[f'fill_rate_ma{window}'] = merged_data['fill_rate'].rolling(window=window, min_periods=1).mean()
        merged_data[f'user_count_ma{window}'] = merged_data['user_count'].rolling(window=window, min_periods=1).mean()
        merged_data[f'ppc_ma{window}'] = merged_data['total_ppc_spend'].rolling(window=window, min_periods=1).mean()
        merged_data[f'fill_rate_std{window}'] = merged_data['fill_rate'].rolling(window=window, min_periods=1).std()

    # Lag jellemzők
    for lag in [7, 14]:
        merged_data[f'fill_rate_lag{lag}'] = merged_data['fill_rate'].shift(lag)
        merged_data[f'user_count_lag{lag}'] = merged_data['user_count'].shift(lag)
        merged_data[f'ppc_lag{lag}'] = merged_data['total_ppc_spend'].shift(lag)

    # Ciklikus jellemzők
    merged_data['weekday_sin'] = np.sin(2 * np.pi * merged_data['weekday']/7)
    merged_data['weekday_cos'] = np.cos(2 * np.pi * merged_data['weekday']/7)
    merged_data['month_sin'] = np.sin(2 * np.pi * merged_data['month']/12)
    merged_data['month_cos'] = np.cos(2 * np.pi * merged_data['month']/12)

    # Kiválasztott jellemzők
    selected_features = [
        'fill_rate', 'user_count', 'session_count', 'total_ppc_spend',
        'weekday', 'is_weekend', 'ppc_per_user', 'conversion_rate'
    ] + [f'fill_rate_ma{w}' for w in windows] + \
        [f'user_count_ma{w}' for w in windows] + \
        [f'ppc_ma{w}' for w in windows] + \
        [f'fill_rate_std{w}' for w in windows] + \
        [f'fill_rate_lag{lag}' for lag in [7, 14]] + \
        [f'user_count_lag{lag}' for lag in [7, 14]] + \
        [f'ppc_lag{lag}' for lag in [7, 14]] + \
        [f'weekday_sin', f'weekday_cos', f'month_sin', f'month_cos']

    final_data = merged_data[selected_features].copy()

    # Adataugmentáció
    augmented_data = []
    for idx in range(len(final_data)):
        row = final_data.iloc[idx]
        augmented_data.append(row)

        # Kis zajt adunk az adatokhoz
        noise = np.random.normal(0, 0.1, len(row))
        noisy_row = row + pd.Series(noise, index=row.index)
        augmented_data.append(noisy_row)

    final_data = pd.DataFrame(augmented_data)

    # Normalizálás
    scaler = StandardScaler()
    numeric_cols = [col for col in final_data.columns if col not in ['fill_rate', 'weekday', 'is_weekend']]
    final_data[numeric_cols] = scaler.fit_transform(final_data[numeric_cols])

    return final_data.dropna()
def create_models():
    """Modellek és hiperparaméter rácsok definiálása"""
    models = {
        'LinearRegression': {
            'model': LinearRegression(),
            'params': {}
        },
        'Ridge': {
            'model': Ridge(),
            'params': {
                'alpha': [0.1, 1.0, 10.0],
                'solver': ['auto', 'svd', 'cholesky']
            }
        },
        'ElasticNet': {
            'model': ElasticNet(max_iter=10000),  # Növelt iterációszám
            'params': {
                'alpha': [0.1, 0.5, 1.0],
                'l1_ratio': [0.1, 0.5, 0.9],
                'tol': [1e-4, 1e-3]  # Módosított tolerancia
            }
        },
        'RandomForest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'min_samples_split': [2, 5]
            }
        }
    }
    return models

def print_results(results):
    """Eredmények kiírása"""
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"RMSE: {metrics['RMSE']:.3f}")
        print(f"R2: {metrics['R2']:.3f}")
        print(f"MAE: {metrics['MAE']:.3f}")
        if 'Best_Params' in metrics and metrics['Best_Params'] != 'N/A':
            print(f"Legjobb paraméterek: {metrics['Best_Params']}")

def plot_results(y_test, predictions, results, title_suffix=''):
    """Eredmények vizualizációja"""
    plt.figure(figsize=(15, 8))

    # Egyszerűsített index kezelés
    x_values = range(len(y_test))

    plt.plot(x_values, y_test.values, label='Tényleges', linewidth=2)

    for name, pred in predictions.items():
        # Eltávolítjuk a suffix-et a megjelenítéshez
        display_name = name.split('_')[0]  # Csak az első részt vesszük (a suffix előttit)
        plt.plot(x_values, pred, label=f'{display_name} előrejelzés', alpha=0.7)

    plt.title(f'Tényleges vs Előrejelzett értékek {title_suffix}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('Megfigyelések')
    plt.ylabel('Érték')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Metrikák vizualizációja
    metrics_df = pd.DataFrame({k: {metric: v[metric] for metric in ['RMSE', 'R2', 'MAE']} 
                              for k, v in results.items()}).round(3)

    plt.figure(figsize=(12, 6))
    metrics_df.plot(kind='bar')
    plt.title(f'Modell teljesítmények összehasonlítása {title_suffix}')
    plt.xlabel('Metrikák')
    plt.ylabel('Érték')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def try_models(X_train, X_test, y_train, y_test, suffix=''):
    """Modellek kipróbálása és értékelése"""
    models = create_models()
    results = {}
    predictions = {}
    best_models = {}

    # Time Series Cross Validation
    tscv = TimeSeriesSplit(n_splits=5)

    for name, model_info in models.items():
        print(f"\nHangolás és tanítás: {name}")

        try:
            grid_search = GridSearchCV(
                model_info['model'],
                model_info['params'],
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )

            grid_search.fit(X_train, y_train)
            best_models[name] = grid_search.best_estimator_

            # Predikciók
            y_pred = grid_search.predict(X_test)
            predictions[f'{name}_{suffix}'] = y_pred

            # Metrikák számítása
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            results[f'{name}_{suffix}'] = {
                'RMSE': rmse,
                'R2': r2,
                'MAE': mae,
                'Best_Params': grid_search.best_params_
            }
        except Exception as e:
            print(f"Hiba történt a {name} modell futtatása során: {str(e)}")
            continue

    # Visszaadjuk a három szükséges objektumot
    return results, predictions, best_models
def main():
    # Adatok előkészítése
    print("Adatok előkészítése...")
    data = prepare_analysis_data()
    
    # Target és feature-ök szétválasztása
    X = data.drop('fill_rate', axis=1)
    y = data['fill_rate']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Feature selection
    print("\nJellemzők kiválasztása...")
    selector = SelectKBest(score_func=f_regression, k=20)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    selected_features = X.columns[selector.get_support()].tolist()
    print(f"Kiválasztott jellemzők: {selected_features}")
    
    # PCA
    print("\nPCA transzformáció...")
    pca = PCA(n_components=0.95)  # 95% of variance
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"PCA komponensek száma: {pca.n_components_}")
    
    # Modellek futtatása kiválasztott jellemzőkkel
    print("\nModellek futtatása kiválasztott jellemzőkkel...")
    results_selected, predictions_selected, best_models_selected = try_models(
        pd.DataFrame(X_train_selected, columns=selected_features),
        pd.DataFrame(X_test_selected, columns=selected_features),
        y_train, y_test, 'selected'
    )
    
    # Modellek futtatása PCA jellemzőkkel
    print("\nModellek futtatása PCA jellemzőkkel...")
    results_pca, predictions_pca, best_models_pca = try_models(
        X_train_pca, X_test_pca, y_train, y_test, 'pca'
    )
    
    # Eredmények megjelenítése és elemzése
    print("\nEredmények kiválasztott jellemzőkkel:")
    print_results(results_selected)
    plot_results(y_test, predictions_selected, results_selected, "- Kiválasztott jellemzők")
    
    print("\nEredmények PCA jellemzőkkel:")
    print_results(results_pca)
    plot_results(y_test, predictions_pca, results_pca, "- PCA jellemzők")
    
    # Legjobb modell kiválasztása és részletes elemzése
    best_method = 'Selected' if min(results_selected.values(), key=lambda x: x['RMSE'])['RMSE'] < \
                               min(results_pca.values(), key=lambda x: x['RMSE'])['RMSE'] else 'PCA'

    best_results = results_selected if best_method == 'Selected' else results_pca
    best_predictions = predictions_selected if best_method == 'Selected' else predictions_pca

    # A modell neve suffix nélkül
    best_model_name_full = min(best_results.items(), key=lambda x: x[1]['RMSE'])[0]
    best_model_name_base = best_model_name_full.split('_')[0]  # Eltávolítjuk a suffixet

    print(f"\nLegjobb modell: {best_model_name_full}")
    print(f"RMSE: {best_results[best_model_name_full]['RMSE']:.3f}")
    print(f"R2: {best_results[best_model_name_full]['R2']:.3f}")
    print(f"MAE: {best_results[best_model_name_full]['MAE']:.3f}")

    # Reziduálisok elemzése
    residuals = y_test - best_predictions[best_model_name_full]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(best_predictions[best_model_name_full], residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Előrejelzett értékek')
    plt.ylabel('Reziduálisok')
    plt.title('Reziduálisok vs Előrejelzett értékek')

    plt.subplot(1, 2, 2)
    sns.histplot(residuals, kde=True)
    plt.xlabel('Reziduálisok')
    plt.title('Reziduálisok eloszlása')
    plt.tight_layout()
    plt.show()

    # Legjobb modell mentése
    best_models = best_models_selected if best_method == 'Selected' else best_models_pca
    best_model = best_models[best_model_name_base]  # Itt használjuk a base nevet

    model_info = {
        'model': best_model,
        'selected_features': selected_features if best_method == 'Selected' else None,
        'pca': pca if best_method == 'PCA' else None,
        'method': best_method,
        'scaler': StandardScaler()  # Új scaler példány
    }

    joblib.dump(model_info, 'best_hotel_occupancy_model.joblib')
    print("\nLegjobb modell elmentve: 'best_hotel_occupancy_model.joblib'")

if __name__ == "__main__":
    main()