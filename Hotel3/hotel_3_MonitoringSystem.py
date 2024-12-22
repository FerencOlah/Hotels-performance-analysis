import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import matplotlib.pyplot as plt
import json
import logging
from typing import Dict, List, Tuple, Optional

class HotelMonitoringSystem:
    def __init__(self, data_path: str, model_name: str, baseline_rmse: float = 8.350):
        """
        Monitoring rendszer inicializálása

        Args:
            data_path: Az adatok elérési útja
            model_name: A modell neve
            baseline_rmse: Az eredeti modell RMSE értéke
        """
        self.model_name = model_name
        self.baseline_rmse = baseline_rmse
        self.monitoring_data = []
        self.dataframes = {}

        # Logging beállítása
        logging.basicConfig(
            filename=f'hotel_monitoring_{model_name}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def monitor_daily_metrics(self) -> Dict:
        """
        Napi metrikák monitorozása
        """
        daily_metrics = {}

        try:
            # Mivel most nincs valós adat, csak egy példa metrikát adunk vissza
            current_date = datetime.now().strftime('%Y-%m-%d')
            daily_metrics = {
                'daily_bookings': {current_date: 0},
                'occupancy_rate': {current_date: 0.0},
                'website_visitors': {current_date: 0}
            }

        except Exception as e:
            logging.error(f"Error in monitor_daily_metrics: {str(e)}")

        return daily_metrics

    def check_data_quality(self) -> Dict:
        """
        Adatminőség ellenőrzése
        """
        quality_report = {
            'example_metrics': {
                'missing_values': 0,
                'row_count': 0,
                'duplicate_rows': 0,
                'memory_usage': 0.0
            }
        }
        return quality_report

    def create_monitoring_dashboard(self, predictions: np.ndarray, actuals: np.ndarray) -> None:
        """
        Monitoring dashboard létrehozása
        """
        fig = plt.figure(figsize=(15, 10))

        # Előrejelzések vs. Tényleges értékek
        plt.subplot(2, 2, 1)
        plt.scatter(predictions, actuals, alpha=0.5)
        plt.plot([min(predictions), max(predictions)], [min(predictions), max(predictions)], 'r--')
        plt.xlabel('Előrejelzett értékek')
        plt.ylabel('Tényleges értékek')
        plt.title('Előrejelzések vs. Tényleges értékek')

        # Reziduálisok
        residuals = actuals - predictions
        plt.subplot(2, 2, 2)
        plt.hist(residuals, bins=20)
        plt.xlabel('Reziduálisok')
        plt.ylabel('Gyakoriság')
        plt.title('Reziduálisok eloszlása')

        # Reziduálisok vs. Előrejelzett értékek
        plt.subplot(2, 2, 3)
        plt.scatter(predictions, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Előrejelzett értékek')
        plt.ylabel('Reziduálisok')
        plt.title('Reziduálisok vs. Előrejelzett értékek')

        # Metrikák időbeli alakulása
        if self.monitoring_data:
            plt.subplot(2, 2, 4)
            metrics = [d['model_metrics']['rmse'] for d in self.monitoring_data]
            plt.plot(range(len(metrics)), metrics, label='RMSE')
            plt.xlabel('Monitoring időpontok')
            plt.ylabel('Érték')
            plt.title('Metrikák időbeli alakulása')
            plt.legend()

        plt.tight_layout()
        plt.savefig(f'monitoring_dashboard_{self.model_name}_{datetime.now().strftime("%Y%m%d")}.png')
        plt.close()

    def generate_full_report(self, predictions: np.ndarray, actuals: np.ndarray) -> Dict:
        """
        Teljes monitoring jelentés generálása
        """
        # Konvertáljuk a numpy értékeket natív Python típusokká
        predictions = [float(x) for x in predictions]
        actuals = [float(x) for x in actuals]

        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_metrics': {
                'rmse': float(np.sqrt(mean_squared_error(actuals, predictions))),
                'mae': float(mean_absolute_error(actuals, predictions)),
                'r2': float(r2_score(actuals, predictions))
            },
            'data_quality': self.check_data_quality(),
            'daily_metrics': self.monitor_daily_metrics()
        }

        # Jelentés mentése
        self.monitoring_data.append(report)

        # JSON formátumban mentés
        with open(f'monitoring_report_{self.model_name}_{datetime.now().strftime("%Y%m%d")}.json', 'w') as f:
            json.dump(report, f, indent=4)

        # Dashboard létrehozása
        self.create_monitoring_dashboard(np.array(predictions), np.array(actuals))

        return report

# Példa használat:
if __name__ == "__main__":
    # Monitoring rendszer inicializálása
    monitor = HotelMonitoringSystem(
        data_path='/config/workspace/verseny_dataklub_morgens/data/raw/hotel_3',
        model_name="Hotel3_LinearRegression",
        baseline_rmse=8.350
    )

    # Példa predikciók és valós értékek
    predictions = np.array([60, 65, 70, 75])
    actuals = np.array([62, 63, 71, 73])

    # Teljes jelentés generálása
    report = monitor.generate_full_report(predictions, actuals)

    print("Monitoring report and dashboard generated successfully!")
    print(f"RMSE: {report['model_metrics']['rmse']:.3f}")
    print(f"MAE: {report['model_metrics']['mae']:.3f}")
    print(f"R2: {report['model_metrics']['r2']:.3f}")