import pandas as pd
import numpy as np
from pathlib import Path

class DataAnalyzer:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.profiles = None
        self.tariff_dict = None
        self.history_transitions = None
        self.prior_matrix = {}

    def load_data(self, env_profiles=None):
        if env_profiles is not None:
            self.profiles = env_profiles.copy()
        else:
            profiles_path = self.data_dir / "customer_profile.csv"
            if profiles_path.exists():
                self.profiles = pd.read_csv(profiles_path)
        
        self._preprocess_profiles()

        dict_path = self.data_dir / "dict_tariff.csv"
        if dict_path.exists():
            self.tariff_dict = pd.read_csv(dict_path)

        history_path = self.data_dir / "change_tariff.csv"
        if history_path.exists():
            self.history_transitions = pd.read_csv(history_path)
            self._analyze_history()

    def _preprocess_profiles(self):
        if self.profiles is None or self.profiles.empty:
            return
        if 'predicted_arpu' not in self.profiles.columns and 'ARPU_3m_avg' in self.profiles.columns:
            self.profiles['predicted_arpu'] = self.profiles['ARPU_3m_avg']

    def _analyze_history(self):
        df = self.history_transitions.copy()
        if df.empty:
            return

        df['rel_uplift'] = (df['arpu_after'] - df['arpu_before']) / (df['arpu_before'] + 1e-5)
        
        grouped = df.groupby(['current_tariff', 'target_tariff']).agg(
            sample_size=('rel_uplift', 'count'),
            mean_uplift=('rel_uplift', 'mean'),
            positive_rate=('rel_uplift', lambda x: (x > 0).mean())
        ).reset_index()

        for _, row in grouped.iterrows():
            key = (row['current_tariff'], row['target_tariff'])
            self.prior_matrix[key] = {
                "sample_size": int(row['sample_size']),
                "mean_uplift": float(row['mean_uplift']),
                "conversion_rate": float(row['positive_rate'])
            }

    def get_historical_prior(self, current_tariff: str, target_tariff: str) -> dict:
        key = (current_tariff, target_tariff)
        return self.prior_matrix.get(key, {"sample_size": 0, "mean_uplift": 0.05, "conversion_rate": 0.1})

    def get_segment_users(self, arpu_seg=None, data_seg=None, call_seg=None, current_tariff=None) -> pd.DataFrame:
        if self.profiles is None:
            return pd.DataFrame()
        df = self.profiles.copy()
        if arpu_seg:
            df = df[df['arpu_segment'] == arpu_seg]
        if data_seg:
            df = df[df['data_segment'] == data_seg]
        if call_seg and 'call_segment' in df.columns:
            df = df[df['call_segment'] == call_seg]
        if current_tariff:
            tariffs = current_tariff.split(';')
            df = df[df['current_tariff'].isin(tariffs)]
        return df
