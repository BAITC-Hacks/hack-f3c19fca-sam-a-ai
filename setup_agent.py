import os
import subprocess
import sys

# 1. Создаем папку src
os.makedirs("src", exist_ok=True)

# 2. Файл src/data_analyzer.py
data_analyzer_code = '''import pandas as pd
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
'''

with open("src/data_analyzer.py", "w", encoding="utf-8") as f:
    f.write(data_analyzer_code)

# 3. Файл src/hypothesis_engine.py
hypothesis_engine_code = '''import pandas as pd

class HypothesisEngine:
    def __init__(self, data_analyzer):
        self.analyzer = data_analyzer

    def generate_candidate_hypotheses(self) -> list[dict]:
        hypotheses = []
        df_profiles = self.analyzer.profiles
        if df_profiles is None or df_profiles.empty:
            return hypotheses

        all_tariffs = [f"tariff_{i}" for i in range(1, 22)]
        arpu_segs = ['LOW', 'MID', 'HIGH']
        data_segs = ['NON_USER', 'LITE', 'HEAVY']

        for a_seg in arpu_segs:
            for d_seg in data_segs:
                sub_df = self.analyzer.get_segment_users(arpu_seg=a_seg, data_seg=d_seg)
                if len(sub_df) < 30:
                    continue

                current_tariff_counts = sub_df['current_tariff'].value_counts()
                top_current_tariffs = current_tariff_counts.head(3).index.tolist()

                for curr_t in top_current_tariffs:
                    curr_sub = sub_df[sub_df['current_tariff'] == curr_t]
                    if len(curr_sub) < 30:
                        continue

                    avg_arpu = curr_sub['predicted_arpu'].mean()

                    for target_t in all_tariffs:
                        if target_t == curr_t:
                            continue

                        prior = self.analyzer.get_historical_prior(curr_t, target_t)
                        
                        if prior['sample_size'] > 10 and prior['mean_uplift'] < 0:
                            continue

                        hyp_id = f"hyp_{a_seg}_{d_seg}_{curr_t}_to_{target_t}"
                        hypotheses.append({
                            "hypothesis_id": hyp_id,
                            "campaign_name": f"Upsell {curr_t} -> {target_t}",
                            "filter_arpu_segment": a_seg,
                            "filter_data_segment": d_seg,
                            "filter_current_tariff": curr_t,
                            "target_tariff": target_t,
                            "segment_size": len(curr_sub),
                            "avg_arpu": avg_arpu,
                            "prior_uplift": prior['mean_uplift'],
                            "prior_conv": prior['conversion_rate']
                        })

        hypotheses.sort(
            key=lambda h: h['segment_size'] * h['avg_arpu'] * max(h['prior_uplift'], 0.05),
            reverse=True
        )
        return hypotheses
'''

with open("src/hypothesis_engine.py", "w", encoding="utf-8") as f:
    f.write(hypothesis_engine_code)

# 4. Файл src/learning_engine.py
learning_engine_code = '''import numpy as np

class LearningEngine:
    CHANNEL_COSTS = {"push": 0, "sms": 4, "digital_ads": 22, "call": 160}
    CHANNEL_MULT = {"push": 0.50, "sms": 0.65, "digital_ads": 0.85, "call": 1.20}

    def __init__(self):
        self.tested_results = {}

    def select_best_channel(self, avg_arpu: float, conv_rate: float, uplift_rate: float) -> str:
        best_channel = "push"
        max_unit_profit = -float('inf')

        for ch, cost in self.CHANNEL_COSTS.items():
            mult = self.CHANNEL_MULT[ch]
            expected_rev = avg_arpu * (conv_rate * mult) * uplift_rate
            unit_profit = expected_rev - cost

            if unit_profit > max_unit_profit:
                max_unit_profit = unit_profit
                best_channel = ch

        return best_channel

    def process_pilot_output(self, hypothesis: dict, pilot_res: dict, channel_used: str) -> dict:
        n_cust = pilot_res.get("n_customers", 30)
        converted = pilot_res.get("converted", 0)
        
        raw_conv = converted / n_cust if n_cust > 0 else 0.0
        
        base_conv_est = raw_conv / self.CHANNEL_MULT.get(channel_used, 0.65)
        base_conv_est = min(max(base_conv_est, 0.01), 1.0)

        arpu_gain = pilot_res.get("arpu_gain", 0.0) 
        avg_arpu = hypothesis.get("avg_arpu", 2000.0)
        uplift_rate = arpu_gain / (avg_arpu + 1e-5) if avg_arpu > 0 else 0.05

        se = np.sqrt((base_conv_est * (1 - base_conv_est)) / max(n_cust, 1))
        lcb_conv = max(0.0, base_conv_est - 1.28 * se)

        opt_channel = self.select_best_channel(avg_arpu, lcb_conv, uplift_rate)
        opt_mult = self.CHANNEL_MULT[opt_channel]
        opt_cost = self.CHANNEL_COSTS[opt_channel]

        exp_unit_profit = (avg_arpu * (lcb_conv * opt_mult) * uplift_rate) - opt_cost

        result_entry = {
            "hypothesis_id": hypothesis["hypothesis_id"],
            "hypothesis": hypothesis,
            "sample_size": n_cust,
            "raw_conv": raw_conv,
            "lcb_conv": lcb_conv,
            "uplift_rate": uplift_rate,
            "recommended_channel": opt_channel,
            "expected_unit_profit": exp_unit_profit,
            "is_profitable": exp_unit_profit > 0
        }

        self.tested_results[hypothesis["hypothesis_id"]] = result_entry
        return result_entry

    def build_final_campaigns(self, remaining_budget: float, remaining_contacts: int) -> list[dict]:
        profitable = [res for res in self.tested_results.values() if res['is_profitable']]
        profitable.sort(key=lambda x: x['expected_unit_profit'], reverse=True)

        final_campaigns = []
        used_budget = 0.0
        used_contacts = 0

        for res in profitable:
            if len(final_campaigns) >= 10:
                break

            hyp = res['hypothesis']
            ch = res['recommended_channel']
            cost_per_contact = self.CHANNEL_COSTS[ch]

            target_size = min(hyp['segment_size'], 5000)

            max_by_contacts = remaining_contacts - used_contacts
            if cost_per_contact > 0:
                max_by_budget = int((remaining_budget - used_budget) // cost_per_contact)
            else:
                max_by_budget = 5000

            actual_size = min(target_size, max_by_contacts, max_by_budget)

            if actual_size < 50:
                continue

            campaign = {
                "campaign_name": hyp['campaign_name'],
                "filter_arpu_segment": hyp['filter_arpu_segment'],
                "filter_data_segment": hyp['filter_data_segment'],
                "filter_current_tariff": hyp['filter_current_tariff'],
                "target_tariff": hyp['target_tariff'],
                "channel": ch
            }

            final_campaigns.append(campaign)
            used_budget += actual_size * cost_per_contact
            used_contacts += actual_size

        return final_campaigns
'''

with open("src/learning_engine.py", "w", encoding="utf-8") as f:
    f.write(learning_engine_code)

# 5. Файл agent.py в корне
agent_code = '''from src.data_analyzer import DataAnalyzer
from src.hypothesis_engine import HypothesisEngine
from src.learning_engine import LearningEngine

class Agent:
    def __init__(self):
        self.analyzer = DataAnalyzer()
        self.learning_engine = LearningEngine()

    def act(self, env) -> list[dict]:
        profiles = getattr(env, 'customer_profile', None)
        self.analyzer.load_data(env_profiles=profiles)
        
        hypo_engine = HypothesisEngine(self.analyzer)
        candidates = hypo_engine.generate_candidate_hypotheses()

        for hyp in candidates[:15]:
            pilots_left = getattr(env, 'pilots_left', 0)
            if pilots_left <= 0:
                break

            try:
                pilot_res = env.run_pilot(
                    target_tariff=hyp['target_tariff'],
                    channel="sms",
                    n_customers=30,
                    filter_arpu_segment=hyp['filter_arpu_segment'],
                    filter_current_tariff=hyp['filter_current_tariff']
                )
                self.learning_engine.process_pilot_output(hyp, pilot_res, channel_used="sms")
            except Exception:
                continue

        rem_budget = getattr(env, 'remaining_budget', 100000.0)
        rem_contacts = getattr(env, 'remaining_contacts', 15000)

        return self.learning_engine.build_final_campaigns(rem_budget, rem_contacts)
'''

with open("agent.py", "w", encoding="utf-8") as f:
    f.write(agent_code)

print("✅ Все файлы и модули успешно созданы в папках src/ и agent.py!")