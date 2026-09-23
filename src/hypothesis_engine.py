import pandas as pd

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
