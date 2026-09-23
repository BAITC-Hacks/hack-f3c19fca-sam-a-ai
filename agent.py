from src.data_analyzer import DataAnalyzer
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
