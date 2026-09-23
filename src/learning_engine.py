import numpy as np

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
