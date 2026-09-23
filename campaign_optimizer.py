from typing import List, Dict, Any
import pandas as pd

class CampaignOptimizer:
    def __init__(
        self, 
        max_campaigns: int = 10, 
        max_contacts_per_campaign: int = 5000,
        max_total_contacts: int = 15000,
        max_budget: float = 100000.0
    ):
        self.max_campaigns = max_campaigns
        self.max_contacts_per_campaign = max_contacts_per_campaign
        self.max_total_contacts = max_total_contacts
        self.max_budget = max_budget

    def build_final_plan(
        self, 
        evaluated_pilots: List[Dict[str, Any]], 
        customer_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Отбирает до 10 лучших кампаний с учетом бюджета и охватов.
        """
        # 1. Отфильтровываем прибыльные гипотезы (по lower bound или mean lift)
        valid_pilots = [
            p for p in evaluated_pilots 
            if p and p["lower_bound_lift"] > 0
        ]
        
        # Если не нашли с lower_bound > 0, берем хотя бы по mean_lift > 0
        if not valid_pilots:
            valid_pilots = [p for p in evaluated_pilots if p and p["mean_lift_per_user"] > 0]

        # 2. Сортируем по ожидаемому ROI/Прибыли
        valid_pilots.sort(key=lambda x: x["lower_bound_lift"], reverse=True)

        final_campaigns = []
        used_contacts = 0
        used_budget = 0.0

        for item in valid_pilots:
            if len(final_campaigns) >= self.max_campaigns:
                break

            hyp = item["hypothesis"]
            channel = item["channel_used"]
            
            # Рассчитываем допустимый размер аудитории под лимиты
            channel_cost = {"push": 0, "sms": 4, "digital_ads": 22, "call": 160}.get(channel, 0)
            
            # Лимит по контактам и бюджету
            remaining_contacts = self.max_total_contacts - used_contacts
            if remaining_contacts <= 0:
                break

            target_n = min(self.max_contacts_per_campaign, remaining_contacts)
            
            if channel_cost > 0:
                max_affordable = int((self.max_budget - used_budget) // channel_cost)
                target_n = min(target_n, max_affordable)

            if target_n <= 0:
                continue

            # Добавляем кампанию в итоговый план
            campaign_entry = {
                "campaign_name": hyp.get("campaign_name", f"Campaign_{len(final_campaigns)+1}"),
                "target_tariff": hyp["target_tariff"],
                "channel": channel
            }
            
            # Опциональные фильтры
            if hyp.get("filter_arpu_segment"):
                campaign_entry["filter_arpu_segment"] = hyp["filter_arpu_segment"]
            if hyp.get("filter_data_segment"):
                campaign_entry["filter_data_segment"] = hyp["filter_data_segment"]
            if hyp.get("filter_current_tariff"):
                campaign_entry["filter_current_tariff"] = hyp["filter_current_tariff"]

            final_campaigns.append(campaign_entry)
            
            used_contacts += target_n
            used_budget += target_n * channel_cost

        return final_campaigns
