import math
import numpy as np
from typing import Dict, Any, List

class PilotManager:
    def __init__(self, max_pilots: int = 20):
        self.max_pilots = max_pilots
        self.pilots_run = 0

    def evaluate_hypothesis(
        self, 
        env, 
        hypothesis: Dict[str, Any], 
        sample_size: int = 100, 
        channel: str = "push"
    ) -> Dict[str, Any]:
        """
        Запускает пилот для одной гипотезы и рассчитывает метрики (ARPU lift, CI, ROI).
        """
        if env.pilots_left <= 0 or self.pilots_run >= self.max_pilots:
            return None

        # Выполняем пилот через env
        pilot_result = env.run_pilot(
            target_tariff=hypothesis["target_tariff"],
            channel=channel,
            n_customers=sample_size,
            filter_arpu_segment=hypothesis.get("filter_arpu_segment"),
            filter_current_tariff=hypothesis.get("filter_current_tariff")
        )
        self.pilots_run += 1

        # Извлекаем и рассчитываем эффекты
        # pilot_result возвращает статистику переходов и затрат
        n = pilot_result.get("n_contacts", sample_size)
        total_lift = pilot_result.get("net_arpu_lift", 0)
        mean_lift = total_lift / n if n > 0 else 0
        
        # Оценка неопределенности (Standard Error & Lower Bound Confidence Interval)
        # Упрощенная оценка Variance на основе среднего
        std_dev = pilot_result.get("std_arpu_lift", abs(mean_lift) * 0.5 + 1.0)
        sem = std_dev / math.sqrt(n) if n > 0 else 0
        
        # Conservative estimate (Lower Bound 90% CI)
        lower_bound_lift = mean_lift - 1.28 * sem

        return {
            "hypothesis": hypothesis,
            "sample_size": n,
            "mean_lift_per_user": mean_lift,
            "lower_bound_lift": lower_bound_lift,
            "total_net_profit": total_lift,
            "channel_used": channel,
            "raw_pilot_output": pilot_result
        }
