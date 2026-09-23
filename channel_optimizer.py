from typing import Dict, Any

class ChannelOptimizer:
    # Ограничения и характеристики каналов из ТЗ
    CHANNELS = {
        "push": {"cost": 0.0, "eff": 0.50},
        "sms": {"cost": 4.0, "eff": 0.65},
        "digital_ads": {"cost": 22.0, "eff": 0.85},
        "call": {"cost": 160.0, "eff": 1.20}
    }

    @classmethod
    def select_best_channel(
        cls, 
        expected_arpu_lift: float, 
        avg_segment_arpu: float = 2000.0
    ) -> str:
        """
        Динамический подбор канала на основе прогнозируемого прироста выручки и стоимости.
        """
        best_channel = "push"
        max_net_margin = -float("inf")

        for ch_name, params in cls.CHANNELS.items():
            # Ожидаемый прирост с учетом коэффициента эффективности канала
            expected_revenue_gain = expected_arpu_lift * params["eff"]
            expected_cost = params["cost"]
            net_margin = expected_revenue_gain - expected_cost

            # Защитное правило: Не использовать звонки (call) для низких чеков
            if ch_name == "call" and avg_segment_arpu < 4000:
                continue

            if net_margin > max_net_margin:
                max_net_margin = net_margin
                best_channel = ch_name

        return best_channel