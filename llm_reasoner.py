import os
import json
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

# Pydantic схема для жесткой валидации ответа от LLM
class HypothesisSchema(BaseModel):
    campaign_name: str
    filter_arpu_segment: Optional[str] = Field(default=None, pattern=r"^(LOW|MID|HIGH)$")
    filter_data_segment: Optional[str] = Field(default=None, pattern=r"^(NON_USER|LITE|HEAVY)$")
    filter_call_segment: Optional[str] = Field(default=None, pattern=r"^(LOW|MEDIUM|HIGH)$")
    filter_current_tariff: Optional[str] = None
    target_tariff: str = Field(pattern=r"^tariff_([1-9]|1[0-9]|2[0-1])$") # Ограничение 1-21
    reasoning: Optional[str] = None

class LLMReasoner:
    def __init__(self, model_name: str = "gpt-4o-mini", timeout: float = 15.0):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.model_name = model_name
        self.timeout = timeout
        self.client = None
        
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key, timeout=self.timeout)
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")

    def generate_campaign_hypotheses(
        self, 
        tariff_summary: List[Dict[str, Any]], 
        data_insights: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Генерирует валидные гипотезы с помощью Chain-of-Thought и Pydantic-проверки.
        """
        if not self.client:
            return self._fallback_hypotheses()

        system_prompt = (
            "You are an expert marketing strategist for Beeline. "
            "Analyze customer segments and tariffs to suggest optimal tariff migration campaigns. "
            "Always think step-by-step before producing JSON outputs."
        )

        user_prompt = f"""
        Available Tariffs Info:
        {json.dumps(tariff_summary, indent=2)}
        
        Data Insights / Historical Tariff Shifts:
        {json.dumps(data_insights, indent=2)}
        
        Task: Generate up to 12 distinct, high-ROI migration hypotheses.
        
        Return ONLY a JSON array of objects. Each object MUST strictly follow this JSON structure:
        [
          {{
            "reasoning": "Brief logic for why this segment will accept this target_tariff",
            "campaign_name": "Unique Name",
            "filter_arpu_segment": "LOW" | "MID" | "HIGH" | null,
            "filter_data_segment": "NON_USER" | "LITE" | "HEAVY" | null,
            "filter_call_segment": "LOW" | "MEDIUM" | "HIGH" | null,
            "filter_current_tariff": "tariff_1;tariff_2" | null,
            "target_tariff": "tariff_10"
          }}
        ]
        Note: target_tariff MUST be strictly between tariff_1 and tariff_21.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=1800
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:-3].strip()
            
            raw_json = json.loads(content)
            validated_hypotheses = []
            
            for item in raw_json:
                try:
                    # Валидируем через Pydantic
                    valid_item = HypothesisSchema(**item)
                    validated_hypotheses.append(valid_item.model_dump())
                except ValidationError as ve:
                    logger.warning(f"Discarding invalid hypothesis from LLM: {ve}")

            return validated_hypotheses if validated_hypotheses else self._fallback_hypotheses()

        except Exception as e:
            logger.error(f"LLM API Call failed or timed out: {e}. Switching to Fallback.")
            return self._fallback_hypotheses()

    def _fallback_hypotheses(self) -> List[Dict[str, Any]]:
        """Гарантированные валидные эвристические гипотезы."""
        hypotheses = []
        for arpu in ["LOW", "MID"]:
            for t_id in [3, 5, 8, 10, 12, 15]:
                hypotheses.append({
                    "campaign_name": f"Fallback Upsell {arpu} to tariff_{t_id}",
                    "filter_arpu_segment": arpu,
                    "filter_data_segment": None,
                    "filter_call_segment": None,
                    "filter_current_tariff": None,
                    "target_tariff": f"tariff_{t_id}",
                    "reasoning": "Fallback rule-based heuristic"
                })
        return hypotheses[:12]