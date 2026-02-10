from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from openai import OpenAI
from deepeval.models import DeepEvalBaseLLM

# ===============================
# 自訂 LLM
# ===============================
class LlamaCppModel(DeepEvalBaseLLM):
    def __init__(
        self,
        base_url="https://ws-06.huannago.com/v1",
        model_name="gemma-3-27b-it"
    ):
        self.base_url = base_url
        self.model_name = model_name

    def load_model(self):
        return OpenAI(
            api_key="NoNeed",
            base_url=self.base_url
        )

    def generate(self, prompt: str) -> str:
        client = self.load_model()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return f"Llama.cpp ({self.model_name})"


custom_llm = LlamaCppModel(
    base_url="https://ws-06.huannago.com/v1",
    model_name="gemma-3-27b-it"
)

# ===============================
# 建立 Contextual Relevancy 指標
# ===============================
contextual_relevancy = ContextualRelevancyMetric(
    threshold=0.5,
    include_reason=True,
    model=custom_llm
)

# ===============================
# 建立測試案例
# ===============================
test_case = LLMTestCase(
    input="如果鞋子尺寸不合怎麼辦？",
    actual_output="我們提供 30 天內免費退費服務。",
    retrieval_context=[
        "所有顧客皆可在 30 天內申請全額退費，且不需額外費用。"
    ]
)

# ===============================
# 執行評估
# ===============================
contextual_relevancy.measure(test_case)
print("Contextual Relevancy:", contextual_relevancy.score)
print("Reason:", contextual_relevancy.reason)

