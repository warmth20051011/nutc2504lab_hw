from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from openai import OpenAI
from deepeval.models import DeepEvalBaseLLM

class LlamaCppModel(DeepEvalBaseLLM):
    def __init__(
        self,
        base_url="https://ws-06.huannago.com/v1",
        model_name="gemma-3-27b-it"
    ):
        self.base_url = base_url
        self.model_name = model_name
       
    def load_model(self):
        # 建立 OpenAI 客戶端
        return OpenAI(
            api_key="NoNeed",
            base_url=self.base_url
        )
   
    def generate(self, prompt: str) -> str:
        client = self.load_model()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content
   
    async def a_generate(self, prompt: str) -> str:
        # 如果需要非同步版本，可以使用 AsyncOpenAI
        # 這裡為簡化示範，直接重用同步方法
        return self.generate(prompt)
   
    def get_model_name(self):
        return f"Llama.cpp ({self.model_name})"

custom_llm = LlamaCppModel(
    base_url="https://ws-06.huannago.com/v1",
    model_name="gemma-3-27b-it"
)


faithfulness = FaithfulnessMetric(
    threshold=0.1, #自行調整
    include_reason=True,
    model=custom_llm
)


# ===============================
# 建立測試案例
# ===============================
test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    retrieval_context=[
        "All customers are eligible for a 30 day full refund at no extra cost."
    ]
)



faithfulness.measure(test_case)
print("Faithfulness:", faithfulness.score)
print("Reason:", faithfulness.reason)

