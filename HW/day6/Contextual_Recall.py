from deepeval.metrics import ContextualRecallMetric
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
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return f"Llama.cpp ({self.model_name})"


# ===============================
# 建立 custom_llm
# ===============================
custom_llm = LlamaCppModel(
    base_url="https://ws-06.huannago.com/v1",
    model_name="gemma-3-27b-it"
)

# ===============================
# Contextual Recall Metric
# ===============================
contextual_recall = ContextualRecallMetric(
    threshold=0.3,
    include_reason=True,
    model=custom_llm
)

# ===============================
# Test case
# ===============================
test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    expected_output="You are eligible for a 30 day full refund at no extra cost.",
    retrieval_context=[
        "All customers are eligible for a 30 day full refund at no extra cost."
    ]
)
# ===============================
# Run
# ===============================
contextual_recall.measure(test_case)
print("Contextual Recall:", contextual_recall.score)
print("Reason:", contextual_recall.reason)

