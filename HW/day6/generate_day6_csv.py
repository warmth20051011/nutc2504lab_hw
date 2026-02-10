import csv
import random
import time

from openai import OpenAI
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    ContextualRelevancyMetric
)
from deepeval.test_case import LLMTestCase


# ======================
# 基本設定
# ======================
INPUT_CSV = "day6_HW_questions.csv"
OUTPUT_CSV = "day6_HW_results.csv"

SAMPLE_SIZE = 5          # ⭐ 每次隨機抽幾題
SLEEP_BEFORE_LLM = 1     # ⭐ 每次問 LLM 前休息秒數
SLEEP_AFTER_LLM = 1      # ⭐ 每次問完 LLM 後休息秒數


# ======================
# 自訂 Llama.cpp LLM
# ======================
class LlamaCppModel(DeepEvalBaseLLM):
    def __init__(
        self,
        base_url="https://ws-06.huannago.com/v1",
        model_name="google/gemma-3-27b-it"
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


# ======================
# 初始化 LLM
# ======================
custom_llm = LlamaCppModel()


# ======================
# Metrics
# ======================
faithfulness = FaithfulnessMetric(model=custom_llm)
answer_rel = AnswerRelevancyMetric(model=custom_llm)
context_recall = ContextualRecallMetric(model=custom_llm)
context_precision = ContextualPrecisionMetric(model=custom_llm)
context_relevancy = ContextualRelevancyMetric(model=custom_llm)


# ======================
# 預設 Context
# ======================
DEFAULT_CONTEXT = [
    "自來水公司依照國家飲用水水質標準進行淨水與消毒處理，以確保供水安全與品質。",
    "自來水相關業務包含水費帳單寄送、繳費方式、電子帳單申請及用水問題諮詢等服務。",
    "若民眾在用水、水質或帳單方面遇到疑問，可洽詢自來水公司客服或至營業所辦理。"
]

# ======================
# 讀取 CSV & 抽樣
# ======================
with open(INPUT_CSV, newline="", encoding="utf-8") as fin:
    rows = list(csv.DictReader(fin))

sampled_rows = random.sample(rows, min(SAMPLE_SIZE, len(rows)))

print("\n🎯 本次隨機抽到的題目 ID：")
print([row["q_id"] for row in sampled_rows])
print("=" * 80)


# ======================
# 開始評估
# ======================
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fout:
    fieldnames = [
        "q_id",
        "questions",
        "answer",
        "Faithfulness",
        "Answer_Relevancy",
        "Contextual_Recall",
        "Contextual_Precision",
        "Contextual_Relevancy"
    ]
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()

    for idx, row in enumerate(sampled_rows, start=1):
        print(f"\n🟦 第 {idx} 題開始")
        print("-" * 80)

        question = row["questions"]
        print("❓ Question:")
        print(question)

        # ---- 呼叫 LLM（慢慢來）----
        print("\n⏳ 等待 LLM 回答中...")
        time.sleep(SLEEP_BEFORE_LLM)

        try:
            answer = custom_llm.generate(question)
        except Exception as e:
            print("❌ LLM 發生錯誤，跳過此題")
            print(e)
            answer = "LLM Error"

        time.sleep(SLEEP_AFTER_LLM)

        print("\n🤖 LLM Answer:")
        print(answer)

        # ---- 建立測試案例 ----
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=answer,
            retrieval_context=DEFAULT_CONTEXT
        )

        # ---- 評估 ----
        faithfulness.measure(test_case)
        answer_rel.measure(test_case)
        context_recall.measure(test_case)
        context_precision.measure(test_case)
        context_relevancy.measure(test_case)

        print("\n📊 Metric Scores:")
        print(f"Faithfulness           : {faithfulness.score:.3f}")
        print(f"Answer Relevancy       : {answer_rel.score:.3f}")
        print(f"Contextual Recall      : {context_recall.score:.3f}")
        print(f"Contextual Precision   : {context_precision.score:.3f}")
        print(f"Contextual Relevancy   : {context_relevancy.score:.3f}")

        # ---- 寫入 CSV ----
        writer.writerow({
            "q_id": row["q_id"],
            "questions": question,
            "answer": answer,
            "Faithfulness": round(faithfulness.score, 3),
            "Answer_Relevancy": round(answer_rel.score, 3),
            "Contextual_Recall": round(context_recall.score, 3),
            "Contextual_Precision": round(context_precision.score, 3),
            "Contextual_Relevancy": round(context_relevancy.score, 3),
        })

        print("-" * 80)
        print(f"🟩 第 {idx} 題完成")
        time.sleep(1)


print("\n✅ 全部評估完成")
print(f"📄 結果輸出至：{OUTPUT_CSV}")

