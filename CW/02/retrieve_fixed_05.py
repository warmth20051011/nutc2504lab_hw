import requests
from test_02_embedding import get_embeddings

# =========================
# 基本設定
# =========================
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "CW02_fixed"
TOP_K = 3

# =========================
# 1. 查詢文字
# =========================
query_text = ["請問這份文件在說什麼？"]

# =========================
# 2. 轉成向量
# =========================
query_embeddings, VECTOR_SIZE = get_embeddings(query_text)
query_vector = query_embeddings[0]

print("Query VECTOR_SIZE =", len(query_vector))

# =========================
# 3. 呼叫 Qdrant Search API
# =========================
response = requests.post(
    f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search",
    json={
        "vector": query_vector,
        "limit": TOP_K,
        "with_payload": True
    }
)

results = response.json()["result"]

# =========================
# 4. 印出召回結果
# =========================
print("\n🔍 查詢問題：", query_text[0])
print("=" * 50)

for i, r in enumerate(results, 1):
    print(f"[結果 {i}] score = {r['score']}")
    print(r["payload"]["text"])
    print("-" * 50)

