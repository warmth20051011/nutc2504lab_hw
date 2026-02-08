from qdrant_client import QdrantClient
from test_02_embedding import get_embeddings

# =========================
# 基本設定
# =========================
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "CW02_table_semantic"

QUERY = "AI 邊緣運算的技術發展趨勢是什麼？"

# =========================
#  Query embedding
# =========================
query_vector, VECTOR_SIZE = get_embeddings([QUERY])
query_vector = query_vector[0]


# =========================
#  建立 Qdrant client
# =========================
client = QdrantClient(url=QDRANT_URL)

# =========================
#  搜尋
# =========================
results = client.query_points(
    collection_name=COLLECTION_NAME,
    prefetch=[],
    query=query_vector,
    limit=3
).points

# =========================
#  印出結果
# =========================
print("🔍 查詢問題：", QUERY)
print("\n📋 表格召回結果：\n")

for i, r in enumerate(results, 1):
    print(f"--- Result {i} ---")
    print("Score:", r.score)
    print("Text:", r.payload["text"])
    print()
