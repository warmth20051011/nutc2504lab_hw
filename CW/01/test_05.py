import requests
from test_02_embedding import get_embeddings

QDRANT_URL = "http://localhost:6333"
TOP_K = 3

# 查詢文字
query_text = ["請假需要注意什麼？"]

# 取得 query 向量
query_embeddings, VECTOR_SIZE = get_embeddings(query_text)
query_vector = query_embeddings[0]

print("Query VECTOR_SIZE：", len(query_vector))

collections = [
    "CW01_cosine",
    "CW01_dot",
    "CW01_euclidean"
]

for col in collections:
    print(f"\n=== 搜尋結果：{col} ===")

    response = requests.post(
        f"{QDRANT_URL}/collections/{col}/points/search",
        json={
            "vector": query_vector,
            "limit": TOP_K,
            "with_payload": True 
        }
    )

    results = response.json()["result"]

    results = response.json().get("result", [])

    for r in results:
        score = r.get("score")
        payload = r.get("payload")
        text = payload.get("text") if payload else None

        print(f"score={score:.4f} | {text}")
        

