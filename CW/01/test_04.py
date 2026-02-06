from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from test_02_embedding import get_embeddings

# Qdrant 連線
client = QdrantClient(url="http://localhost:6333")

# Step 4：原始文字
texts = [
    "公司請假需要提前三天申請。",
    "病假可以事後補單。",
    "特休依照年資給予。",
    "外出需要填寫外出單。",
    "請假需主管核准。"
]

# 取得 embeddings
embeddings, VECTOR_SIZE = get_embeddings(texts)

print("VECTOR_SIZE：", VECTOR_SIZE)

# 建立 Points
points = []
for idx, vector in enumerate(embeddings):
    points.append(
        PointStruct(
            id=idx,
            vector=vector,
            payload={"text": texts[idx]}
        )
    )

print("Point 數量：", len(points))

# 三個 collection（對齊 Step 1）
collections = [
    "CW01_cosine",
    "CW01_dot",
    "CW01_euclidean"
]

# Upsert
for col in collections:
    client.upsert(
        collection_name=col,
        points=points
    )
    print(f"Upsert 完成：{col}")

