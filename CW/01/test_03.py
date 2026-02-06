from qdrant_client.models import PointStruct
from test_02_embedding import get_embeddings

# Step 3：原始文字
texts = [
    "公司請假需要提前三天申請。",
    "病假可以事後補單。",
    "特休依照年資給予。",
    "外出需要填寫外出單。",
    "請假需主管核准。"
]

# 取得 embeddings（重用 Step 2 的 def）
embeddings, VECTOR_SIZE = get_embeddings(texts)

print("VECTOR_SIZE：", VECTOR_SIZE)

# 建立 Points
points = []

for idx, vector in enumerate(embeddings):
    point = PointStruct(
        id=idx,
        vector=vector,
        payload={
            "text": texts[idx]
        }
    )
    points.append(point)

print("Point 數量：", len(points))
print("第一個 Point：")
print(points[0])

