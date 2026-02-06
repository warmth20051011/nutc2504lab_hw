import requests

# Embedding API
API_URL = "https://ws-04.wade0426.me/embed"

#把embedding變成funtion能重複使用

def get_embeddings(texts, normalize=True):
    """
    呼叫 embedding API
    回傳：
        embeddings: List[List[float]]
        vector_size: int
    """
    response = requests.post(
        API_URL,
        json={
            "texts": texts,
            "normalize": normalize
        }
    )
    
    
    print("狀態碼：", response.status_code)

    if response.status_code != 200:
        raise Exception("Embedding API 呼叫失敗")

    data = response.json()
    embeddings = data["embeddings"]
    
    # ⭐ 不寫死，動態取得維度
    vector_size = len(embeddings[0])

    return embeddings, vector_size
    
    
    
# ====== 以下是「使用範例」（教材示範區） ======

texts = [
    "公司請假需要提前三天申請。",
    "病假可以事後補單。",
    "特休依照年資給予。",
    "外出需要填寫外出單。",
    "請假需主管核准。"
]

embeddings, VECTOR_SIZE = get_embeddings(texts)

print("向量數量：", len(embeddings))
print("VECTOR_SIZE：", VECTOR_SIZE)
    

