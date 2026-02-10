import requests


API_URL = "https://ws-04.wade0426.me/embed"



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
    


    

