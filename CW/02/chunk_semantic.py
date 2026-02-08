from semantic_text_splitter import TextSplitter

# 讀取 text.txt
with open("text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 設定「每塊最大字元數」
max_characters = 500
splitter = TextSplitter(max_characters)

# 進行語句切塊
chunks = splitter.chunks(text)
chunks
print(f"總共切成 {len(chunks)} 塊\n")

for i, chunk in enumerate(chunks, 1):
    print(f"--- 分塊 {i} ---")
    print(f"長度：{len(chunk)}")
    print(chunk)
    print()

