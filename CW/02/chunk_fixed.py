from langchain_text_splitters import CharacterTextSplitter

# 1. 讀取文字檔
with open("text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 2. 固定大小切塊
text_splitter = CharacterTextSplitter(
    chunk_size=500,     
    chunk_overlap=0,# 不重疊
    separator="",
    length_function=len
)

# 3. 進行切分
chunks = text_splitter.split_text(text)

# 4. 印出結果
print(f"總共切成 {len(chunks)} 塊\n")

for i, chunk in enumerate(chunks, 1):
    print(f"--- 分塊  {i} ---")
    print(f"長度：{len(chunk)}")
    print(f"內容：{chunk.strip()}")
    print()

