from langchain_text_splitters import CharacterTextSplitter

with open("text.txt", "r", encoding="utf-8") as f:
    text = f.read()

#  滑動視窗切塊
text_splitter = CharacterTextSplitter(
    chunk_size=500,       # 每塊 500 字
    chunk_overlap=100,    # ⭐ 重疊 100 字
    separator="",         
    length_function=len
)

#  切分
chunks = text_splitter.split_text(text)

#  印出結果
print(f"總共切成 {len(chunks)} 塊\n")

for i, chunk in enumerate(chunks, 1):
    print(f"--- 分塊 {i} ---")
    print(f"長度：{len(chunk)}")
    print(f"內容：{chunk.strip()}")
    print()

