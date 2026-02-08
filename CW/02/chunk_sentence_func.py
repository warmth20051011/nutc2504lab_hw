def get_sentence_chunks(text: str):
    """
    表格語意切塊：
    - 以「一列」為一個 chunk
    - 移除空行
    """
    lines = text.split("\n")

    chunks = []
    for line in lines:
        line = line.strip()
        if line:
            chunks.append(line)

    return chunks

