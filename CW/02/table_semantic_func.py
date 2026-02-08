def get_table_semantic_chunks(file_path: str):
    """
    一行表格 = 一個語意 chunk
    """
    chunks = []

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if not line:
            continue
        if set(line) <= {"|", "-", " "}:
            continue

        chunks.append(line)

    return chunks
