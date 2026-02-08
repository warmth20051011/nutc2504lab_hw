from langchain_text_splitters import CharacterTextSplitter

def get_sliding_chunks(text, chunk_size=500, chunk_overlap=100):
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="",
        length_function=len
    )
    return text_splitter.split_text(text)
