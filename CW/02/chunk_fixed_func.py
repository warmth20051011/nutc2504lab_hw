from langchain_text_splitters import CharacterTextSplitter

def get_fixed_chunks(text, chunk_size=500, overlap=0):
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separator="",
        length_function=len
    )
    chunks = splitter.split_text(text)
    return chunks
