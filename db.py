import chromadb


def get_collection(persist_dir: str = "./chroma_db", name: str = "documents"):
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
