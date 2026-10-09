import ollama


def embed_texts(texts: list[str], model: str = "nomic-embed-text") -> list[list[float]]:
    return [ollama.embeddings(model=model, prompt=text)["embedding"] for text in texts]
