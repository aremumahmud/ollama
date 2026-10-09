import ollama

DESCRIBE_PROMPT = (
    "First, transcribe verbatim any text visible in this image. "
    "Then, factually and concisely describe any non-text visual content "
    "(charts, diagrams, photos, tables rendered as an image, etc.). "
    "Do not speculate beyond what is visibly present."
)


def describe_image(image_bytes: bytes, model: str = "qwen2.5vl:7b") -> str:
    response = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": DESCRIBE_PROMPT,
            "images": [image_bytes],
        }],
        # Ollama's default context window (4096) is too small for a
        # full-page scanned image — the image tokens alone can exceed it,
        # which previously made every page of a scanned PDF silently fail
        # to describe (caught and skipped in ingest.py's per-image try/except).
        options={"num_ctx": 8192},
    )
    return response["message"]["content"].strip()


if __name__ == "__main__":
    import sys
    with open(sys.argv[1], "rb") as f:
        print(describe_image(f.read()))
