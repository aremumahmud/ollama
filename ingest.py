import glob
import os
import sys

from pdf_loader import extract_lines
from table_extractor import extract_tables
from image_extractor import extract_images
from vision import describe_image
from chunker import chunk_lines
from embeddings import embed_texts
from db import get_collection

EMBED_BATCH_SIZE = 24


def _image_chunks(pdf_path: str, source_filename: str) -> list[dict]:
    chunks = []
    for img in extract_images(pdf_path):
        try:
            description = describe_image(img["image_bytes"])
        except Exception as e:
            print(f"  ! failed to describe image p{img['page']}#{img['image_index']}: {e}")
            continue
        chunks.append({
            "chunk_id": f"{source_filename}::p{img['page']}::image{img['image_index']}",
            "text": description,
            "source": source_filename,
            "page": img["page"],
            "content_type": "image",
            "image_index": img["image_index"],
            "bbox": img.get("bbox"),
        })
    return chunks


def _normalize_metadata(chunk: dict, patient_id: str | None) -> dict:
    bbox = chunk.get("bbox")
    return {
        "source": chunk["source"],
        "page": chunk["page"],
        "content_type": chunk["content_type"],
        "line_start": chunk.get("line_start", -1),
        "line_end": chunk.get("line_end", -1),
        "table_index": chunk.get("table_index", -1),
        "image_index": chunk.get("image_index", -1),
        "bbox_x0": bbox[0] if bbox else -1.0,
        "bbox_y0": bbox[1] if bbox else -1.0,
        "bbox_x1": bbox[2] if bbox else -1.0,
        "bbox_y1": bbox[3] if bbox else -1.0,
        # Empty string (not None) since Chroma metadata values can't be null;
        # ask.py's `where={"patient_id": ...}` filter matches on this same convention.
        "patient_id": patient_id or "",
    }


def ingest_pdf(pdf_path: str, collection, patient_id: str | None = None) -> dict:
    source_filename = os.path.basename(pdf_path)

    lines = extract_lines(pdf_path)
    text_chunks = chunk_lines(lines, source_filename)
    table_chunks = extract_tables(pdf_path, source_filename)
    image_chunks = _image_chunks(pdf_path, source_filename)

    all_chunks = text_chunks + table_chunks + image_chunks
    if not all_chunks:
        return {"text": 0, "table": 0, "image": 0}

    for i in range(0, len(all_chunks), EMBED_BATCH_SIZE):
        batch = all_chunks[i:i + EMBED_BATCH_SIZE]
        vectors = embed_texts([c["text"] for c in batch])
        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=vectors,
            metadatas=[_normalize_metadata(c, patient_id) for c in batch],
        )

    return {"text": len(text_chunks), "table": len(table_chunks), "image": len(image_chunks)}


def main():
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <pdf_path_or_glob> [more...]")
        sys.exit(1)

    paths = []
    for arg in sys.argv[1:]:
        matches = glob.glob(arg)
        paths.extend(matches if matches else [arg])

    collection = get_collection()
    for pdf_path in paths:
        print(f"Ingesting {pdf_path} ...")
        counts = ingest_pdf(pdf_path, collection)
        print(f"  text chunks: {counts['text']}, tables: {counts['table']}, images: {counts['image']}")


if __name__ == "__main__":
    main()
