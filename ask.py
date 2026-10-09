import re
import sys

from openai import OpenAI

from embeddings import embed_texts
from db import get_collection

CHAT_MODEL = "qwen2.5:7b-instruct"
TOP_K = 6
DISTANCE_THRESHOLD = 0.55  # cosine distance; higher = more permissive
NOT_FOUND = "NOT_FOUND: the answer is not contained in the provided documents."

SYSTEM_PROMPT = """You are a strict retrieval-grounded assistant. You will be given patient notes and/or numbered context chunks [chunk_N], each labeled with its content type. Rules:
1. Answer ONLY using information explicitly present in the provided notes and/or chunks.
2. Every factual claim grounded in a chunk must end with a bracketed reference to it, e.g. "...as stated [chunk_2]." Claims grounded only in the patient notes must NOT have a bracket citation — never write [chunk_N] unless a chunk with that exact number was given to you above.
3. Do NOT state page numbers, line numbers, table numbers, image numbers, or filenames yourself — only use the [chunk_N] markers; the system will resolve those to exact locations.
4. If several notes and/or chunks discuss the same topic (e.g. the same symptom) consistently, synthesize them into one confident, comprehensive statement citing all of the supporting sources — do not treat each mention as a separate fragment, and do not respond NOT_FOUND just because the information is spread across multiple notes/chunks.
5. If notes and/or chunks genuinely conflict on a topic, do not silently pick one — say so explicitly and cite both sides.
6. If the answer is not fully supported by the provided notes or chunks, respond with exactly:
"NOT_FOUND: the answer is not contained in the provided documents."
Do not guess or use outside knowledge."""

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


def _format_location(meta: dict) -> str:
    if meta["content_type"] == "text":
        return f"page {meta['page']}, lines {meta['line_start']}-{meta['line_end']} ({meta['source']})"
    if meta["content_type"] == "table":
        return f"page {meta['page']}, Table {meta['table_index']} ({meta['source']})"
    return f"page {meta['page']}, Image {meta['image_index']} ({meta['source']})"


def retrieve(
    question: str,
    collection,
    k: int = TOP_K,
    source: str | None = None,
    sources: list[str] | None = None,
    patient_id: str | None = None,
):
    q_vec = embed_texts([question])[0]
    conditions = []
    if sources:
        conditions.append({"source": {"$in": sources}})
    elif source:
        conditions.append({"source": source})
    if patient_id:
        conditions.append({"patient_id": patient_id})
    elif not source and not sources:
        # System-wide "all documents" search (no source, no patient_id):
        # exclude patient-scoped chunks so a patient's private documents
        # never surface outside that patient's own Ask/Library.
        conditions.append({"patient_id": ""})
    where = None
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}
    res = collection.query(query_embeddings=[q_vec], n_results=k, where=where)
    return res["documents"][0], res["metadatas"][0], res["distances"][0]


def _format_notes_block(notes: list[dict]) -> str | None:
    if not notes:
        return None
    lines = [
        f"- (dated {n['created_at']}, {n['kind']}): {n.get('text') or n.get('image_description') or '(no content)'}"
        for n in notes
    ]
    return "Patient notes:\n" + "\n".join(lines)


def build_prompt(
    question: str,
    documents: list[str],
    metadatas: list[dict],
    notes: list[dict] | None = None,
):
    index_map = {}
    labeled = []
    notes_block = _format_notes_block(notes or [])
    if notes_block:
        labeled.append(notes_block)
    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        index_map[i] = {"meta": meta, "text": doc}
        labeled.append(f"[chunk_{i}] ({meta['content_type']})\n{doc}")
    context_block = "\n\n".join(labeled)
    user_message = f"{context_block}\n\nQuestion: {question}"
    return user_message, index_map


def resolve_citations(answer: str, index_map: dict) -> str:
    cited = sorted({int(n) for n in re.findall(r"chunk_(\d+)", answer)})
    if not cited:
        return answer

    sources_lines = []
    for n in cited:
        entry = index_map.get(n)
        if entry is None:
            continue  # model referenced a chunk that wasn't retrieved; ignore
        sources_lines.append(f"  [chunk_{n}] -> {_format_location(entry['meta'])}")

    if not sources_lines:
        return answer
    return answer + "\n\nSources:\n" + "\n".join(sources_lines)


def _bbox_from_meta(meta: dict) -> list[float] | None:
    x0, y0, x1, y1 = meta.get("bbox_x0", -1.0), meta.get("bbox_y0", -1.0), meta.get("bbox_x1", -1.0), meta.get("bbox_y1", -1.0)
    if x0 == -1.0 and y0 == -1.0 and x1 == -1.0 and y1 == -1.0:
        return None
    return [x0, y0, x1, y1]


def resolve_citations_structured(answer: str, index_map: dict) -> list[dict]:
    cited = sorted({int(n) for n in re.findall(r"chunk_(\d+)", answer)})
    citations = []
    for n in cited:
        entry = index_map.get(n)
        if entry is None:
            continue  # model referenced a chunk that wasn't retrieved; ignore
        meta = entry["meta"]
        citations.append({
            "marker": n,
            "content_type": meta["content_type"],
            "source": meta["source"],
            "page": meta["page"],
            "line_start": meta["line_start"] if meta["line_start"] != -1 else None,
            "line_end": meta["line_end"] if meta["line_end"] != -1 else None,
            "table_index": meta["table_index"] if meta["table_index"] != -1 else None,
            "image_index": meta["image_index"] if meta["image_index"] != -1 else None,
            "snippet": entry["text"],
            "bbox": _bbox_from_meta(meta),
        })
    return citations


def answer_question(question: str, collection) -> str:
    documents, metadatas, distances = retrieve(question, collection)

    if not documents or distances[0] > DISTANCE_THRESHOLD:
        return "Not found in the provided documents."

    user_message, index_map = build_prompt(question, documents, metadatas)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    raw_answer = response.choices[0].message.content.strip()

    if raw_answer.startswith("NOT_FOUND"):
        return "Not found in the provided documents."

    return resolve_citations(raw_answer, index_map)


def _system_prompt_with_tone(tone: str | None) -> str:
    if not tone or not tone.strip():
        return SYSTEM_PROMPT
    return f"{SYSTEM_PROMPT}\n\nAdditionally, write your answer in this tone/style: {tone.strip()}"


def answer_question_structured(
    question: str,
    collection,
    k: int = TOP_K,
    source: str | None = None,
    sources: list[str] | None = None,
    patient_id: str | None = None,
    tone: str | None = None,
    notes: list[dict] | None = None,
) -> dict:
    documents, metadatas, distances = retrieve(
        question, collection, k=k, source=source, sources=sources, patient_id=patient_id
    )

    # Chunks are only useful if they're actually relevant (distance-gated);
    # notes are given directly (no retrieval/relevance gate), so as long as
    # there's *something* to ground on — notes or relevant chunks — we go
    # ahead and let the LLM's own NOT_FOUND check handle unanswerable cases.
    chunks_relevant = bool(documents) and distances[0] <= DISTANCE_THRESHOLD
    if not chunks_relevant:
        documents, metadatas = [], []
    if not chunks_relevant and not notes:
        return {"answer": "Not found in the provided documents.", "citations": [], "not_found": True}

    user_message, index_map = build_prompt(question, documents, metadatas, notes=notes)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": _system_prompt_with_tone(tone)},
            {"role": "user", "content": user_message},
        ],
    )
    raw_answer = response.choices[0].message.content.strip()

    if raw_answer.startswith("NOT_FOUND"):
        return {"answer": "Not found in the provided documents.", "citations": [], "not_found": True}

    citations = resolve_citations_structured(raw_answer, index_map)
    return {"answer": raw_answer, "citations": citations, "not_found": False}


def main():
    if len(sys.argv) < 2:
        print('Usage: python ask.py "your question"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    collection = get_collection()
    print(answer_question(question, collection))


if __name__ == "__main__":
    main()
