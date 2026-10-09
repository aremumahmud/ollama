# Local PDF RAG System

A fully local (no data leaves the machine) retrieval-augmented question-answering
system built on [Ollama](https://ollama.com). Ingests PDFs — including prose,
tables, and images — and answers questions strictly from that content, with
every claim traceable back to an exact page, line range, table, or image.

## Why local

Runs entirely on-device via Ollama. No API keys, no network calls to a
third-party LLM provider. This matters for sensitive source material (e.g.
patient records in later phases of this project) where the content should
never leave the machine.

## Models used

| Purpose              | Model                 | Notes |
|-----------------------|------------------------|-------|
| Chat / answering       | `qwen2.5:7b-instruct`  | Strong at following strict output-formatting instructions (needed for citation grounding). |
| Embeddings             | `nomic-embed-text`     | ~274MB, fast, general-purpose retrieval embeddings. |
| Vision (image analysis) | `qwen2.5vl:7b`        | Strong at document/chart/table-image understanding; transcribes visible text and describes visual content. |

Pull them with:
```
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
ollama pull qwen2.5vl:7b
```

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install openai pymupdf pdfplumber chromadb ollama
```

## Architecture

```
PDF
 ├─ pdf_loader.py      -> prose lines (page, line_no, text, bbox), table regions excluded
 ├─ table_extractor.py -> tables as markdown (page, table_index)
 └─ image_extractor.py -> raw embedded images (page, image_index, bytes)
                            │
                            ▼
                       vision.py (qwen2.5vl:7b) -> text description per image
                            │
        text lines ─► chunker.py -> text chunks (page, line_start, line_end)
                            │
        all chunks (text + table + image) ──► embeddings.py (nomic-embed-text)
                            │
                            ▼
                       db.py -> ChromaDB (persistent, cosine distance)
```

Query side:

```
question -> embed -> Chroma similarity search (top-k)
         -> confidence gate (reject if best match too weak)
         -> labeled [chunk_N] context sent to qwen2.5:7b-instruct
            with a system prompt forbidding the model from stating
            page/line/table/image numbers itself
         -> regex-extract [chunk_N] citations from the answer
         -> resolve each against OUR OWN metadata (never the model's claim)
         -> print answer + a "Sources" section with exact locations
```

### Why this avoids hallucinated citations

The model is only ever allowed to reference an opaque `[chunk_N]` marker. It
never sees or states a real page/line/table/image number. All location
metadata is captured deterministically during ingestion (from PDF structure,
not from the LLM), and citations shown to the user are resolved from that
metadata — not from anything the model asserts about where the information
came from. If the model references a chunk number that wasn't actually
retrieved, it's silently dropped rather than trusted.

Unsupported questions trigger a literal sentinel (`NOT_FOUND: ...`) from the
model, or are rejected before ever reaching the model if the top retrieval
match is too weak (cosine distance above threshold).

## Usage

**Ingest one or more PDFs** (accepts paths or globs):
```
python3 ingest.py path/to/document.pdf
python3 ingest.py path/to/folder/*.pdf
```
Re-running ingestion on the same file is idempotent (chunk IDs are
deterministic, so re-ingested content overwrites rather than duplicates).

**Ask a question:**
```
python3 ask.py "What does rule R1 say about buying a computer?"
```

Example output:
```
Rule R1 states that if the age is youth and the person is a student, then
they will buy a computer [chunk_1].

Sources:
  [chunk_1] -> page 32, Table 0 (document.pdf)
```

## File reference

| File | Responsibility |
|------|----------------|
| `pdf_loader.py` | Extracts prose text lines with page/line/bbox metadata; excludes image blocks and lines overlapping detected tables. |
| `table_extractor.py` | Detects and extracts tables via `pdfplumber`, converts to markdown. |
| `image_extractor.py` | Extracts embedded image bytes per page (skips tiny decorative images). |
| `vision.py` | Sends an image to `qwen2.5vl:7b` for transcription + description. |
| `chunker.py` | Groups prose lines into ~350-450 token chunks, never spanning a page. |
| `embeddings.py` | Wraps Ollama's embeddings API (`nomic-embed-text`). |
| `db.py` | ChromaDB persistent collection accessor (cosine distance space). |
| `ingest.py` | CLI: orchestrates extraction, chunking, embedding, and storage for one or more PDFs. |
| `ask.py` | CLI: retrieval, grounded generation, and citation resolution. |
| `test_llm.py` | Minimal smoke test confirming the chat model responds via Ollama's OpenAI-compatible API. |
| `patient_trends.py` | Full-context (no retrieval) analysis of a patient's chronological notes; classifies IMPROVING/DECLINING/STAGNANT/MIXED with the same opaque-marker citation grounding as `ask.py`. |
| `api/main.py` + `api/routers/*.py` | FastAPI service wrapping all of the above (`/documents/ingest` with background ingestion + polling, `/ask`, `/files/{filename}`, `/patients/{id}/analyze-trend`, `/notes/describe-image`, `/health`). Consumed by the Next.js app in `../ollama-web`. |

## Known limitations

- No special handling for rotated pages or complex multi-column reading order.
- Table detection may occasionally miss borderless tables (falls through to prose extraction instead).
- Vision descriptions are only as accurate as `qwen2.5vl:7b`'s transcription — dense or low-resolution scans may have OCR errors.
- Prompting prevents hallucinated *location citations* but cannot 100% guarantee the answer text never blends in outside knowledge — a standard limitation of prompt-based grounding.

## Web UI, patient records, and remote access

Built out in the sibling `../ollama-web` Next.js project: document upload/citation UI with a PDF viewer that highlights the exact cited region, admin/nurse accounts, patient profiles with care notes (text + photo), and the trend-analysis UI. See `../ollama-web/README.md`, including the Tailscale runbook for private remote access from nurse devices.
# ollama
