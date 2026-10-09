import os

from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, UploadFile

from db import get_collection
from ingest import ingest_pdf
from api.schemas import DocumentInfo, IngestStatusResponse

router = APIRouter()

UPLOADS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# In-memory ingestion status, keyed by filename. Ingestion of image-heavy PDFs
# (one vision-model call per image) can take several minutes, well past typical
# HTTP client timeouts, so ingestion runs in the background and the client polls.
_ingest_status: dict[str, dict] = {}


def _run_ingest(dest_path: str, filename: str, patient_id: str | None):
    collection = get_collection()
    try:
        counts = ingest_pdf(dest_path, collection, patient_id=patient_id)
        _ingest_status[filename] = {
            "status": "READY",
            "text_chunks": counts["text"],
            "table_chunks": counts["table"],
            "image_chunks": counts["image"],
            "error": None,
        }
    except Exception as e:
        _ingest_status[filename] = {
            "status": "FAILED",
            "text_chunks": 0,
            "table_chunks": 0,
            "image_chunks": 0,
            "error": str(e),
        }


@router.post("/documents/ingest", response_model=IngestStatusResponse)
async def ingest_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    patient_id: str | None = Form(default=None),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    dest_path = os.path.join(UPLOADS_DIR, file.filename)
    with open(dest_path, "wb") as f:
        f.write(await file.read())

    _ingest_status[file.filename] = {
        "status": "INGESTING",
        "text_chunks": 0,
        "table_chunks": 0,
        "image_chunks": 0,
        "error": None,
    }
    background_tasks.add_task(_run_ingest, dest_path, file.filename, patient_id)

    return IngestStatusResponse(filename=file.filename, **_ingest_status[file.filename])


@router.get("/documents/{filename}/status", response_model=IngestStatusResponse)
async def ingest_status(filename: str):
    status = _ingest_status.get(filename)
    if status is None:
        raise HTTPException(status_code=404, detail="Unknown filename")
    return IngestStatusResponse(filename=filename, **status)


@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents():
    collection = get_collection()
    all_rows = collection.get(include=["metadatas"])
    sources: dict[str, dict] = {}
    for meta in all_rows["metadatas"]:
        source = meta["source"]
        entry = sources.setdefault(source, {"text_chunks": 0, "table_chunks": 0, "image_chunks": 0})
        entry[f"{meta['content_type']}_chunks"] += 1

    return [
        DocumentInfo(filename=source, **counts)
        for source, counts in sources.items()
    ]


@router.delete("/documents/{filename}")
async def delete_document(filename: str):
    collection = get_collection()
    collection.delete(where={"source": filename})

    file_path = os.path.join(UPLOADS_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    return {"deleted": True}
