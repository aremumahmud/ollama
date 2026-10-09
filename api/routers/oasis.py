from fastapi import APIRouter, UploadFile

from document_text import extract_pdf_pages
from oasis import extract_page_questions, detect_contradictions
from api.schemas import (
    OasisPagesResponse,
    OasisExtractQuestionsRequest,
    OasisExtractQuestionsResponse,
    OasisContradictionCheckRequest,
    OasisContradictionCheckResponse,
)

router = APIRouter()


@router.post("/oasis/pages", response_model=OasisPagesResponse)
async def get_oasis_pages(file: UploadFile):
    data = await file.read()
    pages = extract_pdf_pages(data)
    return OasisPagesResponse(pages=pages)


@router.post("/oasis/extract-questions", response_model=OasisExtractQuestionsResponse)
async def extract_questions(request: OasisExtractQuestionsRequest):
    questions = extract_page_questions(request.page_text)
    return OasisExtractQuestionsResponse(questions=questions)


@router.post("/oasis/detect-contradictions", response_model=OasisContradictionCheckResponse)
async def check_contradictions(request: OasisContradictionCheckRequest):
    contradictions = detect_contradictions([p.model_dump() for p in request.qa_pairs])
    return OasisContradictionCheckResponse(contradictions=contradictions)
