from fastapi import APIRouter

from db import get_collection
from ask import answer_question_structured, TOP_K
from api.schemas import AskRequest, AskResponse

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    collection = get_collection()
    result = answer_question_structured(
        request.question,
        collection,
        k=request.k or TOP_K,
        source=request.source,
        sources=request.sources,
        patient_id=request.patient_id,
        tone=request.tone,
        notes=[n.model_dump() for n in request.notes],
    )
    return AskResponse(**result)
