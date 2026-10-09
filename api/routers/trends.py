from fastapi import APIRouter, HTTPException, UploadFile

from patient_trends import analyze_trend, answer_preset_question
from vision import describe_image
from document_text import extract_document_text
from api.schemas import (
    TrendRequest,
    TrendResponse,
    DescribeImageResponse,
    ExtractDocumentResponse,
    PatientAnswerQuestionRequest,
    TrendPresetAnswer,
)

router = APIRouter()


@router.post("/patients/{patient_id}/analyze-trend", response_model=TrendResponse)
async def analyze_patient_trend(patient_id: str, request: TrendRequest):
    notes = [note.model_dump() for note in request.notes]
    result = analyze_trend(notes, tone=request.tone, preset_questions=request.preset_questions)
    return TrendResponse(**result)


@router.post("/patients/{patient_id}/answer-question", response_model=TrendPresetAnswer)
async def answer_patient_question(patient_id: str, request: PatientAnswerQuestionRequest):
    notes = [note.model_dump() for note in request.notes]
    result = answer_preset_question(request.question, notes, tone=request.tone)
    return TrendPresetAnswer(**result)


@router.post("/notes/describe-image", response_model=DescribeImageResponse)
async def describe_note_image(file: UploadFile):
    image_bytes = await file.read()
    description = describe_image(image_bytes)
    return DescribeImageResponse(description=description)


@router.post("/notes/extract-document", response_model=ExtractDocumentResponse)
async def extract_note_document(file: UploadFile):
    data = await file.read()
    try:
        text = extract_document_text(data, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return ExtractDocumentResponse(text=text)
