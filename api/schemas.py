from typing import Literal, Optional

from pydantic import BaseModel


class Citation(BaseModel):
    marker: int
    content_type: Literal["text", "table", "image"]
    source: str
    page: int
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    table_index: Optional[int] = None
    image_index: Optional[int] = None
    snippet: str
    bbox: Optional[list[float]] = None


class AskContextNote(BaseModel):
    created_at: str
    kind: str
    text: Optional[str] = None
    image_description: Optional[str] = None


class AskRequest(BaseModel):
    question: str
    k: Optional[int] = None
    source: Optional[str] = None  # restrict retrieval to one document (filename)
    sources: list[str] = []  # restrict retrieval to a set of documents (filenames)
    patient_id: Optional[str] = None  # restrict retrieval to one patient's documents
    tone: Optional[str] = None  # admin-configured response tone/style directive
    notes: list[AskContextNote] = []  # patient notes given directly as extra grounding context


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    not_found: bool


class DocumentInfo(BaseModel):
    filename: str
    text_chunks: int
    table_chunks: int
    image_chunks: int


class IngestStatusResponse(BaseModel):
    filename: str
    status: Literal["INGESTING", "READY", "FAILED"]
    text_chunks: int = 0
    table_chunks: int = 0
    image_chunks: int = 0
    error: Optional[str] = None


class NoteInput(BaseModel):
    id: str
    created_at: str
    kind: Literal["TEXT", "IMAGE", "DOCUMENT", "MIXED"]
    text: Optional[str] = None
    image_description: Optional[str] = None


class TrendRequest(BaseModel):
    notes: list[NoteInput]
    tone: Optional[str] = None
    preset_questions: list[str] = []


class TrendCitation(BaseModel):
    marker: int
    note_id: str
    created_at: str


class TrendPresetAnswer(BaseModel):
    question: str
    answer: str
    not_found: bool
    citations: list[TrendCitation]


class PatientAnswerQuestionRequest(BaseModel):
    question: str
    notes: list[NoteInput]
    tone: Optional[str] = None


class TrendResponse(BaseModel):
    trend: Literal["IMPROVING", "DECLINING", "STAGNANT", "MIXED"]
    summary: str
    citations: list[TrendCitation]
    preset_answers: list[TrendPresetAnswer] = []


class DescribeImageResponse(BaseModel):
    description: str


class ExtractDocumentResponse(BaseModel):
    text: str


class OasisPage(BaseModel):
    page: int
    text: str


class OasisPagesResponse(BaseModel):
    pages: list[OasisPage]


class OasisExtractQuestionsRequest(BaseModel):
    page_text: str


class OasisExtractQuestionsResponse(BaseModel):
    questions: list[str]


class HealthResponse(BaseModel):
    ollama: bool
    chroma: bool


class QaJudgeRequest(BaseModel):
    question: str
    expected_answer: str
    actual_answer: Optional[str] = None
    not_found: bool = False


class QaJudgeResponse(BaseModel):
    verdict: Literal["PASS", "FAIL", "UNSURE"]
    reasoning: str


class OasisQaPair(BaseModel):
    question: str
    answer: str


class OasisContradictionCheckRequest(BaseModel):
    qa_pairs: list[OasisQaPair]


class OasisContradiction(BaseModel):
    question_a: str
    question_b: str
    explanation: str


class OasisContradictionCheckResponse(BaseModel):
    contradictions: list[OasisContradiction]
