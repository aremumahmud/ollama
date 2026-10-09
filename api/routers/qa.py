from fastapi import APIRouter

from qa import judge_answer
from api.schemas import QaJudgeRequest, QaJudgeResponse

router = APIRouter()


@router.post("/qa/judge", response_model=QaJudgeResponse)
async def judge(request: QaJudgeRequest):
    result = judge_answer(
        request.question,
        request.expected_answer,
        request.actual_answer,
        not_found=request.not_found,
    )
    return QaJudgeResponse(**result)
