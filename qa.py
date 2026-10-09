"""LLM judge for admin-defined QA preset questions against a document's
grounded Ask answer. Kept separate from ask.py: judging (checking an answer
against a rubric) is a distinct task from answering, and reuses whatever
grounded answer the normal /ask pipeline already produced."""

import re

from openai import OpenAI

CHAT_MODEL = "qwen2.5:7b-instruct"

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

JUDGE_SYSTEM_PROMPT = """You are a strict QA judge for a clinical document review system. You will be given a question, a rubric describing what a correct answer must contain, and the answer a retrieval system actually produced from the document. Rules:
1. Judge ONLY whether the actual answer satisfies the rubric — do not use outside knowledge of the patient or topic.
2. Respond with the verdict alone on the FIRST line as "VERDICT: <label>" where <label> is exactly one of PASS, FAIL, UNSURE.
   - PASS: the actual answer clearly satisfies the rubric.
   - FAIL: the actual answer clearly does not satisfy the rubric, or contradicts it.
   - UNSURE: it's ambiguous whether the answer satisfies the rubric.
3. After the VERDICT line, write one short sentence explaining the verdict."""

VALID_VERDICTS = ("PASS", "FAIL", "UNSURE")


def _extract_verdict(raw: str) -> str:
    match = re.search(r"VERDICT:\s*(\w+)", raw)
    if match and match.group(1).upper() in VALID_VERDICTS:
        return match.group(1).upper()
    return "UNSURE"


def judge_answer(
    question: str,
    expected_answer: str,
    actual_answer: str | None,
    not_found: bool = False,
) -> dict:
    if not_found or not actual_answer:
        return {
            "verdict": "FAIL",
            "reasoning": "The answer was not found in the document.",
        }

    user_message = (
        f"Question: {question}\n\n"
        f"Rubric (what a correct answer must contain): {expected_answer}\n\n"
        f"Actual answer produced by the system:\n{actual_answer}"
    )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    raw_answer = response.choices[0].message.content.strip()
    verdict = _extract_verdict(raw_answer)
    reasoning = re.sub(r"^\s*VERDICT:\s*\w+[^\n]*\n?", "", raw_answer).strip() or raw_answer

    return {"verdict": verdict, "reasoning": reasoning}
