"""Page-by-page question extraction for OASIS assessment uploads.

Deliberately kept separate from the answering step (ask.py): a page's raw
text is extracted here as a small, single-page LLM call — never the whole
30+ page document at once — so the model is only ever asked to reason about
one page's worth of content. Answering each extracted question is a further
breakdown, handled per-question by the existing grounded ask.py pipeline
against the patient's own notes/documents (called from the Next.js
orchestrator, not from here).
"""

import json
import re

from openai import OpenAI

CHAT_MODEL = "qwen2.5:7b-instruct"

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

QUESTION_EXTRACTION_PROMPT = """You extract the distinct question(s) or data-entry item(s) present on a single page of a clinical assessment form (such as OASIS). You will be given the raw text of ONE page only. Rules:
1. Identify each distinct question or item a clinician would need to answer or fill in, e.g. "What is the patient's primary diagnosis?", "Is the patient's skin intact?".
2. Ignore headers, footers, page numbers, form instructions, legal text, and section titles that are not themselves questions.
3. Rephrase each item as a clear, standalone question if it is not already phrased as one.
4. Respond with ONLY a JSON array of strings, e.g. ["Question 1?", "Question 2?"]. If the page has no questions or items to answer, respond with exactly [].
Do not include any explanation, markdown formatting, or code fences — the response must be valid JSON and nothing else."""

CONTRADICTION_DETECTION_PROMPT = """You review the answers given across a single clinical assessment (such as OASIS) for medical contradictions. You will be given a numbered list of question/answer pairs answered from the same patient's records. Rules:
1. Flag a pair of answers ONLY if they factually contradict each other (e.g. one answer says the patient has no history of falls while another describes a recent fall; one says a wound is healed while another describes an open wound).
2. Do not flag answers that are merely about different topics, or that are consistent but phrased differently.
3. Respond with ONLY a JSON array of objects, each with keys "question_a", "question_b", "explanation" (a short sentence on why they contradict), using the exact question text as given. If there are no contradictions, respond with exactly [].
Do not include any explanation, markdown formatting, or code fences — the response must be valid JSON and nothing else."""


def extract_page_questions(page_text: str) -> list[str]:
    if not page_text or not page_text.strip():
        return []

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": QUESTION_EXTRACTION_PROMPT},
            {"role": "user", "content": page_text},
        ],
    )
    raw = response.choices[0].message.content.strip()
    # The model is instructed not to use code fences, but small instruct
    # models sometimes add them anyway — strip if present before parsing.
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())

    try:
        questions = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(questions, list):
        return []
    return [str(q).strip() for q in questions if str(q).strip()]


def detect_contradictions(qa_pairs: list[dict]) -> list[dict]:
    """qa_pairs: [{"question": str, "answer": str}, ...] — all DONE (not
    not_found) answers from one OASIS document. One LLM call over the whole
    set so contradictions across pages/sections can be caught."""
    if len(qa_pairs) < 2:
        return []

    numbered = "\n\n".join(
        f"{i}. Q: {p['question']}\n   A: {p['answer']}" for i, p in enumerate(qa_pairs, start=1)
    )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": CONTRADICTION_DETECTION_PROMPT},
            {"role": "user", "content": numbered},
        ],
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())

    try:
        contradictions = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(contradictions, list):
        return []

    result = []
    for c in contradictions:
        if not isinstance(c, dict):
            continue
        question_a, question_b, explanation = c.get("question_a"), c.get("question_b"), c.get("explanation")
        if question_a and question_b and explanation:
            result.append({
                "question_a": str(question_a),
                "question_b": str(question_b),
                "explanation": str(explanation),
            })
    return result
