import re

from openai import OpenAI

CHAT_MODEL = "qwen2.5:7b-instruct"
VALID_TRENDS = ("IMPROVING", "DECLINING", "STAGNANT", "MIXED")
NOT_FOUND = "NOT_FOUND: the answer is not contained in the provided notes."

TREND_SYSTEM_PROMPT = """You are a strict clinical note trend-analysis assistant. You will be given a chronological list of numbered notes [note_N], each labeled with its date and kind. Rules:
1. Base your analysis ONLY on the information explicitly present in the provided notes.
2. Every claim about the patient's condition must end with a bracketed reference to the note(s) it came from, e.g. "...wound appears to be healing [note_3]."
3. Do NOT state dates or note authorship yourself — only use the [note_N] markers; the system will resolve those to exact dates.
4. Classify the overall trend as exactly one of: IMPROVING, DECLINING, STAGNANT, MIXED. State this classification alone on the FIRST line as "TREND: <label>" with nothing else on that line.
5. After the TREND line, write a brief summary (2-4 sentences) describing how the patient's condition evolved across the notes, citing the relevant [note_N] markers.
6. If there are too few notes or the notes are too inconsistent to draw a conclusion, use "TREND: STAGNANT" and say so explicitly rather than guessing."""

NOTE_QA_SYSTEM_PROMPT = """You are a strict retrieval-grounded assistant answering a question about one patient using only their chronological care notes [note_N]. Rules:
1. Answer ONLY using information explicitly present in the provided notes.
2. Every factual claim must end with a bracketed reference to the note(s) it came from, e.g. "...as stated [note_2]."
3. Do NOT state dates or authorship yourself — only use the [note_N] markers; the system will resolve those to exact dates.
4. If the answer is not fully supported by the provided notes, respond with exactly:
"NOT_FOUND: the answer is not contained in the provided notes."
Do not guess or use outside knowledge."""

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


def _label_notes(notes: list[dict]) -> tuple[str, dict]:
    """notes: chronological list of {id, created_at, kind, text, image_description}."""
    index_map = {}
    labeled = []
    for i, note in enumerate(notes, start=1):
        index_map[i] = note
        body = note.get("text") or note.get("image_description") or "(no content)"
        labeled.append(f"[note_{i}] ({note['created_at']}, {note['kind']})\n{body}")
    return "\n\n".join(labeled), index_map


def build_trend_prompt(notes: list[dict]) -> tuple[str, dict]:
    context_block, index_map = _label_notes(notes)
    user_message = context_block + "\n\nAnalyze the trend in this patient's condition over time."
    return user_message, index_map


def resolve_note_citations(answer: str, index_map: dict) -> list[dict]:
    cited = sorted({int(n) for n in re.findall(r"note_(\d+)", answer)})
    citations = []
    for n in cited:
        note = index_map.get(n)
        if note is None:
            continue  # model referenced a note that wasn't provided; ignore
        citations.append({
            "marker": n,
            "note_id": note["id"],
            "created_at": note["created_at"],
        })
    return citations


def _extract_trend_label(answer: str) -> str:
    match = re.search(r"TREND:\s*(\w+)", answer)
    if match and match.group(1).upper() in VALID_TRENDS:
        return match.group(1).upper()
    return "STAGNANT"


def _system_prompt_with_tone(base_prompt: str, tone: str | None) -> str:
    if not tone or not tone.strip():
        return base_prompt
    return f"{base_prompt}\n\nAdditionally, write your answer in this tone/style: {tone.strip()}"


def answer_preset_question(question: str, notes: list[dict], tone: str | None = None) -> dict:
    """Answers one admin-configured trend preset question, grounded only in this
    patient's notes — same NOT_FOUND-safe pattern as document Ask, just over
    notes instead of retrieved chunks (there's no retrieval step here: trend
    analysis already reasons over the patient's whole note history)."""
    if not notes:
        return {"question": question, "answer": "Not found in the patient's notes.", "not_found": True, "citations": []}

    context_block, index_map = _label_notes(notes)
    user_message = f"{context_block}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": _system_prompt_with_tone(NOTE_QA_SYSTEM_PROMPT, tone)},
            {"role": "user", "content": user_message},
        ],
    )
    raw_answer = response.choices[0].message.content.strip()

    if raw_answer.startswith("NOT_FOUND"):
        return {"question": question, "answer": "Not found in the patient's notes.", "not_found": True, "citations": []}

    return {
        "question": question,
        "answer": raw_answer,
        "not_found": False,
        "citations": resolve_note_citations(raw_answer, index_map),
    }


def analyze_trend(
    patient_notes: list[dict],
    tone: str | None = None,
    preset_questions: list[str] | None = None,
) -> dict:
    preset_answers = [
        answer_preset_question(q, patient_notes, tone=tone) for q in (preset_questions or [])
    ]

    if len(patient_notes) < 2:
        return {
            "trend": "STAGNANT",
            "summary": "Not enough notes to determine a trend.",
            "citations": [],
            "preset_answers": preset_answers,
        }

    user_message, index_map = build_trend_prompt(patient_notes)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": _system_prompt_with_tone(TREND_SYSTEM_PROMPT, tone)},
            {"role": "user", "content": user_message},
        ],
    )
    raw_answer = response.choices[0].message.content.strip()

    trend_label = _extract_trend_label(raw_answer)
    citations = resolve_note_citations(raw_answer, index_map)

    # The label is returned separately; drop the "TREND: X" line (even if the
    # model appended citations to it) from the prose.
    summary = re.sub(r"^\s*TREND:\s*\w+[^\n]*$", "", raw_answer, flags=re.MULTILINE).strip()
    if not summary:
        summary = raw_answer

    return {"trend": trend_label, "summary": summary, "citations": citations, "preset_answers": preset_answers}


if __name__ == "__main__":
    fake_notes = [
        {"id": "n1", "created_at": "2026-06-01", "kind": "TEXT", "text": "Wound on left leg is red and showing signs of infection.", "image_description": None},
        {"id": "n2", "created_at": "2026-06-08", "kind": "TEXT", "text": "Wound redness has reduced, patient reports less pain.", "image_description": None},
        {"id": "n3", "created_at": "2026-06-15", "kind": "TEXT", "text": "Wound is closed and healing well, no signs of infection.", "image_description": None},
    ]
    import json
    print(json.dumps(analyze_trend(fake_notes, preset_questions=["Is the wound healing?"]), indent=2))
