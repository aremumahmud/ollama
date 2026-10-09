def chunk_lines(
    lines: list[dict],
    source_filename: str,
    target_chars: int = 1600,
    max_chars: int = 2200,
) -> list[dict]:
    """Groups per-page prose line records into citable chunks.

    Chunks never span pages, so a chunk's page number is always unambiguous.
    Prefers to close a chunk at a blank-line boundary near the size target.
    """
    chunks = []
    by_page: dict[int, list[dict]] = {}
    for line in lines:
        by_page.setdefault(line["page"], []).append(line)

    for page, page_lines in by_page.items():
        seq = 0
        current: list[dict] = []
        current_len = 0

        def flush():
            nonlocal current, current_len, seq
            if not current:
                return
            text = "\n".join(l["text"] for l in current)
            start, end = current[0]["line_no"], current[-1]["line_no"]
            bbox = [
                min(l["bbox"][0] for l in current),
                min(l["bbox"][1] for l in current),
                max(l["bbox"][2] for l in current),
                max(l["bbox"][3] for l in current),
            ]
            chunks.append({
                "chunk_id": f"{source_filename}::p{page}::{start}-{end}::{seq}",
                "text": text,
                "source": source_filename,
                "page": page,
                "content_type": "text",
                "line_start": start,
                "line_end": end,
                "bbox": bbox,
            })
            seq += 1
            current = []
            current_len = 0

        for i, line in enumerate(page_lines):
            line_len = len(line["text"]) + 1
            is_blank_boundary = line["text"] == ""

            if current_len + line_len > max_chars:
                flush()
            elif current_len >= target_chars and is_blank_boundary:
                flush()

            current.append(line)
            current_len += line_len

        flush()

    return chunks


if __name__ == "__main__":
    import sys
    from pdf_loader import extract_lines
    lines = extract_lines(sys.argv[1])
    for c in chunk_lines(lines, sys.argv[1]):
        print(f"--- {c['chunk_id']} ({len(c['text'])} chars) ---")
        print(c["text"][:200])
