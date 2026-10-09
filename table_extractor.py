import pdfplumber


def _rows_to_markdown(rows: list[list[str | None]]) -> str:
    cleaned = [[(cell or "").replace("\n", " ").strip() for cell in row] for row in rows]
    header, *body = cleaned
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def extract_tables(pdf_path: str, source_filename: str) -> list[dict]:
    """Returns one chunk per table found in the PDF, as markdown text."""
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            for table_index, table in enumerate(page.find_tables()):
                rows = table.extract()
                if not rows or not rows[0]:
                    continue
                markdown = _rows_to_markdown(rows)
                chunks.append({
                    "chunk_id": f"{source_filename}::p{page_num + 1}::table{table_index}",
                    "text": markdown,
                    "source": source_filename,
                    "page": page_num + 1,
                    "content_type": "table",
                    "table_index": table_index,
                    "bbox": list(table.bbox),
                })
    return chunks


if __name__ == "__main__":
    import sys
    for chunk in extract_tables(sys.argv[1], sys.argv[1]):
        print(f"--- page {chunk['page']} table {chunk['table_index']} ---")
        print(chunk["text"])
