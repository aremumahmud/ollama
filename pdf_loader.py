import pdfplumber
import fitz  # pymupdf


def _bbox_overlap_ratio(line_bbox, table_bbox):
    lx0, ly0, lx1, ly1 = line_bbox
    tx0, ty0, tx1, ty1 = table_bbox

    ix0, iy0 = max(lx0, tx0), max(ly0, ty0)
    ix1, iy1 = min(lx1, tx1), min(ly1, ty1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0

    inter_area = (ix1 - ix0) * (iy1 - iy0)
    line_area = max((lx1 - lx0) * (ly1 - ly0), 1e-6)
    return inter_area / line_area


def _table_bboxes_per_page(pdf_path):
    bboxes = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            bboxes[page_num] = [table.bbox for table in page.find_tables()]
    return bboxes


def extract_lines(pdf_path: str) -> list[dict]:
    """Returns ordered prose line records (page, line_no, text, bbox).

    Image blocks and any line overlapping a detected table region are
    excluded here since those are handled separately (image_extractor.py,
    table_extractor.py) to avoid duplicating content across content types.
    """
    table_bboxes = _table_bboxes_per_page(pdf_path)
    lines = []

    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_tables = table_bboxes.get(page_num, [])
        line_no = 0

        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:  # skip image blocks (type 1)
                continue
            for line in block["lines"]:
                text = "".join(span["text"] for span in line["spans"]).strip()
                if not text:
                    continue
                bbox = line["bbox"]
                if any(_bbox_overlap_ratio(bbox, tb) > 0.5 for tb in page_tables):
                    continue
                lines.append({
                    "page": page_num + 1,
                    "line_no": line_no,
                    "text": text,
                    "bbox": bbox,
                })
                line_no += 1
    doc.close()
    return lines


if __name__ == "__main__":
    import sys
    for rec in extract_lines(sys.argv[1]):
        print(f"p{rec['page']} L{rec['line_no']}: {rec['text']}")
