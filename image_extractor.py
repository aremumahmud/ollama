import fitz  # pymupdf

MIN_DIMENSION_PX = 50


def extract_images(pdf_path: str) -> list[dict]:
    """Returns raw embedded image records (page, image_index, image_bytes, ext).

    Skips tiny images (below MIN_DIMENSION_PX) since these are almost always
    decorative icons/bullets rather than meaningful content.
    """
    images = []
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        for image_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base = doc.extract_image(xref)
            if base["width"] < MIN_DIMENSION_PX or base["height"] < MIN_DIMENSION_PX:
                continue
            rects = page.get_image_rects(xref)
            bbox = list(rects[0]) if rects else None
            images.append({
                "page": page_num + 1,
                "image_index": image_index,
                "image_bytes": base["image"],
                "ext": base["ext"],
                "bbox": bbox,
            })
    doc.close()
    return images


if __name__ == "__main__":
    import sys
    for rec in extract_images(sys.argv[1]):
        print(f"page {rec['page']} image {rec['image_index']}: {len(rec['image_bytes'])} bytes ({rec['ext']})")
