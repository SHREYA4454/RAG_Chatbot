from __future__ import annotations

import fitz


MIN_OCR_IMAGE_PIXELS = 30_000
FULL_PAGE_IMAGE_RATIO = 0.7


def should_ocr_embedded_image(page: fitz.Page, xref: int) -> bool:
    image = page.parent.extract_image(xref)
    if image["width"] * image["height"] < MIN_OCR_IMAGE_PIXELS:
        return False

    page_area = page.rect.width * page.rect.height
    if page_area <= 0:
        return False
    placements = page.get_image_rects(xref)
    if not placements:
        return False
    return not any(
        rect.width * rect.height / page_area >= FULL_PAGE_IMAGE_RATIO
        for rect in placements
    )
