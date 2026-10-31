import unittest

from chunking import CHUNK_OVERLAP_WORDS, CHUNK_WORDS, chunks
from pdf_processing import should_ocr_embedded_image
import fitz


class ChunkingTests(unittest.TestCase):
    def test_chunks_have_configured_size_and_overlap(self):
        words = [f"word{i}" for i in range(CHUNK_WORDS + 100)]
        passages = chunks(" ".join(words))
        self.assertEqual(len(passages[0].split()), CHUNK_WORDS)
        self.assertEqual(len(passages[1].split()), 100 + CHUNK_OVERLAP_WORDS)
        self.assertEqual(
            passages[0].split()[-CHUNK_OVERLAP_WORDS:],
            passages[1].split()[:CHUNK_OVERLAP_WORDS],
        )

    def test_empty_text_has_no_chunks(self):
        self.assertEqual(chunks(" \n "), [])

    def test_full_page_embedded_image_is_not_redundantly_ocr_processed(self):
        image = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 200), False)
        document = fitz.open()
        page = document.new_page(width=400, height=400)
        page.insert_image(page.rect, stream=image.tobytes("png"))
        xref = page.get_images(full=True)[0][0]
        self.assertFalse(should_ocr_embedded_image(page, xref))
        document.close()

    def test_small_embedded_image_is_eligible_for_ocr(self):
        image = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 200), False)
        document = fitz.open()
        page = document.new_page(width=400, height=400)
        page.insert_image(fitz.Rect(20, 20, 120, 120), stream=image.tobytes("png"))
        xref = page.get_images(full=True)[0][0]
        self.assertTrue(should_ocr_embedded_image(page, xref))
        document.close()


if __name__ == "__main__":
    unittest.main()
