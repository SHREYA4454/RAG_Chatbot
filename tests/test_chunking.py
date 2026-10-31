import unittest

from chunking import CHUNK_OVERLAP_WORDS, CHUNK_WORDS, chunks


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


if __name__ == "__main__":
    unittest.main()
