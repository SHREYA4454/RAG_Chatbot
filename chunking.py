CHUNK_WORDS = 650
CHUNK_OVERLAP_WORDS = 130


def chunks(text: str) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = CHUNK_WORDS - CHUNK_OVERLAP_WORDS
    return [
        " ".join(words[start : start + CHUNK_WORDS])
        for start in range(0, len(words), step)
    ]
