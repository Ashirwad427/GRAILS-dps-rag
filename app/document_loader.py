import re
from pathlib import Path


STATUTE_ANCHOR_PATTERN = re.compile(
    r"(?=^[ \t]*(?:(?:Section|Article|Sec\.|Rule|Regulation|Clause)[ \t]+\d+"
    r"|\d+\.?[ \t]+(?=[A-Z(])|CHAPTER[ \t]+[IVXLCDM\d]+|CLAUSE[ \t]+\d+))",
    re.IGNORECASE | re.MULTILINE,
)


def _split_long_provision(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if boundary > start + max_chars // 2:
                end = boundary + (1 if text[boundary] == "\n" else 2)

        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(0, end - overlap_chars)

    return [chunk for chunk in chunks if chunk]


def chunk_legal_document(
    text: str,
    act_name: str,
    max_chunk_chars: int = 1200,
    overlap_chars: int = 150,
) -> list[dict]:
    provisions = STATUTE_ANCHOR_PATTERN.split(text)
    chunks: list[dict] = []

    for provision in provisions:
        cleaned = "\n".join(
            line.strip() for line in provision.splitlines() if line.strip()
        ).strip()
        if not cleaned:
            continue

        first_line = cleaned.splitlines()[0]
        provision_match = re.match(
            r"(?:(?:Section|Article|Sec\.|Rule|Regulation|Clause)[ \t]+\d+"
            r"|\d+\.?[ \t]+(?=[A-Z(])|CHAPTER[ \t]+[IVXLCDM\d]+|CLAUSE[ \t]+\d+)[^\n]*",
            first_line,
            re.IGNORECASE,
        )
        provision_title = (
            provision_match.group(0).strip()
            if provision_match
            else first_line[:100]
        )
        header = f"[{act_name}] {provision_title}\n------------------------------\n"
        parts = (
            [cleaned]
            if len(cleaned) <= max_chunk_chars
            else _split_long_provision(
                cleaned,
                max_chunk_chars - len(header),
                overlap_chars,
            )
        )

        for part_index, part in enumerate(parts, start=1):
            chunks.append(
                {
                    "text": f"{header}(Part {part_index})\n{part}"
                    if len(parts) > 1
                    else f"{header}{part}",
                    "provision": provision_title,
                    "part": part_index if len(parts) > 1 else None,
                }
            )

    return chunks


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Keep the old helper available for callers outside the index pipeline."""
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start = max(0, end - overlap)

    return chunks


def load_documents(directory: Path) -> list[dict]:
    if not directory.exists():
        raise FileNotFoundError(
            f"Legal document directory does not exist: {directory}"
        )

    documents: list[dict] = []

    for path in sorted(directory.glob("*.txt")):
        text = path.read_text(encoding="utf-8")

        for chunk_id, chunk in enumerate(chunk_legal_document(text, path.stem)):
            documents.append(
                {
                    "document": path.stem,
                    "chunk_id": chunk_id,
                    "text": chunk["text"],
                    "provision": chunk["provision"],
                    "part": chunk["part"],
                }
            )

    return documents
