import json

import faiss
import numpy as np

from app.config import (
    FAISS_INDEX_PATH,
    INDEX_DIR,
    METADATA_PATH,
    LEGAL_DIR,
)
from app.document_loader import load_documents
from app.embeddings import EmbeddingModel


def main() -> None:
    documents = load_documents(LEGAL_DIR)

    if not documents:
        raise RuntimeError(
            f"No .txt legal documents found in {LEGAL_DIR}. "
            "Add legal documents and run this command again."
        )

    texts = [doc["text"] for doc in documents]

    print(f"Loaded {len(documents)} legal chunks.")

    embedder = EmbeddingModel()

    print("Generating embeddings...")
    embeddings = embedder.encode_documents(texts)

    embeddings = np.asarray(
        embeddings,
        dtype="float32",
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(
        index,
        str(FAISS_INDEX_PATH),
    )

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            documents,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved FAISS index to: {FAISS_INDEX_PATH}")
    print(f"Saved metadata to: {METADATA_PATH}")
    print(f"Indexed {len(documents)} chunks.")


if __name__ == "__main__":
    main()
