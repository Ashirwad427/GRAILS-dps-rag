import json
from pathlib import Path

import bm25s
import faiss
import numpy as np

from app.config import FAISS_INDEX_PATH, METADATA_PATH, TOP_K
from app.embeddings import EmbeddingModel


class LegalRetriever:
    def __init__(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        metadata_path: Path = METADATA_PATH,
    ) -> None:
        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}. "
                f"Run: python -m scripts.build_index"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}. "
                f"Run: python -m scripts.build_index"
            )

        self.index = faiss.read_index(str(index_path))

        with metadata_path.open("r", encoding="utf-8") as f:
            self.documents = json.load(f)

        self.embedder = EmbeddingModel()
        self.bm25 = bm25s.BM25(method="lucene")
        self.bm25.index(
            bm25s.tokenize(
                [document["text"] for document in self.documents]
            ),
            show_progress=False,
        )

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        k = min(top_k or TOP_K, self.index.ntotal)

        if k <= 0:
            return []

        query_embedding = self.embedder.encode_query(query)
        query_embedding = np.asarray(
            [query_embedding],
            dtype="float32",
        )

        dense_k = min(self.index.ntotal, max(k * 3, k))
        _, indices = self.index.search(
            query_embedding,
            dense_k,
        )

        sparse_scores, sparse_indices = self.bm25.retrieve(
            bm25s.tokenize([query]),
            k=dense_k,
            show_progress=False,
        )
        rankings: dict[int, float] = {}

        for rank, idx in enumerate(indices[0]):
            if idx >= 0:
                rankings[int(idx)] = rankings.get(int(idx), 0.0) + 1 / (60 + rank + 1)
        for rank, (score, idx) in enumerate(
            zip(sparse_scores[0], sparse_indices[0])
        ):
            if score > 0:
                idx = int(idx)
                rankings[idx] = rankings.get(idx, 0.0) + 1 / (60 + rank + 1)

        results: list[dict] = []
        for idx, score in sorted(
            rankings.items(), key=lambda item: item[1], reverse=True
        )[:k]:
            item = dict(self.documents[idx])
            item["score"] = float(score)
            results.append(item)

        return results
