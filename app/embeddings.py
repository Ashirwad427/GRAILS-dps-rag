from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL


class EmbeddingModel:
    def __init__(self) -> None:
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def encode_documents(self, texts: list[str]):
        if hasattr(self.model, "encode_document"):
            return self.model.encode_document(
                texts,
                normalize_embeddings=True,
            )
        return self.model.encode(
            texts,
            normalize_embeddings=True,
        )

    def encode_query(self, text: str):
        if hasattr(self.model, "encode_query"):
            return self.model.encode_query(
                text,
                normalize_embeddings=True,
            )
        return self.model.encode(
            text,
            normalize_embeddings=True,
        )
