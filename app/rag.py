from app.llm import LLMClient
from app.prompts import build_llm_prompt, build_retrieval_query
from app.retriever import LegalRetriever


class DataProtectionRAG:
    def __init__(self) -> None:
        self.retriever = LegalRetriever()
        self.llm = LLMClient()

    def generate_profile(self, request):
        retrieval_query = build_retrieval_query(request)

        retrieved_docs = self.retriever.search(
            retrieval_query
        )

        if not retrieved_docs:
            raise RuntimeError(
                "No legal evidence was retrieved from the vector index."
            )

        llm_prompt = build_llm_prompt(
            request,
            retrieved_docs,
        )

        profile = self.llm.generate_profile(
            llm_prompt
        )

        return profile, retrieved_docs
