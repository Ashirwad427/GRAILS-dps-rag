from fastapi import FastAPI, HTTPException

from app.models import (
    DataProtectionProfile,
    GrailsDPSRequest,
    GrailsDPSResponse,
    RetrievedEvidence,
)
from app.rag import DataProtectionRAG


app = FastAPI(
    title="GRAILS Data Protection Service - RAG Prototype",
    version="1.0.0",
    description=(
        "RAG-enhanced Data Protection Service following the "
        "GRAILS framework interface."
    ),
)

rag_service = DataProtectionRAG()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post(
    "/data-protection-profile",
    response_model=GrailsDPSResponse,
)
def create_data_protection_profile(
    request: GrailsDPSRequest,
):
    try:
        profile_dict, retrieved_docs = rag_service.generate_profile(
            request
        )
        # Verify due to unpredictable nature of LLMs even after structured prompting
        profile = DataProtectionProfile.model_validate(
            profile_dict
        )

        evidence = [
            RetrievedEvidence.model_validate(doc)
            for doc in retrieved_docs
        ]

        return GrailsDPSResponse(
            data_protection_profile=profile,
            retrieved_evidence=evidence,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
