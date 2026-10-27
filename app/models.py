from typing import List, Optional

from pydantic import BaseModel, Field


class User(BaseModel):
    user_id: str
    user_type: str
    organization: Optional[str] = None


class DataRequest(BaseModel):
    data_item: str
    operation: str


class Context(BaseModel):
    purpose: str
    jurisdiction: str
    domain: str


class DataMetadata(BaseModel):
    source: str
    ownership: str
    domain: str
    pii: bool
    sensitivity: str
    data_type: str


class GrailsDPSRequest(BaseModel):
    user: User
    data_request: DataRequest
    context: Context
    metadata: DataMetadata


class DataProtectionProfile(BaseModel):
    protection_level: str = Field(
        description="Overall protection level: low, moderate, or high."
    )
    applicable_regulations: List[str]
    requirements: List[str]
    regulatory_references: List[str]
    reasoning_summary: str


class RetrievedEvidence(BaseModel):
    document: str
    chunk_id: int
    score: float
    text: str


class GrailsDPSResponse(BaseModel):
    data_protection_profile: DataProtectionProfile
    retrieved_evidence: List[RetrievedEvidence]
