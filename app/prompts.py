SYSTEM_PROMPT = """
You are the Data Protection Service (DPS) in the GRAILS framework.

Your task is ONLY to generate the Data Protection Profile.

Use ONLY:
1. The supplied data metadata.
2. The retrieved legal/regulatory evidence.

Do NOT:
- invent legal requirements;
- invent regulatory sections;
- reason using anything outside the retrieved legal evidences

Return ONLY valid JSON in exactly this structure:

{
  "protection_level": "low|moderate|high",
  "applicable_regulations": ["..."],
  "requirements": ["..."],
  "regulatory_references": ["..."],
  "reasoning_summary": "..."
}
""".strip()


def build_retrieval_query(request) -> str:
    return f"""
Jurisdiction: {request.context.jurisdiction}
Domain: {request.context.domain}
Purpose: {request.context.purpose}

Data item: {request.data_request.data_item}
Operation: {request.data_request.operation}

Data source: {request.metadata.source}
Ownership: {request.metadata.ownership}
PII: {request.metadata.pii}
Sensitivity: {request.metadata.sensitivity}
Data type: {request.metadata.data_type}

Find the legal and regulatory requirements relevant to
protecting, processing, accessing, sharing, or using this data item.
""".strip()


def build_llm_prompt(request, retrieved_docs: list[dict]) -> str:
    evidence_blocks = []

    for doc in retrieved_docs:
        evidence_blocks.append(
            "\n".join(
                [
                    f"DOCUMENT: {doc['document']}",
                    f"CHUNK_ID: {doc['chunk_id']}",
                    f"SIMILARITY_SCORE: {doc['score']:.4f}",
                    "TEXT:",
                    doc["text"],
                ]
            )
        )

    evidence = "\n\n---\n\n".join(evidence_blocks)

    return f"""
GRAILS DATA ITEM

Data item: {request.data_request.data_item}
Operation: {request.data_request.operation}

Context:
- Purpose: {request.context.purpose}
- Jurisdiction: {request.context.jurisdiction}
- Domain: {request.context.domain}

Data Metadata:
- Source: {request.metadata.source}
- Ownership: {request.metadata.ownership}
- Domain: {request.metadata.domain}
- PII: {request.metadata.pii}
- Sensitivity: {request.metadata.sensitivity}
- Data type: {request.metadata.data_type}

RETRIEVED LEGAL EVIDENCE

{evidence}

Using ONLY the retrieved legal evidence and the supplied metadata,
produce the Data Protection Profile as valid JSON matching the schema
given in the system instructions. Reasoning summary should be concise,
precise and strictly from the retrieved legal evidences mentioning them
as well.
""".strip()
