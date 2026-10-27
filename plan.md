# Engineering Plan: Generic Structural Chunking & Hybrid Retrieval for Legal RAG

This document outlines the step-by-step implementation plan to upgrade the Data Protection Service (DPS) retrieval engine. The objective is to replace arbitrary fixed-character chunking and dense-only search with a generic statutory parser and a BM25 + Dense Hybrid Retriever.

---

## 1. Overview of Architecture Changes

Raw Legal Text (DPDP, GDPR, HIPAA, etc.)
                   │
                   ▼
┌────────────────────────────────────────────────────────┐
│ Generic Structural Legal Chunking                      │
│ - Detects Section / Article / Clause anchors           │
│ - Injects Document & Section breadcrumb headers        │
│ - Recursively partitions over-length provisions        │
└──────────────────────────┬─────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
  ┌───────────────────────┐   ┌──────────────────────┐
  │   Sparse Index (BM25) │   │  Dense Vector Store  │
  │   - Exact section keys│   │  - Semantic concepts │
  │   - Statutory terms   │   │  - Contextual intent │
  └───────────┬───────────┘   └──────────┬───────────┘
              └─────────────┬────────────┘
                            ▼
              ┌───────────────────────────┐
              │ Ensemble Retriever (RRF)  │
              │ Reciprocal Rank Fusion    │
              └─────────────┬─────────────┘
                            ▼
              Context to LLM for DPP Extraction

---

## 2. Generic Structural Legal Chunker

Different legal frameworks use distinct numbering conventions (e.g., *Article 1, 2...* in GDPR; *Section 1, 2...* in DPDP; *Regulation \d+* in administrative rules). The generic chunker detects standard statutory header patterns and preserves parent context across splits.

**Changes to be made in the appropriate text chunking logic:**

Replace existing character splitters with the following regex-driven structural parser:

```python
import re
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Universal regex matching common statutory boundary anchors
GENERIC_STATUTE_PATTERN = (
    r"(?=(?:^|\n)\s*(?:"
    r"Section\s+\d+|"          
    r"Article\s+\d+|"          
    r"Sec\.\s*\d+|"            
    r"Rule\s+\d+|"             
    r"Regulation\s+\d+|"       
    r"CHAPTER\s+[IVXLCDM\d]+|" 
    r"CLAUSE\s+\d+"            
    r")[\.\:\s\-\—])"
)

def chunk_legal_document(
    text: str,
    act_name: str,
    max_chunk_chars: int = 1200,
    overlap_chars: int = 150
) -> List[Document]:
    """
    Splits any legal document by statutory provisions and prepends breadcrumbs.
    """
    raw_provisions = re.split(GENERIC_STATUTE_PATTERN, text, flags=re.IGNORECASE | re.MULTILINE)
    
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_chars,
        chunk_overlap=overlap_chars,
        separators=[
            "\n(?=\([a-z0-9]+\))",  # Subsection anchors like (1), (a), (i)
            "\n\n",
            "\n",
            ". "
        ]
    )

    documents: List[Document] = []

    for provision in raw_provisions:
        cleaned_text = provision.strip()
        if not cleaned_text:
            continue

        lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
        first_line = lines[0] if lines else "General Provision"
        
        header_match = re.match(
            r"((?:Section|Article|Sec\.|Rule|Regulation|CHAPTER|CLAUSE)\s+[^\.\:\n]+[\.\:\s\-\—]?[^\n]*)",
            first_line,
            re.IGNORECASE
        )
        provision_title = header_match.group(1).strip() if header_match else first_line[:80]
        
        # Breadcrumb header injected into every chunk
        context_header = f"[{act_name}] {provision_title}\n" + ("-" * 30) + "\n"

        # Fit within context limit or recursively partition
        if len(cleaned_text) <= max_chunk_chars:
            documents.append(
                Document(
                    page_content=context_header + cleaned_text,
                    metadata={"act": act_name, "provision": provision_title}
                )
            )
        else:
            sub_chunks = fallback_splitter.split_text(cleaned_text)
            for idx, part in enumerate(sub_chunks):
                documents.append(
                    Document(
                        page_content=f"{context_header}(Part {idx + 1})\n{part}",
                        metadata={"act": act_name, "provision": provision_title, "part": idx + 1}
                    )
                )

    return documents
3. Hybrid Retrieval Engine
To prevent missing exact statutory citations while preserving semantic search capabilities, combine BM25Retriever (sparse) and Chroma (dense vector) into an EnsembleRetriever.

Changes to be made in the retrieval setup logic:

Modify the retriever initialization sequence to build and return an ensemble instance:

Python
from typing import List
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

def build_hybrid_retriever(
    documents: List[Document],
    persist_dir: str = "./vector_db",
    k: int = 4
) -> EnsembleRetriever:
    """
    Constructs an Ensemble Hybrid Retriever using BM25 and Chroma Vectorstore.
    """
    # 1. Sparse Lexical Retriever (BM25)
    bm25 = BM25Retriever.from_documents(documents)
    bm25.k = k

    # 2. Dense Semantic Retriever (Chroma)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    dense = vectorstore.as_retriever(search_kwargs={"k": k})

    # 3. Hybrid Ensemble via Reciprocal Rank Fusion (RRF)
    # Balanced 50/50 between exact citations and semantic intent
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[0.5, 0.5]
    )

    return hybrid_retriever
4. Pipeline Integration & DPP Context Extraction
Wire the newly configured hybrid retriever into the existing extraction chain to construct the final prompt payload.

Changes to be made in the extraction pipeline logic:

Ensure the main invocation function passes the query to the ensemble retriever and accurately unpacks the dual-retrieved context for the LLM prompt:

Python
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

PROMPT_TEMPLATE = """
You are the Data Protection Service of an ethical access framework.
Analyze the requested data item against the legal context provided below.

Legal Context:
{context}

Data Item Under Evaluation:
Table: {table_name}
Column: {column_name}
Domain: {domain}

Return a valid JSON object strictly matching these fields:
{{
  "sensitivity": "Low" | "Moderate" | "High",
  "pii_status": true | false,
  "domain_classification": "{domain}",
  "governing_provision": "<Act Section/Article and matched>",
  "required_safeguard": "Masking" | "Anonymization" | "Generalization" | "None"
}}
"""

def generate_dpp_profile(retriever, table: str, column: str, domain: str) -> dict:
    query = f"{domain} {column} Personally Identifiable Information classification and data sensitivity rules"
    retrieved_docs = retriever.invoke(query)
    
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    
    chain = prompt | llm
    response = chain.invoke({
        "context": context_text,
        "table_name": table,
        "column_name": column,
        "domain": domain
    })
    
    return json.loads(response.content)