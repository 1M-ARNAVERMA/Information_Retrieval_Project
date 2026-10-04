# FEVER-X: Explainable Multi-Stage Fact Verification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 96 Passed](https://img.shields.io/badge/tests-96%20passed-brightgreen.svg)]()
[![Architecture: 18-Phase Modular](https://img.shields.io/badge/architecture-18--Phase%20Complete-brightgreen.svg)]()

> **Explainable Multi-Stage Fact Verification using Entity-Aware Neural Retrieval and Knowledge Graph Reasoning**  
> Benchmarked on the [FEVER Dataset](https://fever.ai/dataset/fever.html).

---

## 📌 Overview

**FEVER-X** is an end-to-end, modular fact verification system built for collaborative engineering. Given a natural language claim, FEVER-X retrieves supporting or refuting evidence from a large Wikipedia corpus, performs neural verification, and classifies the claim into:
1. **`SUPPORTS`** — Evidence proves the claim is factual.
2. **`REFUTES`** — Evidence directly contradicts the claim.
3. **`NOT ENOUGH INFO`** — The corpus lacks conclusive proof or refutation.

Beyond raw classification, FEVER-X provides **interpretable explainability**:
- **Interactive Knowledge / Evidence Graphs** (NetworkX + Plotly) visualizing claims, subclaims, entities, and multi-hop reasoning traces.
- **Multi-Factor Score Breakdowns** (Lexical Overlap, Semantic Similarity, Entity Coverage, Neural Reranker Score, NLI Entailment).
- **Human-Readable Audit Trails** articulating the exact step-by-step logic behind the verdict.

---

## 🏗️ Architecture Pipeline

```
USER CLAIM
    ↓
Text Preprocessing (Multi-mode: Lexical / Semantic / Raw)
    ↓
Claim Analysis (Regex/spaCy NER + Decomposition + Entity Expansion)
    ↓
Multi-Stage Retrieval (TF-IDF + BM25 + FAISS Dense Bi-Encoder + ColBERTv2)
    ↓
Reciprocal Rank Fusion (Weighted RRF Consensus)
    ↓
Entity-Aware Boosting (Named Entity Overlap Multiplier)
    ↓
Top-K Diverse Candidates
    ↓
Cross-Encoder Neural Reranking (Joint Query-Evidence Interaction)
    ↓
Maximal Marginal Relevance (MMR Diversity Filter)
    ↓
NLI Verification (DeBERTa-v3 / RoBERTa Entailment & Contradiction)
    ↓
Knowledge / Evidence Graph (NetworkX Multi-hop & Bridge Entity Discovery)
    ↓
FINAL VERDICT & CONFIDENCE ESTIMATION
    ↓
Explainability Breakdown & Interfaces (CLI + FastAPI + Streamlit Dashboard)
```

---

## 📊 Benchmark Ablation Study

Evaluated on FEVER benchmark test claims across all 9 architectural configurations:

| # | Architecture Configuration | Precision@10 | Recall@10 | MAP@10 | nDCG@10 |
| :- | :--- | :---: | :---: | :---: | :---: |
| 1 | **TF-IDF Baseline** | 0.2000 | 1.0000 | 0.7817 | 0.8389 |
| 2 | **BM25 Baseline** | 0.2000 | 1.0000 | 0.9167 | 0.9387 |
| 3 | **Sentence Transformer (Dense)** | 0.2000 | 1.0000 | 0.9167 | 0.9387 |
| 4 | **TF-IDF + BM25 + Dense (Simple Merge)** | 0.2000 | 1.0000 | 0.7817 | 0.8389 |
| 5 | **+ Reciprocal Rank Fusion (RRF)** | 0.2000 | 1.0000 | 0.9167 | 0.9387 |
| 6 | **+ Entity-Aware Retrieval & Boosting** | 0.2000 | 1.0000 | 0.8000 | 0.8597 |
| 7 | **+ ColBERT Late-Interaction MaxSim** | 0.2000 | 1.0000 | 0.9167 | 0.9387 |
| 8 | **+ Cross-Encoder Neural Reranking** | 0.2000 | 1.0000 | 0.8536 | 0.8937 |
| 9 | **Full FEVER-X System (End-to-End)** | **0.2000** | **1.0000** | **0.8536** | **0.8937** |

---

## 👥 4-Member Work Division

| Member | Focus Area | Core Modules |
| :--- | :--- | :--- |
| **Member 1** | Data & Baseline IR | Data Ingestion, Text Preprocessor, TF-IDF, BM25, Corpus Indexing |
| **Member 2** | Advanced & Entity IR | spaCy NER, Claim Decomposition, FAISS Bi-Encoder, ColBERT, Entity Booster |
| **Member 3** | Ranking, NLI & Graph | RRF Engine, Cross-Encoder Reranker, NLI Verifier, NetworkX Evidence Graph |
| **Member 4** | Evaluation, API & UI | P@10/Recall/MAP/nDCG Benchmarks, Ablation Suite, FastAPI, Streamlit UI |

---

## 📂 Repository Structure

```
FEVER-X/
├── config/                     # System & model YAML configs
├── data/
│   ├── raw/                    # Original FEVER dumps
│   ├── processed/              # Normalized JSONL evidence
│   ├── indexes/                # Serialized TF-IDF, BM25, and FAISS indices
│   ├── sample/                 # Reference sample dataset
│   └── evaluation/             # Ablation results (CSV and JSON)
├── src/
│   ├── common/                 # Typed Pydantic schemas, contracts, exceptions
│   ├── preprocessing/          # Multi-mode cleaner, tokenizer, stopword, stemmer
│   ├── claim_analysis/         # NER, claim decomposition, query expansion
│   ├── retrieval/              # TF-IDF, BM25, Dense FAISS, ColBERT, Entity
│   ├── ranking/                # RRF engine, Cross-Encoder reranker, MMR
│   ├── verification/           # DeBERTa NLI verifier, contradiction detector, mapper
│   ├── knowledge_graph/        # NetworkX evidence graph constructor & reasoner
│   ├── explainability/         # Multi-factor evidence scoring & audit trails
│   ├── evaluation/             # Precision@K, Recall@K, MAP@K, nDCG@K, ablation
│   ├── pipeline/               # Full FactVerificationPipeline orchestrator
│   ├── api/                    # FastAPI microservice routes & schemas
│   ├── utils/                  # Centralized logger, I/O, singleton model cache
│   └── cli.py                  # Unified feverx command line interface
├── frontend/                   # Streamlit web UI & interactive components
├── tests/                      # 96 unit, regression & integration test cases
│   ├── api/                    # REST endpoint tests
│   ├── claim_analysis/         # NER & decomposition tests
│   ├── data_ingestion/         # Parser & normalizer tests
│   ├── evaluation/             # Metric & ablation tests
│   ├── explainability/         # Score breakdown tests
│   ├── frontend/               # UI component tests
│   ├── integration/            # Full end-to-end pipeline tests
│   ├── knowledge_graph/        # Graph & path reasoning tests
│   ├── preprocessing/          # Cleaner & tokenizer tests
│   ├── ranking/                # RRF & Cross-Encoder tests
│   ├── retrieval/              # TF-IDF, BM25, Dense & ColBERT tests
│   └── verification/           # NLI inference & verdict mapping tests
├── scripts/                    # CLI runners, index builders & service starters
├── pyproject.toml              # Standard Python package distribution & entrypoint
└── requirements.txt            # System dependencies
```

---

## ⚡ Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-org/FEVER-X.git
cd FEVER-X

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate # Linux / macOS

# Install package and dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Verify Claims via CLI

```bash
# Verify a claim directly from the terminal
python -m src.cli verify "Albert Einstein was a German physicist."

# Or using the installed command
feverx verify "Albert Einstein was born in France and never studied physics."

# Output structured JSON
feverx verify "Albert Einstein developed relativity." --json
```

### 3. Launch the FastAPI Microservice

```bash
# Start backend server
python scripts/start_api.py
# Or: feverx api --port 8000
```
- Interactive Swagger Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Endpoints:
  - `POST /verify`: Full pipeline claim verification.
  - `POST /retrieve`: Candidate retrieval and ranking.
  - `POST /explain`: 5-factor explainability score breakdown.
  - `GET /evaluate/ablation`: 9-architecture evaluation metrics.
  - `GET /health`: Model status, loaded indices, and document count.

### 4. Launch the Streamlit Web Dashboard

```bash
# Start interactive web frontend
python scripts/start_frontend.py
# Or: feverx ui --port 8501
```
Navigate to [http://localhost:8501](http://localhost:8501) to explore:
- Interactive Claim Verifier with confidence meters and verdict banners.
- Expandable Evidence Cards with Plotly factor breakdown charts.
- 2D Interactive Knowledge Graph with multi-hop reasoning traces.
- Architectural Ablation Study comparison bar charts.
- Multi-Retriever Query Sandbox.

### 5. Run the Test Suite

```bash
# Execute all 96 unit and end-to-end integration tests
python -m pytest tests/ -v
```

---

## 🔄 Development Phases (All 18 Completed)

- **[x] Phase 1: Project Skeleton (Configs, Shared Types, Logger, Requirements, Tests)**
- **[x] Phase 2: FEVER Data Ingestion (Parser, Wiki Extractor, Normalizer, Tests)**
- **[x] Phase 3: Preprocessing (Cleaner, Tokenizer, Stopwords with Negations, Stemmer, Multi-mode Pipeline)**
- **[x] Phase 4: TF-IDF + BM25 Baseline (TfidfRetriever, BM25Retriever, Index Builder, Tests)**
- **[x] Phase 5: Sentence Transformer + FAISS (Dense Bi-encoder, FaissVectorIndex, ModelLoader, Tests)**
- **[x] Phase 6: RRF Fusion (ReciprocalRankFusion, Weighted Multi-Retriever, Tests)**
- **[x] Phase 7: NER + Entity Retrieval (EntityExtractor, ClaimType, EntityAwareRetriever, Tests)**
- **[x] Phase 8: Claim Decomposition & Query Expansion (ClaimDecomposer, QueryExpander, Subject Propagation, Tests)**
- **[x] Phase 9: ColBERT Retrieval (ColbertRetriever, MaxSim Late Interaction, Modular Fallback, Tests)**
- **[x] Phase 10: Cross Encoder Reranking (BaseCrossEncoder, CrossEncoderReranker, MMR Diversification, Tests)**
- **[x] Phase 11: NLI Verification (DeBERTa/RoBERTa NLI, ContradictionDetector, VerdictMapper, ConfidenceEstimator, Tests)**
- **[x] Phase 12: Knowledge Graph (NetworkX EvidenceGraph, EvidenceGraphBuilder, GraphReasoner, Tests)**
- **[x] Phase 13: Explainability (EvidenceScorer, ExplanationGenerator, Multi-Factor Scoring, Tests)**
- **[x] Phase 14: Evaluation & Ablation (Precision, Recall, MAP, nDCG, 9-Config Ablation Study, CLI Runner, Tests)**
- **[x] Phase 15: FastAPI Microservice (Endpoints /verify, /retrieve, /explain, /evaluate/ablation, /health, CORS, Lifespan, Tests)**
- **[x] Phase 16: Streamlit UI (Interactive Dashboard, Plotly Knowledge Graph, Evidence Breakdown, Ablation View, Sandbox, Tests)**
- **[x] Phase 17: Integration Testing (End-to-End Tests, Multi-Premise Claims, FEVER Labels, Parity Tests, 96/96 Tests Passing)**
- **[x] Phase 18: Final Packaging (Unified CLI, pyproject.toml, Launch Scripts, Comprehensive Documentation)**
