# Legal Reasoning & Bias Detection System

A research-grade, NLI-powered pipeline for detecting logical reasoning flaws, unsupported claims, and cognitive bias in legal and forensic documents.

> **Architecture:** Transformer embeddings → Zero-Shot Classification → Semantic Evidence Retrieval → NLI Verification → Bias Risk Scoring

---

## What It Detects (Semantically, NOT by keywords)

| Flaw Type | Mechanism |
|---|---|
| Unsupported claims | Claim classified; NLI returns Neutral/Unsupported vs. retrieved evidence |
| Overconfident conclusions | High classifier confidence + Neutral NLI verdict |
| Logical contradictions | NLI returns Contradiction between claim and nearby evidence |
| Isolated reasoning | Sentence disconnected from semantic reasoning graph |

---

## System Architecture

```
PDF / DOCX
    │
    ▼
pdf_parser.py          — Text extraction & sentence segmentation
    │
    ▼
embedding_model.py     — SentenceTransformer (all-MiniLM-L6-v2) embeddings
    │
    ▼
claim_classifier.py    — Zero-shot classification → Claim / Evidence / Argument / Background
    │                    (valhalla/distilbart-mnli-12-3)
    ▼
evidence_retriever.py  — Semantic top-k retrieval per Claim (cosine similarity)
    │
    ▼
nli_verifier.py        — NLI verdict per claim-evidence pair
    │                    (cross-encoder/nli-distilroberta-base)
    ▼
reasoning_engine.py    — Bias Risk Score + Semantic Graph (networkx) + Explanation
    │
    ▼
main.py                — Orchestrator → CSV Report
app.py                 — Streamlit Web UI
```

---

## Models Used

| Module | Model | Size |
|---|---|---|
| `embedding_model.py` | `all-MiniLM-L6-v2` | ~80MB |
| `claim_classifier.py` | `valhalla/distilbart-mnli-12-3` | ~500MB |
| `nli_verifier.py` | `cross-encoder/nli-distilroberta-base` | ~300MB |

> All models are automatically downloaded from HuggingFace Hub on first run.  
> Total download: ~900MB. Subsequent runs use the local cache.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download NLTK data (one-time)

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

---

## Usage

### Option A — Command-Line (generate CSV report)

```bash
python main.py path/to/judgment.pdf
```

Optional flags:
```bash
python main.py path/to/judgment.pdf --top_k 5 --output report.csv
```

### Option B — Streamlit Web UI

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## Output Schema (CSV)

| Column | Description |
|---|---|
| `Sentence` | Raw sentence text |
| `Sentence_Type` | Claim / Evidence / Argument / Background |
| `Classifier_Confidence` | Zero-shot classification score |
| `Retrieved_Evidence` | Top-k semantically similar evidence sentences |
| `NLI_Verdict` | Supported / Contradicted / Neutral/Unsupported |
| `NLI_Score` | Confidence of NLI verdict |
| `Bias_Risk_Score` | 0.0 (no risk) → 1.0 (high risk) |
| `Final_Label` | Well-Reasoned / Weakly Supported / Potentially Biased / Inconclusive |
| `Explanation` | Human-readable reason for the verdict |

---

## Design Decisions

1. **No keyword-based rules** — Classification is zero-shot via an MNLI model. Risk is derived from NLI entailment, not word lists.
2. **Semantic Evidence Retrieval** — Uses cosine similarity over transformer embeddings to find thematically related sentences, not just adjacent ones.
3. **Dedicated NLI Model** — `cross-encoder/nli-distilroberta-base` is fine-tuned specifically for sentence-pair NLI, giving precise contradiction/entailment signals.
4. **Graph Analysis** — `networkx` semantic graph identifies isolated claims with no supporting reasoning chain.
5. **Explainability** — Every verdict includes the actual evidence text used and a natural-language justification.
