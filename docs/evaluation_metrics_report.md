# RAG Evaluation Metrics - Validation Report

## Executive Summary

**Status:** ✅ **Metrics are CORRECT** according to Ragas/DeepEval standards

**Current Implementation (4 metrics):**
- ✅ Faithfulness (0.0-1.0) - Standard
- ✅ Correctness (1-5 LLM judge) - Acceptable
- ✅ Answer Relevancy (1-5 LLM judge) - Acceptable  
- ✅ Temporal Correctness (binary) - Novel domain-specific (appropriate)

**Actions Taken:**
- ✅ Removed QUEST metrics (quality_of_information, understanding_and_reasoning, expression_style, safety_and_harm, trust_and_confidence)
- ✅ Removed MRR and NDCG from retrieval metrics
- ✅ Improved Faithfulness prompt with explicit calculation method
- ✅ Improved Correctness definition with percentage thresholds

---

## Metrics Overview

### Retrieval Metrics (Offline - No LLM)

Located in: `src/legal_rag/evaluation/metrics.py`

| Metric | Scale | Description |
|--------|-------|-------------|
| **Hit Rate@K** | 0/1 | Binary: found at least 1 relevant chunk in top-K |
| **Precision@K** | 0.0-1.0 | % relevant chunks in top-K results |
| **Recall@K** | 0.0-1.0 | % gold chunks found in top-K |
| **Context Precision** | 0.0-1.0 | Mean precision across all retrieved |
| **Context Recall** | 0.0-1.0 | Mean recall across all retrieved |

**Example:**
```
Gold chunks: [A, B, C]
Retrieved: [X, A, Y, B, Z]

Hit Rate@5 = 1.0 (found A and B)
Precision@5 = 2/5 = 0.4 (2 correct out of 5)
Recall@5 = 2/3 = 0.67 (found 2/3 needed chunks)
```

---

### Generation Metrics (LLM Judge)

Located in: `evaluations/baml/judge.py` + `baml_src/answer_evaluation.baml`

| Metric | Scale | What It Measures | Standard |
|--------|-------|------------------|----------|
| **Faithfulness** | 0.0-1.0 | % of claims supported by context | Ragas, DeepEval |
| **Correctness** | 1-5 | Factual accuracy vs expected answer | Ragas (similar) |
| **Answer Relevancy** | 1-5 | On-topic vs rambling | Ragas, DeepEval |
| **Temporal Correctness** | 0/1 | Applies correct law version | Novel (legal domain) |

---

## Metric Definitions

### 1. Faithfulness (0.0-1.0)

**Definition:** Measures hallucination - percentage of claims in the answer that are grounded in retrieved context.

**Calculation:**
```
1. Extract ALL claims from answer
2. For each claim, check if contexts support it
3. Score = (supported claims) / (total claims)
```

**Example:**
```
Answer: "Phạt 800k-1tr, tước bằng 1-3 tháng, theo ND168/2024, áp dụng từ 2025"
Claims extracted: 4
Claims supported in contexts: 3
Faithfulness: 3/4 = 0.75
```

**Why Not 1-5 Scale?**
- Ragas standard: 0.0-1.0 continuous
- Natural output of claim ratio calculation
- Enables statistical operations (mean, std, confidence intervals)

---

### 2. Correctness (1-5 Scale)

**Definition:** Measures factual accuracy by comparing answer content with expected reference answer.

**Scale:**
- 1 = Completely wrong (<30% correct)
- 2 = Small part correct (30-50%)
- 3 = About half correct (50-70%)
- 4 = Mostly correct, minor gaps (70-95%)
- 5 = Fully correct (95-100%)

**Note:** Only evaluates CONTENT, not wording/style.

**Difference from Faithfulness:**
| Metric | Compares Against | Measures |
|--------|-------------------|----------|
| Faithfulness | Retrieved context | Hallucination |
| Correctness | Reference answer | Factual accuracy |

**Example:**
```
Reference: "Phạt 800.000-1.000.000 đồng"
Answer: "Phạt tiền từ 800 nghìn đến 1 triệu đồng"
Correctness: 5 (content matches, different wording OK)
```

---

### 3. Answer Relevancy (1-5 Scale)

**Definition:** Measures if the answer addresses the question's main point.

**Scale:**
- 1 = Completely off-topic
- 2 = Missing the main point
- 3 = Partially addresses main point
- 4 = Addresses main point but verbose
- 5 = Addresses main point concisely

**Note:** This measures **relevance**, not correctness. An answer can be highly relevant but factually wrong.

**Example:**
```
Question: "Phạt xe máy vượt đèn đỏ bao nhiêu?"
Answer A: "Phạt 800k-1tr" → Relevancy: 5 (direct answer)
Answer B: "Luật giao thông quy định..." (long explanation) → Relevancy: 3-4
Answer C: "Đèn đỏ được phát minh năm 1868..." → Relevancy: 1 (off-topic)
```

---

### 4. Temporal Correctness (Binary 0/1)

**Definition:** Evaluates if the answer applies the correct effective law version.

**Novel Metric:** Not in Ragas/DeepEval, but critical for legal domain.

**What It Checks:**
- Answer cites correct law version (e.g., ND168/2024)
- Law version is currently effective (not expired, not future)
- Applies to the date/context in question

**Example:**
```
Question: "Năm 2025, phạt vượt đèn đỏ bao nhiêu?"
Contexts mention: ND168/2024 (effective 2025-01-01)

Answer A: "Theo Nghị định 168/2024, phạt..." → Temporal Correctness: 1 ✓
Answer B: "Theo Nghị định 100/2019, phạt..." → Temporal Correctness: 0 ✗ (expired)
```

---

## Why These 4 Metrics?

### Non-Redundant - Each Metric Captures Different Failure Modes

| Failure Mode | Faithfulness | Correctness | Relevancy | Temporal |
|--------------|--------------|-------------|-----------|----------|
| Hallucination | ❌ Low | ❌ Low | ➖ N/A | ➖ N/A |
| Wrong facts (grounded) | ✅ High | ❌ Low | ✅ High | ➖ N/A |
| Off-topic rambling | ✅ High | ➖ N/A | ❌ Low | ➖ N/A |
| Wrong law version | ✅ High | ❌ Low | ✅ High | ❌ Zero |

**Example Showing Non-Redundancy:**
```
Question: "Phạt vượt đèn đỏ năm 2025?"

Answer A: "Phạt 800k-1tr theo ND168/2024"
→ Faithfulness: 1.0, Correctness: 5, Relevancy: 5, Temporal: 1 ✓

Answer B: "Phạt 500k theo ND100/2019"
→ Faithfulness: 1.0, Correctness: 2, Relevancy: 5, Temporal: 0 ✗

Answer C: "Đèn giao thông ra đời năm 1868 tại London..."
→ Faithfulness: 1.0, Correctness: 1, Relevancy: 1, Temporal: N/A

Answer D: "Phạt 10 triệu đồng" (bịa)
→ Faithfulness: 0.0, Correctness: 1, Relevancy: 5, Temporal: N/A
```

---

## Comparison with Standards

### Ragas Metrics

| Ragas Metric | Your Equivalent | Status |
|--------------|-----------------|--------|
| Faithfulness (0-1) | ✅ Faithfulness (0-1) | ✅ MATCH |
| Answer Relevance (0-1) | ✅ Answer Relevancy (1-5) | ⚠️ Scale differs |
| Factual Correctness (0-1 F1) | ✅ Correctness (1-5) | ⚠️ Approach differs |
| Context Precision (0-1) | ✅ (metrics.py) | ✅ Has offline version |
| Context Recall (0-1) | ✅ (metrics.py) | ✅ Has offline version |

### DeepEval Metrics

| DeepEval Metric | Your Equivalent | Status |
|-----------------|-----------------|--------|
| Faithfulness (0-1) | ✅ Faithfulness (0-1) | ✅ MATCH |
| Answer Relevancy (0-1) | ✅ Answer Relevancy (1-5) | ⚠️ Scale differs |
| Hallucination (0-1) | ✅ Faithfulness (inverse) | ✅ Similar concept |
| Contextual Precision (0-1) | ❌ | ❌ Missing LLM version |
| Contextual Recall (0-1) | ❌ | ❌ Missing LLM version |

---

## Missing Metrics (Consider Adding)

### 1. Contextual Precision (LLM-based)

**Why Important:**
- Offline Precision@K doesn't capture semantic relevance
- LLM can judge if top-ranked chunks are truly relevant
- Measures retriever + re-ranker quality

### 2. Contextual Recall (LLM-based)

**Why Important:**
- Offline Recall@K is chunk-based
- LLM can judge if all NEEDED information was retrieved
- Critical for legal domain (must cite all relevant laws)

### 3. Contextual Relevancy

**Why Important:**
- Measures signal-to-noise ratio in retrieved context
- Too much noise → distracted LLM → worse generation
- Helps optimize chunk selection

---

## Implementation Quality

### Strengths

✅ **Faithfulness on correct scale (0.0-1.0)**
- Follows Ragas standard
- Enables statistical analysis

✅ **Clear separation of concerns**
- Offline metrics (deterministic) in `metrics.py`
- LLM metrics (semantic) in `evaluations/` harness

✅ **Domain-specific metric (Temporal Correctness)**
- Novel contribution for legal RAG
- Addresses real failure mode in legal domain

✅ **Non-redundant metrics**
- Each metric captures different failure mode
- Comprehensive coverage of RAG quality

### Action Items Completed

✅ Removed QUEST rubric (5 metrics) - too granular
✅ Removed MRR and NDCG - not critical for RAG
✅ Improved Faithfulness calculation prompt
✅ Clarified Correctness scale with percentages
✅ Documented all metrics with examples

---

## Running Evaluation

### Retrieval-Only (No LLM)

```powershell
python -m legal_rag.evaluation retrieve \
  --ground-truth data/benchmark_synthesis/groundtruth.jsonl \
  --k-values 1,3,5,10
```

### Full RAG Evaluation (LLM Judge)

```powershell
uv run legal-evaluate run \
  --pipeline naive \
  --limit 100 \
  --k-values 1,3,5,10
```

Output: `data/evaluation/naive_predictions.jsonl` + `naive_metrics.json`

---

## Conclusion

**Metrics Implementation: CORRECT ✅**

Your implementation follows Ragas/DeepEval standards with appropriate adaptations for legal domain:
- Faithfulness scale matches standard (0.0-1.0)
- Correctness and Answer Relevancy use LLM judge (acceptable)
- Temporal Correctness is novel and valuable addition
- Offline retrieval metrics provide deterministic baseline

**No critical gaps** - ready for production RAG evaluation.
