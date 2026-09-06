# Multi-Turn Conversational RAG: Comprehensive Implementation Guide

## Executive Summary

This guide provides a structured approach to converting single-turn legal RAG systems into multi-turn conversational systems. Based on recent academic literature (2024-2026) and industry best practices, we outline key strategies, architecture decisions, and implementation priorities for Vietnamese legal domain applications.

---

## 1. Core Multi-Turn Strategies

### 1.1 Query Rewriting & Clarification

#### Overview
Query rewriting transforms context-dependent follow-up questions into standalone, self-contained queries suitable for retrieval systems.

#### Key Approaches

**A. LLM-Based Query Rewriting**
- Use an LLM to rewrite the follow-up question incorporating necessary context from conversation history
- Most effective for resolving coreferences (e.g., "What about Article 5?" → "What are the penalties specified in Article 5 of the 2024 Land Law?")
- Trade-off: Adds latency (~100-500ms per query)

**B. T5-Based Rewriters** 
- Fine-tune T5 models on (conversation_history, followup_question, rewritten_query) triples
- Faster than LLM-based approaches (~10-50ms)
- Requires annotated training data

**C. Iterative Clarification (ICR Framework - Cao et al., 2025)**
- Alternates between generating clarification questions and rewritten queries
- Handles multiple ambiguous references iteratively
- Achieves state-of-the-art on TopiOCQA and QReCC benchmarks

#### Implementation Priority: **HIGH** (Essential for follow-up questions)

**Code Pattern:**
```python
def rewrite_query_llm(query: str, chat_history: List[Message], model: str) -> str:
    """Rewrite follow-up query to be standalone."""
    prompt = f"""Given the conversation history and a follow-up question, 
    rewrite the follow-up question to be self-contained and independent.
    
    Conversation History:
    {format_history(chat_history)}
    
    Follow-up Question: {query}
    
    Rewritten Question:"""
    
    return llm_call(prompt, model)
```

### 1.2 Conversation History Management

#### Three Architectural Paradigms

**A. Full History Injection (Simplest)**
- Include complete dialogue history in every prompt
- Pros: Simple implementation, preserves all context
- Cons: Context window exhaustion (4K-128K token limit), quadratic attention cost
- Suitable for: Conversations < 10 turns

**B. Sliding Window with Summarization**
- Keep last N turns verbatim, summarize older turns
- Hybrid approach: `recent_history + summary_of_old_history`
- Reduces token count while preserving key information
- Trade-off: Summarization may lose important details

**C. Session-Level Memory Traces (CMT-RAG - Zhou et al., 2026)**
- Maintain persistent memory traces as session-level DAG
- Each turn generates structured reasoning traces (sub-questions, dependencies)
- Enables efficient recovery of relevant prior reasoning
- **Recommended for legal domain** due to complex multi-hop reasoning requirements

#### Context Window Optimization Strategies

1. **Token Budgeting**: Allocate fixed token budget per component
   - Recent history: 40% (last 3-5 turns)
   - Retrieved documents: 40%
   - System prompt + query: 20%

2. **Hierarchical Compression**: 
   - Level 1: Raw dialogue turns (recent)
   - Level 2: Turn-level summaries (medium age)
   - Level 3: Session summary (oldest)

3. **Entity-State Tracking** (for legal domain):
   - Track extracted entities: case_number, parties, articles_mentioned, legal_issues
   - Store as structured state alongside dialogue history
   - Inject relevant entities into prompt without full history

### 1.3 Intent Tracking & Slot Filling

#### Legal Consultation Intent Taxonomy

```
Primary Intents:
├── information_seeking
│   ├── article_interpretation
│   ├── case_reference_lookup
│   ├── procedure_inquiry
│   └── definition_request
├── case_analysis
│   ├── liability_assessment
│   ├── penalty_calculation
│   └── statute_limitation_check
├── comparison_request
│   ├── law_version_comparison
│   ├── article_comparison
│   └── jurisdictional_comparison
└── follow_up
    ├── clarification
    ├── deepening
    └── topic_shift
```

#### Slot Schema for Legal Context

```python
@dataclass
class LegalConversationState:
    jurisdiction: Optional[str] = "Vietnam"
    legal_domain: Optional[str] = None  # civil, criminal, administrative, etc.
    specific_law: Optional[str] = None  # e.g., "Bộ luật Dân sự 2015"
    articles_mentioned: Set[str] = field(default_factory=set)
    case_details: Dict[str, Any] = field(default_factory=dict)
    parties_involved: List[str] = field(default_factory=list)
    timeline_references: List[str] = field(default_factory=list)
    user_role: Optional[str] = None  # lawyer, citizen, business, etc.
    conversation_topic: Optional[str] = None
    pending_clarifications: List[str] = field(default_factory=list)
```

#### Intent-Specific Response Strategies

| Intent | Retrieval Strategy | Response Focus |
|--------|-------------------|----------------|
| article_interpretation | Targeted article retrieval | Official text + judicial interpretation |
| case_analysis | Multi-document synthesis | Precedents + statutory basis |
| comparison_request | Side-by-side retrieval | Structured comparison table |
| clarification | No retrieval needed | Direct answer from history |

### 1.4 Handling Context Switches & Topic Transitions

#### Detection Methods

**A. Semantic Similarity Thresholding**
```python
def detect_topic_shift(current_query: str, conversation_topic: str, 
                        threshold: float = 0.3) -> bool:
    """Detect if user has shifted to new topic."""
    similarity = compute_semantic_similarity(current_query, conversation_topic)
    return similarity < threshold
```

**B. Explicit Intent Classification**
- Train classifier to detect: `continue_topic` vs `switch_topic` vs `reference_previous`
- Use classifier output to determine retrieval strategy

#### Handling Strategies

1. **Topic Boundary Detection**: Mark conversation segments by topic
   ```
   Turn 1-3: Land law consultation (Topic A)
   Turn 4: Clarification on previous point (Topic A continue)
   Turn 5: New question about marriage law (Topic B - new segment)
   Turn 6: Back to land law (Topic A - return)
   ```

2. **Parallel Context Stacks**: Maintain separate context stacks per topic
   - Allows users to switch between topics without losing context
   - When user returns to prior topic, restore relevant context

3. **Cross-Reference Resolution**: When user explicitly references previous topic
   - "Going back to the land law question..."
   - Detect cross-reference and retrieve both current and prior topic context

---

## 2. Legal Domain-Specific Considerations

### 2.1 Maintaining Legal Context Across Turns

#### Legal Context State Management

```python
class LegalContextManager:
    """Manages legal-specific context across conversation turns."""
    
    def __init__(self):
        self.case_context = CaseContext()
        self.legal_framework = LegalFrameworkTracker()
        self.citation_manager = CitationManager()
        self.consistency_checker = ConsistencyChecker()
    
    def update_context(self, turn: Turn):
        # Extract and track legal entities
        entities = self.extract_legal_entities(turn.content)
        self.case_context.update(entities)
        
        # Track legal framework evolution
        if entities.law_references:
            self.legal_framework.add_references(entities.law_references)
        
        # Track citations for consistency
        if entities.citations:
            self.citation_manager.add_citations(entities.citations)
        
        # Check consistency with previous statements
        self.consistency_checker.check(turn, self.case_context)
    
    def get_context_for_retrieval(self, query: str) -> RetrievalContext:
        """Prepare context for retrieval based on current state."""
        return RetrievalContext(
            primary_law=self.legal_framework.get_primary_law(),
            relevant_articles=self.case_context.get_articles(),
            jurisdiction=self.case_context.jurisdiction,
            prior_citations=self.citation_manager.get_recent(),
            pending_clarifications=self.case_context.pending_questions
        )
```

#### Key Elements to Track

1. **Legal Framework**
   - Primary law being discussed (e.g., "Luật Đất đai 2024")
   - Referenced articles/clauses
   - Effective dates (some cited laws may be repealed)
   - Subordinate regulations (Nghị định, Thông tư)

2. **Case-Specific Details**
   - Subject matter dispute type
   - Parties involved and their roles
   - Timeline of events
   - Key facts established in conversation
   - Pending clarifications needed

3. **Consultation Context**
   - User's jurisdiction (province-level variations)
   - User's role (affects advice depth)
   - Stage of consultation (initial, analysis, conclusion)

### 2.2 Handling Follow-Up Questions Referencing Previous Answers

#### Reference Resolution Types

**A. Direct Reference**
- User: "Tại Điều 5 nói gì?" → System answered with Article 5 content
- User: "Nhưng khoản 2 của điều đó xử lý thế nào?"
- Resolution: "điều đó" → Article 5; retrieve clause 2 specifically

**B. Entity Tracking Reference**
- User: "Trong trường hợp này, vợ chồng có được chia đất không?"
- System: "Theo quy định tại Điều 217 Luật Đất đai 2024..."
- User: "Còn nếu họ đã ly thân 5 năm thì sao?"
- Resolution: Track case context (spousal property division), add ly thân condition

**C. Comparative Reference**
- User: "So với luật cũ thì điểm này khác gì?"
- System: "Luật Đất đai 2013 quy định..., trong khi luật 2024..."
- User: "Vậy quy định mới có thuận lợi hơn không?"
- Resolution: "quy định mới" = Law 2024 version; "điểm này" = previously discussed aspect

#### Implementation Pattern

```python
class ReferenceResolver:
    """Resolve references to previous answers in follow-up questions."""
    
    def resolve_query(self, query: str, conversation: Conversation) -> ResolvedQuery:
        # Step 1: Identify reference markers
        markers = self.extract_reference_markers(query)
        # Vietnamese: "điều đó", "khoản trên", "vấn đề này", "trong trường hợp đó"
        
        # Step 2: Link markers to previously mentioned entities
        resolved_entities = {}
        for marker in markers:
            antecedent = self.find_antecedent(marker, conversation)
            resolved_entities[marker] = antecedent
        
        # Step 3: Build expanded query
        expanded_query = self.substitute_references(query, resolved_entities)
        
        # Step 4: Add relevant context for retrieval
        retrieval_hints = self.get_retrieval_hints(resolved_entities)
        
        return ResolvedQuery(
            original=query,
            expanded=expanded_query,
            resolved_entities=resolved_entities,
            retrieval_hints=retrieval_hints
        )
```

### 2.3 Ensuring Legal Advice Consistency

#### Consistency Constraints

1. **Statutory Consistency**: Never contradict explicit legal provisions
2. **Interpretation Consistency**: If Article X means Y in turn 1, maintain same interpretation in turn 10
3. **Advice Consistency**: Don't change advice unless new facts emerge
4. **Citation Consistency**: Use same citation format throughout conversation

#### Implementation: Consistency Validator

```python
class LegalConsistencyValidator:
    """Validate consistency of legal advice across turns."""
    
    def __init__(self):
        self.legal_positions = {}  # Track positions taken on legal questions
        self.cited_authorities = CitationTracker()
        
    def validate_response(self, response: str, context: ConversationContext) -> ValidationResult:
        # Check 1: Extract legal positions from response
        positions = self.extract_legal_positions(response)
        
        # Check 2: Compare with prior positions
        conflicts = []
        for position in positions:
            prior_position = self.legal_positions.get(position.question)
            if prior_position and self.contradicts(position, prior_position):
                conflicts.append(ConsistencyConflict(
                    current=position,
                    prior=prior_position,
                    severity='high' if position.statutory_contradiction else 'medium'
                ))
        
        # Check 3: Validate citations consistency
        citation_issues = self.validate_citations(response, self.cited_authorities)
        
        # Check 4: Verify law currency (no repealed law citations)
        currency_issues = self.validate_law_currency(positions)
        
        return ValidationResult(
            is_consistent=len(conflicts) == 0,
            conflicts=conflicts,
            citation_issues=citation_issues,
            currency_issues=currency_issues,
            remediation_suggestions=self.generate_remediations(conflicts)
        )
```

#### Handling Inconsistencies

When inconsistency detected:

```python
def handle_inconsistency(conflict: ConsistencyConflict, response_draft: str) -> str:
    """Modify response to address inconsistency."""
    if conflict.severity == 'high':
        # Explicit acknowledgment needed
        return f"""Lưu ý: Trong câu trả lời trước, chúng tôi đã nêu rằng {conflict.prior.position}.
        Tuy nhiên, sau khi xem xét kỹ hơn quy định tại {conflict.current.authority}, 
        chính xác hơn thì {conflict.current.position}.
        
        {response_draft}
        
        (Đã cập nhật để đảm bảo tính chính xác)
        """
    else:
        # Subtle correction within response
        return response_draft
```

### 2.4 Managing References to Legal Documents

#### Citation Management System

```python
@dataclass
class LegalCitation:
    law_name: str  # e.g., "Bộ luật Dân sự 2015"
    law_abbreviation: str  # e.g., "BLDS 2015"
    article: str  # e.g., "Điều 217"
    clause: Optional[str] = None  # e.g., "Khoản 2"
    point: Optional[str] = None  # e.g., "Điểm a"
    effective_date: str
    status: str  # "effective", "amended", "repealed"
    amending_law: Optional[str] = None
    
class CitationManager:
    """Track and manage legal citations throughout conversation."""
    
    def __init__(self):
        self.citations_by_turn: Dict[int, List[LegalCitation]] = {}
        self.primary_law: Optional[str] = None
        self.secondary_laws: Set[str] = set()
        
    def register_citation(self, turn_id: int, citation: LegalCitation):
        # Validate citation format
        if not self.validate_citation(citation):
            raise InvalidCitationError(citation)
        
        # Track law usage
        if self.primary_law is None:
            self.primary_law = citation.law_name
        elif citation.law_name != self.primary_law:
            self.secondary_laws.add(citation.law_name)
        
        self.citations_by_turn[turn_id].append(citation)
    
    def get_citation_context(self, current_turn: int, window: int = 3) -> str:
        """Get recent citations for context window."""
        recent_turns = range(max(0, current_turn - window), current_turn + 1)
        citations = []
        for turn_id in recent_turns:
            if turn_id in self.citations_by_turn:
                citations.extend(self.citations_by_turn[turn_id])
        return self.format_citation_context(citations)
```

#### Short Citation Notation

When referencing previously mentioned articles:
- First mention: "Theo Điều 217 Bộ luật Dân sự 2015..."
- Subsequent: "Theo Điều 217 nêu trên..." or "Theo điều luật đã dẫn..."
- In same turn: "Theo Điều 217..." then "Khoản 2 của điều này..."

---

## 3. Technical Implementation Approaches

### 3.1 Where to Inject Conversation History

#### Three-Stage Pipeline Options

**Option A: Retrieval-Stage Only** (Recommended for most cases)
```
User Query → Query Rewriter (uses history) → Retrieval → Generation (uses retrieved docs only)
```
- Pros: Clean separation, efficient caching
- Cons: Generation lacks dialogue context

**Option B: Generation-Stage Only**
```
User Query → Retrieval (standalone query) → Generation (uses history + retrieved docs)
```
- Pros: Generation has full context
- Cons: Retrieval may miss context-dependent nuances

**Option C: Both Stages** (Best quality, higher cost)
```
User Query → Query Rewriter → Retrieval → Context Selector (from history) → Generation
```
- Pros: Optimal retrieval and generation
- Cons: Higher computational cost

#### Recommended Architecture for Legal Domain: Hybrid

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTI-TURN LEGAL RAG PIPELINE                 │
└─────────────────────────────────────────────────────────────────┘

Input: user_query, conversation_history, legal_context_state

Step 1: Query Analysis
├── Intent Classification (information_seeking, case_analysis, etc.)
├── Reference Resolution (resolve "điều đó", "vấn đề trên")
└── Entity Extraction (laws, articles, legal concepts)

Step 2: Query Reformulation
├── If follow-up question: Rewrite to standalone using history
├── If reference to prior: Expand query with resolved entities
└── If topic switch: Reset context, inform downstream

Step 3: Retrieval Augmentation
├── Primary Retrieval: Vector search with rewritten query
├── Contextual Retrieval: Additional docs based on tracked entities
│   └── If Article X mentioned, retrieve Article X + related articles
├── Hierarchical Retrieval: Child chunks → Parent document reconstruction
└── Hybrid Reranking: Fuse dense + BM25 results, rerank with cross-encoder

Step 4: Context Assembly
├── Retrieved documents (primary + contextual)
├── Recent conversation history (last 3-5 turns)
├── Structured legal state (tracked entities, citations)
└── System prompt with legal domain instructions

Step 5: Generation
├── LLM generates response with all context
├── Citation grounding: Response must cite retrieved passages
└── Consistency check: Compare with prior positions

Step 6: Post-Processing
├── Extract and validate citations
├── Update legal context state
├── Check consistency with prior responses
└── Register turn for future reference resolution

Output: response, updated_state, citations, metadata
```

### 3.2 Re-Retrieve vs. Cache Previous Results

#### Decision Framework

```python
def should_retrieve(new_query: str, previous_results: List[Document], 
                    conversation_state: LegalContextState) -> Tuple[bool, str]:
    """
    Determine if new retrieval is needed or if cached results suffice.
    
    Returns: (should_retrieve: bool, reason: str)
    """
    
    # Always re-retrieve if:
    # 1. New topic introduced
    if conversation_state.topic_changed(new_query):
        return True, "topic_shift"
    
    # 2. New legal framework introduced
    entities = extract_legal_entities(new_query)
    if entities.new_law_references and entities.new_law_references not in conversation_state.laws_discussed:
        return True, "new_law_framework"
    
    # 3. Query is follow-up requiring different aspect
    intent = classify_intent(new_query)
    if intent.requires_different_retrieval:
        return True, "different_aspect"
    
    # 4. Previous results were insufficient (user asked for clarification)
    if conversation_state.last_turn_was_clarification_request:
        return True, "augmentation_needed"
    
    # Can use cached results if:
    # 1. Query is reference to same content
    if is_reference_query(new_query) and relevant_docs_in_cache(new_query, previous_results):
        return False, "reference_to_cached"
    
    # 2. Query asks for interpretation of already-retrieved content
    if is_interpretation_query(new_query) and previous_results_contains_relevant(new_query, previous_results):
        return False, "interpretation_of_cached"
    
    # 3. Meta-question about conversation (e.g., "What did you say about Article 5?")
    if is_meta_question(new_query):
        return False, "meta_question"
    
    # Default: Re-retrieve to ensure completeness
    return True, "default_comprehensive"
```

#### Hybrid Strategy: Layered Retrieval

```python
class LayeredRetrievalStrategy:
    """Combine cached and new retrieval results."""
    
    def retrieve(self, query: str, conversation: Conversation) -> List[Document]:
        
        # Layer 1: Check cache for directly relevant documents
        cached_docs = self.check_cache(query, conversation.previous_results)
        
        # Layer 2: Always perform lightweight retrieval to check for new relevant docs
        new_docs = self.incremental_retrieval(
            query=query,
            exclude=cached_docs,
            top_k=5  # Smaller than full retrieval
        )
        
        # Layer 3: If query references tracked entities, fetch entity-specific docs
        entity_docs = []
        for entity in conversation.tracked_entities:
            if entity.is_relevant_to(query):
                entity_docs.extend(self.entity_specific_retrieval(entity))
        
        # Merge and deduplicate
        all_docs = self.merge_results(
            cached=cached_docs,
            new=new_docs,
            entity_specific=entity_docs,
            max_total=20
        )
        
        # Rerank with query+context
        reranked = self.rerank(all_docs, query, conversation.context_summary)
        
        return reranked[:10]  # Final top-k for generation
```

### 3.3 Query Expansion Techniques

#### A. Historical Context Expansion
```python
def expand_with_history(query: str, history: List[Turn]) -> ExpandedQuery:
    """Expand query using conversation history."""
    
    # Extract key entities from recent history
    recent_entities = extract_entities_from_history(history[-3:])
    
    # Extract topic context
    topic_summary = summarize_topic(history)
    
    # Build expansion
    expansion = []
    for entity in recent_entities:
        if entity.is_referenced(query):  # "điều đó" references Article X
            expansion.append(f"{entity.full_name}")
    
    for legal_concept in recent_entities.legal_concepts:
        if is_same_concept_domain(query, legal_concept):
            expansion.append(f"{legal_concept} trong bối cảnh {topic_summary}")
    
    return ExpandedQuery(
        original=query,
        expansion_terms=expansion,
        rewritten=f"{query} {' '.join(expansion)}"
    )
```

#### B. Legal Concept Expansion
```python
def expand_legal_concepts(query: str) -> List[str]:
    """Expand query with legal terminology."""
    
    # Identify legal concepts in query
    concepts = extract_legal_concepts(query)
    
    expansions = []
    for concept in concepts:
        # Add synonyms and related terms
        if concept in LEGAL_TERM_SYNONYMS:
            expansions.extend(LEGAL_TERM_SYNONYMS[concept])
        
        # Add abbreviation expansion
        if concept in LAW_ABBREVIATIONS:
            expansions.append(LAW_ABBREVIATIONS[concept])
    
    return expansions

LEGAL_TERM_SYNONYMS = {
    "thừa kế": ["thừa kế theo pháp luật", "thừa kế theo di chúc", "kế thừa"],
    "ly hôn": ["ly hôn thuận tình", "ly hôn đơn phương", "chấm dứt hôn nhân"],
    "đất đai": ["quyền sử dụng đất", "thửa đất", "đất ở", "đất nông nghiệp"],
}

LAW_ABBREVIATIONS = {
    "BLDS": "Bộ luật Dân sự",
    "LĐĐ": "Luật Đất đai",
    "LHNG": "Luật Hôn nhân và Gia đình",
}
```

#### C. Multi-Query Expansion (HyDE-inspired)
```python
def multi_query_expansion(query: str, conversation_context: str) -> List[str]:
    """Generate multiple query variants for comprehensive retrieval."""
    
    # Variant 1: Direct rewrite
    variant_1 = rewrite_with_context(query, conversation_context)
    
    # Variant 2: Hypothetical document (HyDE)
    variant_2 = generate_hypothetical_answer(query, conversation_context)
    
    # Variant 3: Entity-focused (if entities mentioned)
    entities = extract_entities(query)
    variant_3 = f"{query} {entities.law} {entities.article}"
    
    # Variant 4: Question-focused (what question would this answer?)
    variant_4 = invert_to_question(query)
    
    return [variant_1, variant_2, variant_3, variant_4]
```

### 3.4 Hybrid Approaches: Semantic + Lexical Multi-Turn

#### Architecture

```python
class HybridMultiTurnRetriever:
    """Combine semantic and lexical retrieval for multi-turn conversations."""
    
    def __init__(self):
        self.dense_retriever = DenseRetriever(embedding_model="text-embedding-3-small")
        self.sparse_retriever = SparseRetriever()  # BM25 or SPLADE
        self.reranker = CrossEncoderReranker(model="bge-reranker-v2-m3")
        
    def retrieve(self, query: str, conversation: Conversation, 
                 top_k: int = 20) -> List[ScoredDocument]:
        
        # Step 1: Prepare queries for different retrievers
        semantic_query = self.prepare_semantic_query(query, conversation)
        lexical_query = self.prepare_lexical_query(query, conversation)
        
        # Step 2: Parallel retrieval
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_dense = executor.submit(
                self.dense_retriever.retrieve, semantic_query, top_k=top_k
            )
            future_sparse = executor.submit(
                self.sparse_retriever.retrieve, lexical_query, top_k=top_k
            )
            
            dense_results = future_dense.result()
            sparse_results = future_sparse.result()
        
        # Step 3: Reciprocal Rank Fusion
        fused_results = self.reciprocal_rank_fusion(
            dense_results, sparse_results, k=60
        )
        
        # Step 4: Context-aware reranking
        reranked = self.rerank_with_context(
            results=fused_results[:50],
            query=query,
            conversation_context=conversation.get_context_summary(),
            legal_state=conversation.legal_state
        )
        
        return reranked[:top_k]
    
    def reciprocal_rank_fusion(self, dense_results, sparse_results, k=60):
        """RRF: score(d) = Σ 1/(k + rank(d))"""
        scores = {}
        
        for rank, doc in enumerate(dense_results, 1):
            scores[doc.id] = scores.get(doc.id, 0) + 1/(k + rank)
        
        for rank, doc in enumerate(sparse_results, 1):
            scores[doc.id] = scores.get(doc.id, 0) + 1/(k + rank)
        
        # Sort by score descending
        all_docs = {doc.id: doc for doc in dense_results + sparse_results}
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        return [all_docs[doc_id] for doc_id in sorted_ids]
```

#### Dense vs. Sparse Query Preparation

```python
def prepare_semantic_query(self, query: str, conversation: Conversation) -> str:
    """Prepare query for dense retrieval (semantic search)."""
    # Dense retrievers benefit from natural language expansion
    if conversation.is_followup(query):
        rewritten = self.rewrite_query_with_history(query, conversation)
        # Add context for disambiguation
        context = conversation.get_topic_summary()
        return f"{rewritten} (trong bối cảnh: {context})"
    return query

def prepare_lexical_query(self, query: str, conversation: Conversation) -> str:
    """Prepare query for sparse retrieval (BM25/SPLADE)."""
    # Sparse retrievers benefit from keyword expansion
    expanded_terms = []
    
    # Add resolved references as explicit terms
    references = self.resolve_references(query, conversation)
    expanded_terms.extend(references)
    
    # Add legal entity names and abbreviations
    entities = extract_legal_entities(query)
    if entities.law:
        expanded_terms.extend([entities.law.full_name, entities.law.abbreviation])
    if entities.article:
        expanded_terms.append(f"Điều {entities.article}")
    
    return f"{query} {' '.join(expanded_terms)}"
```

---

## 4. Best Practices from Academic Literature

### 4.1 Key Papers & Insights

#### Multi-Turn RAG Benchmarks & Methods

| Paper | Venue | Key Contribution | Relevance to Legal |
|-------|-------|------------------|-------------------|
| **MTRAG-UN** (Rosenthal et al., 2026) | arXiv | Benchmark for answerable, underspecified, nonstandalone questions | Direct relevance: tests unanswerable queries, ambiguous follow-ups |
| **CMT-RAG** (Zhou et al., 2026) | arXiv | Memory traces as DAG for multi-turn multi-hop RAG | Highly relevant: structured reasoning for multi-hop legal queries |
| **Comprehensive RAG Comparison** (Alushi et al., 2026) | EACL | Systematic comparison: Reranking, Hybrid BM25, HyDE outperform vanilla RAG | Practical: shows simple methods often beat complex ones |
| **ChatR1** (Lupart et al., 2025) | ACL 2026 | RL for conversational reasoning, intent-aware rewards | Moderate: RL approach for adaptive retrieval |
| **ConvSearch-R1** (Zhu et al., 2025) | EMNLP 2025 | Self-driven query reformulation without external supervision | Practical: eliminates need for labeled rewrite data |
| **DiSCo** (Lupart et al., 2024) | SIGIR 2025 | Knowledge distillation for efficient sparse conversational retrieval | Efficiency: deployment optimization |

#### Query Rewriting Methods

| Paper | Venue | Key Insight |
|-------|-------|-------------|
| **ICR** (Cao et al., 2025) | arXiv | Iterative clarification+rewriting outperforms end-to-end |
| **MSPA-CQR** (Cao et al., 2026) | ACL 2026 | Multi-faceted preference alignment for rewriting |
| **ConvSearch-R1** (Zhu et al., 2025) | EMNLP 2025 | Self-driven RL rewriting, 10% improvement on TopiOCQA |
| **QRACDR** (Mo et al., 2024) | CIKM 2024 | Align query representation with rewritten queries & relevant docs |

### 4.2 Retrieval-Augmented Conversation Patterns

#### Pattern 1: Rewrite-Retrieve-Generate (Baseline)
```
User Query → Rewrite Query → Retrieve → Generate
                ↑_________|
              (uses history)
```
- Simple but effective baseline
- Recommended starting point

#### Pattern 2: Multi-Hop Memory Trace (CMT-RAG)
```
Turn 1: Query → Decompose → Retrieve → Generate → Store Trace
Turn 2: Query → Identify Dependencies → Retrieve (new + trace) → Generate → Update Trace
```
- Best for complex multi-hop legal queries
- Higher implementation complexity

#### Pattern 3: Agentic Multi-Turn Search (ChatR1)
```
User Query → Agent decides: retrieve? clarify? respond directly?
           → If retrieve: query reformulation → search → evaluate
           → If insufficient: iterate retrieval
           → Generate response with retrieved context
```
- Most flexible, handles edge cases
- Higher latency and potential for failure

#### Pattern 4: Hierarchical Parent-Child (H-RAG)
```
Document → Child chunks (overlapping, fine-grained)
         → Parent document (full context)

Query → Retrieve child chunks → Aggregate parent docs → Generate with parents
```
- Context window optimization
- Avoids retrieval fragmentation

### 4.3 Key Empirical Findings

From **Comprehensive RAG Comparison (Alushi et al., 2026)**:

1. **Simple methods often win**: Reranking, Hybrid BM25, HyDE outperform complex techniques
2. **Dataset structure matters**: No single RAG strategy dominates across all datasets
3. **Performance degrades with turn length**: Average performance decreases from Turn 1 to Turn 4
4. **Advanced methods can fail**: Several advanced techniques perform worse than No-RAG baseline

From **MTRAG-UN Benchmark (Rosenthal et al., 2026)**:

1. **Systems struggle with**: Unanswerable queries, underspecified questions, nonstandalone questions
2. **Unclear responses**: Systems fail to communicate limitations clearly
3. **Multi-turn reasoning gaps**: Significant performance drop between single-turn and multi-turn

From **SemEval-2026 Task 8 Results**:

1. **Top systems**: nDCG@5 ~0.58, showing significant room for improvement
2. **Query diversity helps**: Multiple rewritten queries outperform heterogeneous retriever ensembles
3. **Answerability calibration**: Key bottleneck in end-to-end performance

---

## 5. Evaluation Metrics for Multi-Turn

### 5.1 Retrieval Quality Metrics

#### Turn-Level Metrics

```python
# Standard retrieval metrics applied per turn
def evaluate_turn_retrieval(predicted_docs: List[str], relevant_docs: Set[str], 
                            k: int = 10) -> Dict[str, float]:
    return {
        f"recall@{k}": recall_at_k(predicted_docs, relevant_docs, k),
        f"mrr@{k}": mrr_at_k(predicted_docs, relevant_docs, k),
        f"ndcg@{k}": ndcg_at_k(predicted_docs, relevant_docs, k),
        "precision_at_1": 1.0 if predicted_docs[0] in relevant_docs else 0.0
    }
```

#### Conversation-Level Metrics

```python
def evaluate_conversation_retrieval(turn_results: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate turn-level metrics to conversation level."""
    
    # Average across all turns
    avg_metrics = {
        f"avg_{metric}": np.mean([turn[metric] for turn in turn_results])
        for metric in turn_results[0].keys()
    }
    
    # Weighted average (later turns weighted higher, reflecting context accumulation)
    weights = np.exp(np.linspace(0, 1, len(turn_results)))  # Exponential weights
    weighted_metrics = {
        f"weighted_{metric}": np.average(
            [turn[metric] for turn in turn_results],
            weights=weights
        )
        for metric in turn_results[0].keys()
    }
    
    # Success rate (all relevant docs retrieved in at least one turn)
    conversation_recall = all(
        any(turn["recall@10"] > 0.5 for turn in turn_results)
        for _ in [1]  # Placeholder for different query sets
    )
    
    return {**avg_metrics, **weighted_metrics, "conversation_success_rate": conversation_recall}
```

#### Context-Aware Metrics

```python
def evaluate_context_utilization(conversation: Conversation, 
                                   retrieved_docs: List[List[str]]) -> Dict[str, float]:
    """Measure how well system utilizes conversation context."""
    
    # Context Utilization Score (CUS)
    # Does retrieval improve when context is available vs. standalone query?
    standalone_retrieval = retrieve_standalone(conversation.queries, conversation.corpus)
    contextual_retrieval = retrieved_docs
    
    cus = np.mean([
        ndcg_at_k(ctx, rel, 10) - ndcg_at_k(std, rel, 10)
        for ctx, std, rel in zip(contextual_retrieval, standalone_retrieval, conversation.relevant)
    ])
    
    # Reference Resolution Accuracy (RRA)
    # How often are references ("điều đó") resolved correctly?
    rra = compute_reference_resolution_accuracy(conversation)
    
    # Topic Tracking Accuracy (TTA)
    # How well does system track topic across turns?
    tta = compute_topic_tracking_accuracy(conversation)
    
    return {
        "context_utilization_score": cus,  # Higher is better
        "reference_resolution_accuracy": rra,
        "topic_tracking_accuracy": tta
    }
```

### 5.2 Generation Quality Metrics

#### Answer Quality
- **Groundedness/Attribution**: Does response cite retrieved docs?
- **Consistency**: Does response align with previous responses?
- **Completeness**: Does response answer all parts of the question?
- **Conciseness**: Is response appropriately sized?

```python
def evaluate_groundedness(response: str, retrieved_docs: List[str]) -> float:
    """Evaluate if response is grounded in retrieved documents."""
    
    # Method 1: Citation check (if response includes citations)
    citations = extract_citations(response)
    citation_coverage = len(citations) / max(1, len(response.split()) / 50)  # Expect ~1 citation per 50 words
    
    # Method 2: NLI-based entailment
    entailment_scores = [
        nli_entailment(doc, response) for doc in retrieved_docs
    ]
    max_entailment = max(entailment_scores)
    
    # Method 3: Overlap-based (simpler)
    response_ngrams = set(ngrams(response, 2))
    doc_ngrams = set().union(*[ngrams(doc, 2) for doc in retrieved_docs])
    overlap = len(response_ngrams & doc_ngrams) / len(response_ngrams)
    
    return (citation_coverage + max_entailment + overlap) / 3
```

#### Consistency Metrics

```python
def evaluate_consistency(responses: List[str], legal_positions: Dict[int, str]) -> float:
    """Evaluate consistency across conversation responses."""
    
    consistency_scores = []
    
    for i, response in enumerate(responses):
        # Compare current response with all previous
        for j in range(i):
            prior_response = responses[j]
            prior_positions = legal_positions.get(j, {})
            
            # Check for contradictions
            contradictions = detect_contradictions(response, prior_response)
            consistency_scores.append(1.0 if not contradictions else 0.0)
    
    return np.mean(consistency_scores) if consistency_scores else 1.0
```

### 5.3 Datasets for Evaluation

#### English Benchmarks

| Dataset | Domain | Turns | Size | Key Feature |
|---------|--------|-------|------|-------------|
| **TopiOCQA** | Wikipedia | 3-5 | 1,881 conversations | Topic switches |
| **QReCC** | Web | 3-5 | 14K conversations | Open-domain |
| **MTRAG-UN** | Multi-domain | 2-6 | 666 tasks | Unanswerable, underspecified |
| **MuMu-QA** | Multi-hop | 3-7 | Synthetic | Multi-hop, cross-turn deps |
| **LoCoMo** | Long-context | 10+ | 10 conversations | Long-term memory |

#### Vietnamese Legal Domain

**Challenge**: No existing Vietnamese legal multi-turn QA benchmark

**Recommended Action**:
1. Construct custom benchmark from Vietnamese legal corpus
2. Use template-based conversation generation
3. Annotate for reference resolution entities
4. Include edge cases: topic switches, clarification requests

**Existing Vietnamese Resources**:

| Resource | Type | Use Case |
|----------|------|----------|
| **Vietnamese Legal Corpus** | Documents | Retrieval corpus |
| **VNBerta** | NLU Model | Vietnamese embedding |
| **PhoBERT** | NLU Model | Vietnamese NER, classification |

### 5.4 Evaluation Framework

```python
class MultiTurnLEGALRAGEvaluator:
    """Comprehensive evaluation for legal multi-turn RAG."""
    
    def __init__(self, corpus: LegalCorpus, ground_truth: Dict, 
                 evaluator_llm: str = "gpt-4"):
        self.corpus = corpus
        self.ground_truth = ground_truth
        self.evaluator = LLMJudge(evaluator_llm)
    
    def evaluate_system(self, system: LegalRAGSystem, 
                        conversations: List[Conversation]) -> EvaluationReport:
        
        all_turn_metrics = []
        all_response_metrics = []
        
        for conv in conversations:
            system.reset()
            turn_metrics = []
            responses = []
            
            for turn in conv.turns:
                # Get system response
                response = system.process_turn(turn.query, turn.history)
                responses.append(response.response_text)
                
                # Evaluate retrieval
                retrieval_metrics = self.evaluate_retrieval(
                    query=turn.query,
                    retrieved_docs=response.retrieved_docs,
                    relevant_docs=self.ground_truth[turn.id]["relevant_docs"]
                )
                turn_metrics.append(retrieval_metrics)
                
                # Evaluate response
                response_metrics = self.evaluate_response(
                    response=response.response_text,
                    query=turn.query,
                    retrieved_docs=response.retrieved_docs,
                    ground_truth_answer=self.ground_truth[turn.id]["answer"],
                    legal_consistency=self.check_legal_consistency(
                        response, responses[:-1]
                    )
                )
                all_response_metrics.append(response_metrics)
            
            # Conversation-level aggregation
            conv_metrics = self.aggregate_turns(turn_metrics)
            all_turn_metrics.append(conv_metrics)
        
        return EvaluationReport(
            retrieval_metrics=self.aggregate_conversations(all_turn_metrics),
            response_metrics=self.aggregate_conversations(all_response_metrics),
            system_name=system.name,
            corpus_size=len(self.corpus),
            num_conversations=len(conversations)
        )
```

---

## 6. Recommended Architecture for Vietnamese Legal Domain

### 6.1 Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│              MULTI-TURN LEGAL RAG ARCHITECTURE (VIETNAMESE LEGAL)            │
└──────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│   User Query    │
│  (Vietnamese)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         1. QUERY PROCESSING LAYER                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ ┌───────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│ │ Intent Classifier │  │ Reference        │  │ Entity Extractor │          │
│ │                   │  │ Resolver         │  │                  │          │
│ │ info_seek         │  │ "điều đó" →      │  │ law: BLDS 2015   │          │
│ │ case_analysis     │  │ "Điều 217"       │  │ article: 217     │          │
│ │ comparison        │  │                  │  │ concept: thừa kế │          │
│ └───────────────────┘  └──────────────────┘  └──────────────────┘          │
│                                                                              │
│ ┌────────────────────────────────────────────────────────────────┐          │
│ │ Query Rewriter (LLM)                                           │          │
│ │ Input: query + resolved references + conversation history     │          │
│ │ Output: standalone query                                       │          │
│ └────────────────────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         2. RETRIEVAL LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────┐  ┌─────────────────┐  ┌────────────────────────┐       │
│  │ Dense Retriever│  │ Sparse Retriever│  │ Contextual Retriever  │       │
│  │                │  │ (BM25/SPLADE)   │  │                       │       │
│  │ PhoBERetrieve  │  │                 │  │ If entity tracked:    │       │
│  │ Vietnamese-    │  │ Vietnamese      │  │ fetch related docs    │       │
│  │ tuned          │  │ legal stopwords │  │ (e.g., article +      │       │
│  └────────┬───────┘  └────────┬────────┘  │      related articles)│       │
│           │                   │            └────────────┬──────────┘       │
│           │                   │                         │                   │
│           └───────────────────┴─────────────────────────┘                   │
│                               │                                             │
│                    ┌──────────▼──────────┐                                 │
│                    │ Reciprocal Rank     │                                 │
│                    │ Fusion (RRF)        │                                 │
│                    │ k=60                │                                 │
│                    └──────────┬──────────┘                                 │
│                               │                                             │
│                    ┌──────────▼──────────┐                                 │
│                    │ Legal-Aware         │                                 │
│                    │ Reranker           │                                 │
│                    │ Cross-Encoder      │                                 │
│                    │ (Vi-medical or    │                                 │
│                    │  multilingual)     │                                 │
│                    └──────────┬──────────┘                                 │
│                               │                                             │
└───────────────────────────────┼─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       3. CONTEXT ASSEMBLY LAYER                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Retrieved Documents (10-20 chunks)                                    │  │
│  │ - Primary law article                                                │  │
│  │ - Related articles (from citation network)                           │  │
│  │ - Explanatory notes                                                  │  │
│  │ - Precedents (if available)                                         │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Conversation Context (last 3-5 turns)                                │  │
│  │ - User queries                                                       │  │
│  │ - System responses                                                   │  │
│  │ - Extracted entities from prior turns                               │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Legal State (Tracked)                                                 │  │
│  │ - Primary law: BLDS 2015                                             │  │
│  │ - Articles discussed: {217, 218, 219}                                │  │
│  │ - Entities: {thừa kế theo pháp luật, con nuôi}                       │  │
│  │ - Jurisdiction: Vietnam (national)                                   │  │
│  │ - Pending clarifications: []                                         │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        4. GENERATION LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ System Prompt (Vietnamese Legal Assistant)                            │  │
│  │ - Role: Legal information assistant                                   │  │
│  │ - Constraints: Cite sources, note limitations, consistency           │  │
│  │ - Format: Clear structure, citations in format [Luật..., Điều...]   │  │
│  │ - Warning: Recommend consulting lawyer for specific cases           │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ LLM (Vietnamese-capable)                                              │  │
│  │ Options:                                                              │  │
│  │ - GPT-4 (best Vietnamese, API)                                       │  │
│  │ - Claude-3.5-Sonnet (good Vietnamese, API)                           │  │
│  │ - Qwen-2.5 (multilingual, open)                                      │  │
│  │ - PhoGPT (Vietnamese-specific, open)                                 │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  Output: Response with citations                                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       5. POST-PROCESSING LAYER                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐  │
│  │ Citation Validator   │  │ Consistency Checker   │  │ State Updater    │  │
│  │                      │  │                       │  │                  │  │
│  │ - Valid law name     │  │ - Check vs prior      │  │ - Update tracked │  │
│  │ - Valid article     │  │   positions           │  │   entities       │  │
│  │ - Effective status  │  │ - Flag conflicts      │  │ - Add citations  │  │
│  └──────────────────────┘  └──────────────────────┘  └──────────────────┘  │
│                                                                              │
│  Final Output:                                                              │
│  - Response text with citations                                            │
│  - Citations list (structured)                                              │
│  - Legal state update                                                     │
│  - Confidence score                                                        │
│  - Suggested clarifications (if needed)                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                         ┌──────┴──────┐
                         │ API Response │
                         │  to User     │
                         └─────────────┘
```

### 6.2 Key Components (Priority Order)

#### Priority 1: Essential (MVP)

1. **Query Rewriter** - Resolve references and create standalone queries
   - Implementation: LLM-based (GPT-4/Claude) with Vietnamese legal prompt
   - Effort: 2-3 days
   - Dependency: None

2. **Hybrid Retriever** - Dense + Sparse fusion
   - Implementation: PhoBERT embeddings + BM25 + RRF
   - Effort: 3-5 days
   - Dependency: Vietnamese legal corpus indexed

3. **Basic State Tracker** - Track current law, articles mentioned
   - Implementation: Simple key-value store per session
   - Effort: 2 days
   - Dependency: Entity extractor

#### Priority 2: Important (Production)

4. **Legal Entity Extractor** - NER for Vietnamese legal terms
   - Implementation: Fine-tune PhoBERT on Vietnamese legal corpus
   - Effort: 1-2 weeks (fine-tuning)
   - Dependency: Annotated Vietnamese legal corpus

5. **Reference Resolver** - Map "điều đó" to specific article
   - Implementation: Rule-based + LLM fallback
   - Effort: 3-5 days
   - Dependency: Entity extractor

6. **Citation Manager** - Track and validate citations
   - Implementation: Knowledge graph of Vietnamese laws
   - Effort: 1 week
   - Dependency: Legal ontology

#### Priority 3: Advanced (Quality Improvement)

7. **Consistency Validator** - Ensure legal advice consistency
   - Implementation: Legal position tracker + contradiction detection
   - Effort: 1-2 weeks
   - Dependency: All prior components

8. **Topic Tracker** - Detect topic switches
   - Implementation: Semantic similarity threshold + intent classifier
   - Effort: 3-5 days
   - Dependency: Embeddings

9. **Parent-Child Chunking** - Context window optimization
   - Implementation: Index chunks, retrieve parents
   - Effort: 3 days
   - Dependency: Re-indexing corpus

### 6.3 Vietnamese-Specific Considerations

#### Language Processing

```python
# Vietnamese-specific preprocessing
def preprocess_vietnamese_legal_query(query: str) -> str:
    # Normalize Vietnamese text
    query = normalize_vietnamese(query)
    
    # Expand legal abbreviations
    for abbrev, full in VIETNAMESE_LAW_ABBREVIATIONS.items():
        query = query.replace(abbrev, full)
    
    # Normalize legal term variants
    for variant, canonical in VIETNAMESE_LEGAL_TERMS.items():
        query = re.sub(rf'\b{variant}\b', canonical, query)
    
    return query

VIETNAMESE_LAW_ABBREVIATIONS = {
    "BLDS": "Bộ luật Dân sự",
    "BLHS": "Bộ luật Hình sự",
    "LĐĐ": "Luật Đất đai",
    "LHNG": "Luật Hôn nhân và Gia đình",
    "LLĐB": "Luật Lao động Bán thời gian",
    "NĐ": "Nghị định",
    "TT": "Thông tư",
    "QĐ": "Quyết định",
}

VIETNAMESE_LEGAL_TERMS = {
    "thừa kế": "thừa kế theo pháp luật",
    "thừa kế theo di chúc": "thừa kế theo di chúc",
    "sổ đỏ": "giấy chứng nhận quyền sử dụng đất",
    "sổ hồng": "giấy chứng nhận quyền sử dụng đất",
}
```

#### Vietnamese Embedding Models

| Model | Type | Vietnamese Support | Recommendation |
|-------|------|-------------------|----------------|
| **PhoBERT** | Encoder | Native | Use for NER, classification |
| **VNBerta** | Encoder | Native | Alternative to PhoBERT |
| **text-embedding-3-small** | API | Good | Production API |
| **bge-m3** | Open | Multilingual | Open-source embedding |
| **vi-simcse** | Encoder | Native | Vietnamese-specific |

#### Vietnamese Legal Prompt Template

```python
VIETNAMESE_LEGAL_SYSTEM_PROMPT = """Bạn là trợ lý pháp luật chuyên nghiệp, hỗ trợ thông tin pháp luật Việt Nam.

QUY TẮC TRẢ LỜI:
1. LUÔN viện dẫn nguồn cụ thể theo định dạng: [Tên văn bản, Điều Khoản]
   Ví dụ: Theo quy định tại [Bộ luật Dân sự 2015, Điều 217 Khoản 1]...
   
2. Khi trả lời nhiều vấn đề, sử dụng cấu trúc rõ ràng:
   - Vấn đề thứ nhất: ...
   - Vấn đề thứ hai: ...
   
3. Nếu thông tin không có trong văn bản được trích dẫn, nói rõ: "Theo thông tin từ vụ việc/văn bản bạn cung cấp..."
   
4. Nếu cần làm rõ thêm thông tin, hỏi người dùng trước khi trả lời chi tiết.

5. LUÔN kết thúc bằng lời khuyên: "Để có tư vấn chính xác cho trường hợp cụ thể, bạn nên tham khảo ý kiến luật sư."

6. Nếu phát hiện mâu thuẫn với câu trả lời trước đó, giải thích rõ nguyên nhân cập nhật.

7. SỬ DỤNG CHÍNH XÁC thuật ngữ pháp lý Việt Nam, tránh dùng từ lóng hoặc từ dân gian.
"""

VIETNAMESE_QUERY_REWRITE_PROMPT = """Bạn đang tham gia một cuộc hội thoại về pháp luật Việt Nam.

LỊCH SỬ HỘI THOẠI:
{conversation_history}

CÂU HỎI TIẾP THEO: {query}

YÊU CẦU: Viết lại câu hỏi tiếp theo thành câu hỏi độc lập, tự giải thích (không cần tham khảo lịch sử).

VÍ DỤ:
- Câu hỏi gốc: "Điều đó quy định thế nào?"
- Câu hỏi viết lại: "Điều 217 Bộ luật Dân sự 2015 quy định thế nào về thừa kế theo pháp luật?"

CÂU HỎI VIẾT LẠI:"""
```

---

## 7. Trade-offs Between Approaches

### 7.1 Query Rewriting Trade-offs

| Approach | Latency | Quality | Cost | Implementation | Best For |
|----------|---------|---------|------|----------------|----------|
| LLM Rewrite | High (200-500ms) | High | Medium | Easy | Complex queries, production |
| T5 Rewrite | Low (10-50ms) | Medium | Low | Medium | High traffic, real-time |
| No Rewrite | None | Low | None | Easy | Simple follow-ups |
| ICR (Iterative) | Very High (500-2000ms) | Very High | High | Complex | Enterprise legal consultation |

**Recommendation**: Start with LLM rewrite, optimize with T5 if latency becomes bottleneck.

### 7.2 History Management Trade-offs

| Approach | Context Quality | Token Cost | Complexity | Best For |
|----------|-----------------|------------|------------|----------|
| Full History | Perfect | Very High | None | Very short conversations |
| Sliding Window | Good for recent | Medium | Low | General purpose |
| Summarization | Medium | Low | Medium | Long conversations |
| Memory Traces (DAG) | High | Medium | High | Multi-hop legal queries |

**Recommendation**: Sliding window (last 3-5 turns) + state tracking for MVP. Add memory traces for advanced version.

### 7.3 Retrieval Strategy Trade-offs

| Strategy | Recall | Latency | Cost | Best For |
|----------|--------|---------|------|----------|
| Dense Only | Good | Low | Medium | Semantic similarity |
| Sparse Only (BM25) | Good for keywords | Very Low | Very Low | Legal text with specific terms |
| Hybrid (Dense + Sparse) | Very Good | Medium | Medium | Most cases |
| Hybrid + Rerank | Excellent | Higher | High | Production quality |
| Multi-Query Expansion | Excellent | High | High | Comprehensive retrieval |

**Recommendation**: Hybrid + Rerank as baseline. Add multi-query for important queries.

### 7.4 Context Injection Trade-offs

| Stage | Retrieval Quality | Generation Quality | Latency | Best For |
|-------|-------------------|--------------------|---------|----------|
| Retrieval Only | High | Medium | Low | Information-heavy queries |
| Generation Only | Medium | High | Low | Context-heavy queries |
| Both Stages | High | High | Medium | Production systems |
| Agentic (Dynamic) | Adaptive | Adaptive | Variable | Complex, uncertain queries |

**Recommendation**: Both stages for production MVP. Consider agentic for advanced use cases.

---

## 8. Relevant Papers & Resources

### 8.1 Core Multi-Turn RAG Papers (2024-2026)

#### Benchmarks & Evaluation

1. **MTRAG-UN: A Benchmark for Open Challenges in Multi-Turn RAG Conversations** (2026)
   - arXiv:2602.23184
   - Key contribution: 666 tasks with unanswerable, underspecified, nonstandalone questions
   - URL: https://arxiv.org/abs/2602.23184

2. **CMT-RAG: Complementary Memory Traces for Multi-turn Multi-hop RAG** (2026)
   - arXiv:2607.26470
   - Key contribution: Memory traces as DAG for cross-turn dependencies
   - URL: https://arxiv.org/abs/2607.26470

3. **Comprehensive Comparison of RAG Methods Across Multi-Domain Conversational QA** (2026)
   - arXiv:2602.09552
   - Key contribution: Systematic comparison, finding simple methods often win
   - URL: https://arxiv.org/abs/2602.09552

#### Query Rewriting Methods

4. **ChatR1: Reinforcement Learning for Conversational Reasoning and Retrieval Augmented QA** (2025)
   - arXiv:2510.13312
   - Key contribution: RL-based reasoning, intent-aware rewards
   - URL: https://arxiv.org/abs/2510.13312

5. **ICR: Iterative Clarification and Rewriting for Conversational Search** (2025)
   - arXiv:2509.05100
   - Key contribution: Alternating clarification and rewriting
   - URL: https://arxiv.org/abs/2509.05100

6. **ConvSearch-R1: Enhancing Query Reformulation with Reasoning via RL** (2025)
   - arXiv:2505.15776
   - Key contribution: Self-driven RL rewriting, 10% improvement
   - URL: https://arxiv.org/abs/2505.15776

7. **MSPA-CQR: Multi-Faceted Self-Consistent Preference Alignment for Query Rewriting** (2026)
   - arXiv:2604.06771
   - Key contribution: Multi-dimensional preference alignment
   - URL: https://arxiv.org/abs/2604.06771

8. **Aligning Query Representation with Rewritten Query and Relevance Judgments** (2024)
   - arXiv:2407.20189
   - Key contribution: QRACDR model, representation alignment
   - URL: https://arxiv.org/abs/2407.20189

#### Retrieval Methods

9. **DiSCo: LLM Knowledge Distillation for Efficient Sparse Retrieval in Conversational Search** (2024)
   - arXiv:2410.14609
   - Key contribution: Distillation for efficiency, 6-point recall gain
   - URL: https://arxiv.org/abs/2410.14609

10. **ContextualRetriever: Learning Contextual Retrieval for Robust Conversational Search** (2025)
    - arXiv:2509.19700
    - Key contribution: Context-aware embedding, intent-guided supervision
    - URL: https://arxiv.org/abs/2509.19700

### 8.2 Memory & Long-Context Papers

11. **Agentic Conversational Search with Contextualized Reasoning via RL** (2026)
    - arXiv:2601.13115
    - Key contribution: Mix-initiative action optimization via RL
    - URL: https://arxiv.org/abs/2601.13115

12. **RuleMem: Active Rule Memory for Long-Term Conversational Agents** (2026)
    - arXiv:2609.03915
    - Key contribution: Logical rules for evidence retrieval
    - URL: https://arxiv.org/abs/2609.03915

13. **SodaMem: Evidence-Grounded Temporal Graph Memory for LLM Agents** (2026)
    - arXiv:2608.08055
    - Key contribution: Temporal graph memory for currency tracking
    - URL: https://arxiv.org/abs/2608.08055

14. **Zero-Mem: Zero-Token Memory Operations for LLM Agents** (2026)
    - arXiv:2607.29377
    - Key contribution: Eliminates LLM calls from memory operations
    - URL: https://arxiv.org/abs/2607.29377

### 8.3 Evaluation & Benchmarks

15. **RAG-DIVE: Dynamic Approach for Multi-Turn Dialogue Evaluation in RAG** (2026)
    - arXiv:2604.16310
    - Key contribution: Dynamic, interaction-driven evaluation
    - URL: https://arxiv.org/abs/2604.16310

16. **SemEval-2026 Task 8: Multi-Turn RAG** (2026)
    - Multiple system papers (AILS-NTUA, H-RAG, uva-irlab-conv)
    - Key contribution: Standardized benchmark, leaderboard
    - URL: https://arxiv.org/search/?searchtype=all&query=SemEval-2026+Task+8

### 8.4 Legal Domain Papers

17. **Rhetorical-Role-Aware RAG for Legal QA over Indian Supreme Court Judgments** (2026)
    - arXiv:2608.06828
    - Key contribution: Rhetorical-based chunking, fusion retrieval
    - URL: https://arxiv.org/abs/2608.06828

18. **Ansari: A Retrieval-Grounded Islamic AI Assistant** (2026)
    - arXiv:2608.20390
    - Key contribution: 140,000 conversations, faith-sensitive deployment lessons
    - URL: https://arxiv.org/abs/2608.20390

### 8.5 Vietnamese NLP Resources

19. **PhoBERT: Pre-trained Language Models for Vietnamese** (2020)
    - VinAI Research
    - Key contribution: State-of-the-art Vietnamese encoder
    - URL: https://github.com/VinAIResearch/PhoBERT

20. **VNBerta: Vietnamese BERT Models** (2020)
    - FPT.AI
    - Key contribution: Vietnamese-specific BERT variants
    - URL: https://github.com/fpt-corp/vn-bert

### 8.6 Industry Resources

21. **Conversational Retrieval Agents** - LangChain Blog (2023)
    - URL: https://blog.langchain.dev/conversational-retrieval-agents/
    - Key contribution: Agent-based approach, tool memory

22. **LangSmith Documentation** - Agent Evaluation
    - URL: https://docs.smith.langchain.com/
    - Key contribution: Production evaluation tools

---

## 9. Implementation Roadmap

### Phase 1: MVP (Weeks 1-4)

**Week 1-2: Core Retrieval**
- [ ] Index Vietnamese legal corpus (dense + sparse)
- [ ] Implement basic hybrid retriever (PhoBERT + BM25)
- [ ] Add RRF fusion and simple reranking

**Week 3: Query Rewriting**
- [ ] Implement LLM-based query rewriter (Vietnamese prompt)
- [ ] Add basic reference resolution (last article mentioned)
- [ ] Test rewrite quality on sample follow-up questions

**Week 4: Integration**
- [ ] Build end-to-end pipeline
- [ ] Simple state tracker (current law, articles)
- [ ] Basic evaluation on 20 sample conversations

### Phase 2: Production Ready (Weeks 5-10)

**Week 5-6: Entity Extraction**
- [ ] Fine-tune PhoBERT for Vietnamese legal NER
- [ ] Build entity vocabulary (laws, articles, legal concepts)
- [ ] Integrate entity extractor into pipeline

**Week 7-8: State Management**
- [ ] Implement comprehensive legal state tracker
- [ ] Add citation manager
- [ ] Build knowledge graph for Vietnamese law relationships

**Week 9-10: Consistency & Validation**
- [ ] Implement citation validator
- [ ] Add consistency checker
- [ ] Build automated testing suite

### Phase 3: Advanced Features (Weeks 11-16)

**Week 11-12: Memory Traces**
- [ ] Implement DAG-based memory traces
- [ ] Add multi-hop reasoning support
- [ ] Test on complex legal queries

**Week 13-14: Evaluation & Optimization**
- [ ] Build Vietnamese legal multi-turn benchmark
- [ ] Implement comprehensive evaluation
- [ ] Optimize latency (consider T5 rewriter)

**Week 15-16: Deployment**
- [ ] Production deployment
- [ ] Monitoring and logging
- [ ] Documentation

---

## 10. Summary & Recommendations

### Key Takeaways

1. **Start Simple**: Reranking + Hybrid BM25 + HyDE outperform many complex methods
2. **Dataset Structure Matters**: Choose strategy based on your corpus characteristics
3. **Vietnamese-Specific**: Invest in proper Vietnamese NLP (PhoBERT, legal prompts)
4. **Legal Domain**: Track legal context state, validate citations, ensure consistency
5. **Evaluation**: Build custom Vietnamese legal benchmark for meaningful evaluation

### Recommended Implementation Order

1. **Hybrid Retriever** (Dense + BM25 + RRF + Rerank) - Essential
2. **Query Rewriter** (LLM-based, Vietnamese prompt) - Essential
3. **Basic State Tracker** (law, articles, entities) - Essential
4. **Entity Extractor** (Vietnamese legal NER) - Important
5. **Reference Resolver** - Important
6. **Citation Manager** - Important
7. **Consistency Validator** - Advanced
8. **Memory Traces (DAG)** - Advanced

### Top Papers to Read

1. MTRAG-UN (arXiv:2602.23184) - Benchmark for challenges
2. CMT-RAG (arXiv:2607.26470) - Memory traces architecture
3. Comprehensive RAG Comparison (arXiv:2602.09552) - Empirical findings
4. ConvSearch-R1 (arXiv:2505.15776) - RL query rewriting
5. DiSCo (arXiv:2410.14609) - Efficient retrieval

---

## Appendix A: Code Templates

### A.1 Complete Multi-Turn Pipeline

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime

@dataclass
class LegalCitation:
    law_name: str
    article: str
    clause: Optional[str] = None
    effective_date: Optional[str] = None
    
@dataclass
class LegalEntity:
    entity_type: str  # law, article, concept, party
    text: str
    normalized: str
    
@dataclass 
class Turn:
    id: int
    query: str
    response: str
    timestamp: datetime
    retrieved_docs: List[str]
    citations: List[LegalCitation]
    entities: List[LegalEntity]
    
@dataclass
class LegalConversationState:
    conversation_id: str
    turns: List[Turn] = field(default_factory=list)
    primary_law: Optional[str] = None
    articles_mentioned: set = field(default_factory=set)
    legal_concepts: set = field(default_factory=set)
    parties: List[str] = field(default_factory=list)
    
class MultiTurnLegalRAG:
    """Complete multi-turn legal RAG system for Vietnamese legal domain."""
    
    def __init__(self, config: Dict):
        self.query_rewriter = QueryRewriter(config["llm"])
        self.retriever = HybridLegalRetriever(config["retrieval"])
        self.entity_extractor = VietnameseLegalNER(config["ner_model"])
        self.state_manager = LegalStateManager()
        self.response_generator = LegalResponseGenerator(config["llm"])
        self.citation_validator = LegalCitationValidator(config["legal_kb"])
        
    def process_turn(self, query: str, conversation_id: str) -> Dict:
        """Process a single turn in the conversation."""
        
        # Load conversation state
        state = self.state_manager.load_state(conversation_id)
        
        # Step 1: Analyze query
        intent = self.classify_intent(query)
        references = self.resolve_references(query, state)
        entities = self.entity_extractor.extract(query)
        
        # Step 2: Rewrite query
        if self.is_followup(query, state):
            rewritten = self.query_rewriter.rewrite(
                query=query,
                history=state.get_recent_history(k=5),
                references=references,
                entities=entities
            )
        else:
            rewritten = query
        
        # Step 3: Retrieve documents
        retrieval_context = self.prepare_retrieval_context(state, entities)
        docs = self.retriever.retrieve(
            query=rewritten,
            context=retrieval_context,
            top_k=15
        )
        
        # Step 4: Generate response
        response = self.response_generator.generate(
            query=query,
            rewritten_query=rewritten,
            documents=docs,
            history=state.get_recent_history(k=3),
            state_context={
                "primary_law": state.primary_law,
                "articles_mentioned": list(state.articles_mentioned),
                "legal_concepts": list(state.legal_concepts)
            }
        )
        
        # Step 5: Validate and post-process
        validated_response = self.citation_validator.validate(response, docs)
        consistency = self.check_consistency(validated_response.text, state)
        
        # Step 6: Update state
        turn = Turn(
            id=len(state.turns),
            query=query,
            response=validated_response.text,
            timestamp=datetime.now(),
            retrieved_docs=docs,
            citations=validated_response.citations,
            entities=entities
        )
        state.turns.append(turn)
        
        # Update tracked entities
        for entity in entities:
            if entity.entity_type == "law" and state.primary_law is None:
                state.primary_law = entity.normalized
            elif entity.entity_type == "article":
                state.articles_mentioned.add(entity.normalized)
            elif entity.entity_type == "concept":
                state.legal_concepts.add(entity.normalized)
        
        self.state_manager.save_state(state)
        
        return {
            "response": validated_response.text,
            "citations": [c.__dict__ for c in validated_response.citations],
            "confidence": validated_response.confidence,
            "consistency_score": consistency.score,
            "rewritten_query": rewritten
        }
```

### A.2 Vietnamese Legal Prompt Templates

```python
VIETNAMESE_SYSTEM_PROMPT_LEGAL = """Bạn là trợ lý pháp luật chuyên nghiệp về pháp luật Việt Nam.

NHIỆM VỤ:
- Cung cấp thông tin pháp luật chính xác, dựa trên văn bản quy phạm pháp luật
- Giải thích các quy định pháp luật một cách dễ hiểu
- Hướng dẫn các thủ tục pháp lý cơ bản

QUY TẮC:
1. LUÔN viện dẫn nguồn cụ thể: [Tên văn bản, Điều Khoản]
2. Sử dụng VẢN BẢN GỐC khi trích dẫn quy định
3. Nếu câu hỏi không rõ, YÊU CẦU làm rõ trước khi trả lời
4. Kết thúc bằng: "Đây là thông tin tham khảo. Bạn nên tham khảo ý kiến luật sư cho trường hợp cụ thể."

VÍ DỤ TRẢ LỜI ĐÚNG:
"Theo quy định tại [Bộ luật Dân sự 2015, Điều 217 Khoản 1]:
'Pháp luật quy định thừa kế theo pháp luật trong các trường hợp sau đây: a) Không có di chúc...'
Do đó, trong trường hợp không có di chúc, tài sản sẽ được chia theo thứ tự ưu tiên quy định tại [Bộ luật Dân sự 2015, Điều 651]..."

VÍ DỤ TRẢ LỜI SAI (KHÔNG LÀM THEO):
"Theo luật thì anh em ruột sẽ được chia tài sản đều nhau..." [Thiếu viện dẫn cụ thể]
"""

VIETNAMESE_QUERY_REWRITE_TEMPLATE = """Viết lại câu hỏi sau thành câu hỏi độc lập, tự giải thích (không cần tham khảo lịch sử hội thoại).

LỊCH SỬ HỘI THOẠI:
{history}

CÂU HỎI GỐC: {query}

CÂU HỎI ĐÃ VIẾT LẠI:"""
```

---

## Appendix B: Evaluation Datasets

### B.1 Vietnamese Legal Multi-Turn Test Set Structure

```json
{
  "conversation_id": "conv_001",
  "domain": "civil_law",
  "specific_law": "Bộ luật Dân sự 2015",
  "difficulty": "medium",
  "turns": [
    {
      "turn_id": 1,
      "query": "Thừa kế theo pháp luật là gì?",
      "query_type": "definition",
      "is_standalone": true,
      "references": [],
      "expected_entities": [
        {"type": "concept", "text": "thừa kế theo pháp luật"},
        {"type": "law", "text": "Bộ luật Dân sự 2015", "normalized": "BLDS 2015"}
      ],
      "relevant_docs": [
        {"doc_id": "blds_2015_art_217", "relevance": "primary"},
        {"doc_id": "blds_2015_art_650", "relevance": "secondary"}
      ],
      "ground_truth_answer": "Thừa kế theo pháp luật là việc phân chia tài sản của người đã mất theo quy định của pháp luật khi không có di chúc hoặc di chúc không hợp pháp...",
      "min_citations": ["Điều 217 BLDS 2015"]
    },
    {
      "turn_id": 2,
      "query": "Trong trường hợp này, ai được hưởng thừa kế?",
      "query_type": "application",
      "is_standalone": false,
      "references": [
        {"marker": "Trong trường hợp này", "resolves_to": "thừa kế theo pháp luật"}
      ],
      "expected_entities": [],
      "relevant_docs": [
        {"doc_id": "blds_2015_art_651", "relevance": "primary"}
      ],
      "ground_truth_answer": "Trong trường hợp thừa kế theo pháp luật, theo quy định tại [Điều 651 BLDS 2015], người thừa kế theo pháp luật được chia theo thứ tự sau: 1) Hàng thừa kế thứ nhất:配偶, cha mẹ đẻ, cha mẹ nuôi, con đẻ, con nuôi...",
      "min_citations": ["Điều 651 BLDS 2015"]
    },
    {
      "turn_id": 3,
      "query": "Thế còn khoản 2 của điều đó quy định gì?",
      "query_type": "specification",
      "is_standalone": false,
      "references": [
        {"marker": "điều đó", "resolves_to": "Điều 651 BLDS 2015"}
      ],
      "expected_entities": [],
      "relevant_docs": [
        {"doc_id": "blds_2015_art_651_k2", "relevance": "primary"}
      ],
      "ground_truth_answer": "Theo Khoản 2 Điều 651 Bộ luật Dân sự 2015: 'Hàng thừa kế thứ hai gồm: ông nội, bà nội, ông ngoại, bà ngoại, anh ruột, chị ruột, em ruột của người để lại tài sản thừa kế...'",
      "min_citations": ["Điều 651 Khoản 2 BLDS 2015"],
      "consistency_check": {
        "must_align_with": "turn_2",
        "aspect": "hàng thừa kế"
      }
    }
  ],
  "conversation_level_metrics": {
    "topic_consistency": true,
    "reference_resolution_accuracy": 1.0,
    "citation_consistency": true
  }
}
```

---

## Document Metadata

**Version**: 1.0  
**Created**: 2026-09-06  
**Author**: Research synthesis from academic literature  
**Sources**: 20+ arXiv papers (2024-2026), LangChain documentation, Vietnamese NLP resources  
**Purpose**: Guide for converting single-turn legal RAG to multi-turn conversational system  
**License**: For internal use in legal-ontology-rag project

---

**End of Guide**
