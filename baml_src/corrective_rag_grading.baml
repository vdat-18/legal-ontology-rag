class DocumentRelevanceEvaluation {
  score float @description(#"
    Retrieval relevance score from 0.0 to 1.0.
    1.0 means the document directly and completely answers the question.
    0.5 means the document is partial or ambiguous.
    0.0 means the document is irrelevant or misleading.
  "#)
  rationale string @description(#"Short Vietnamese explanation for the score."#)
}

function ScoreDocumentRelevance(query_vi: string, document_text: string) -> DocumentRelevanceEvaluation {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    You are a strict retrieval evaluator for Vietnamese road-traffic-law QA.
    Score whether the retrieved document can answer the question accurately.

    Scoring rubric:
    - 1.0: direct and complete legal evidence for the question.
    - 0.5: partial, incomplete, or ambiguous evidence.
    - 0.0: irrelevant, unsupported, or misleading evidence.

    Question:
    {{ query_vi }}

    Retrieved document:
    {{ document_text }}

    {{ ctx.output_format }}
  "#
}
