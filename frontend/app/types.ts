/** Response shapes returned by the Graphiti knowledge-graph API. */

/** One supporting episode's provenance: the chunk and document a fact came from. */
export interface FactSource {
  chunk_id: string;
  source_name: string;
}

/** One graph fact together with every chunk that supports it. */
export interface RetrievedFact {
  fact: string;
  sources: FactSource[];
}

/** One answer plus the graph facts it was grounded in. */
export interface QueryResponse {
  query: string;
  rewritten_query: string;
  answer: string;
  sources: RetrievedFact[];
  unindexed_citations: string[];
}

/** One turn in the transcript. A pending turn has no answer yet. */
export interface Turn {
  question: string;
  response?: QueryResponse;
  error?: string;
}
