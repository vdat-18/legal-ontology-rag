"""LLM-judge evaluation harness for the Legal RAG pipelines.

Metrics (4 generation metrics via LLM judge):
- faithfulness (0.0-1.0): Percentage of claims supported by retrieved context
- correctness (1-5): Matching with expected answer (LLM judge)
- answer_relevancy (1-5): Relevance to the question (LLM judge)
- temporal_correctness (binary 0/1): Applying correct law version
"""
