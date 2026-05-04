from __future__ import annotations

from dataclasses import dataclass, field

from nlp_pipeline.newsbot import SemanticSearchEngine


@dataclass
class EmbeddingStore:
    engine: SemanticSearchEngine = field(default_factory=SemanticSearchEngine)

    def fit(self, corpus):
        return self.engine.fit(corpus)

    def encode_documents(self, documents):
        self.fit(documents)
        return self.engine.matrix

    def semantic_search(self, query_text: str, article_database=None, top_k: int = 5):
        return self.engine.semantic_search(query_text, article_database=article_database, top_k=top_k)

    def cluster_similar_content(self, articles, similarity_threshold: float = 0.3):
        return self.engine.cluster_similar_content(articles, similarity_threshold=similarity_threshold)
