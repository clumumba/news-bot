from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nlp_pipeline.preprocessing import extract_keywords, normalize_text, split_sentences, tokenize


CATEGORY_KEYWORDS = {
    "business": {"market", "stocks", "trade", "profit", "economy", "earnings", "investors", "fiscal"},
    "health": {"health", "hospital", "medical", "doctor", "vaccine", "disease", "care", "public health"},
    "politics": {"election", "policy", "government", "senate", "congress", "parliament", "bill", "minister"},
    "science": {"research", "study", "scientists", "space", "climate", "discovery", "laboratory", "experiment"},
    "sports": {"team", "match", "season", "tournament", "league", "coach", "player", "championship"},
    "technology": {"ai", "software", "chip", "startup", "cloud", "app", "device", "cyber", "data"},
    "world": {"war", "diplomats", "border", "international", "global", "region", "allies", "peace"},
    "entertainment": {"film", "music", "show", "artist", "series", "festival", "streaming", "celebrity"},
}

POSITIVE_WORDS = {
    "gain",
    "gains",
    "growth",
    "improve",
    "improved",
    "improves",
    "success",
    "positive",
    "strong",
    "rise",
    "surge",
    "stable",
    "support",
    "win",
    "wins",
    "record",
    "optimistic",
    "safe",
    "relief",
}

NEGATIVE_WORDS = {
    "drop",
    "drops",
    "fall",
    "falls",
    "loss",
    "losses",
    "crisis",
    "decline",
    "declines",
    "weak",
    "risk",
    "risks",
    "concern",
    "concerns",
    "delay",
    "delays",
    "fail",
    "fails",
    "failure",
    "warning",
    "warnings",
}

LANGUAGE_CUES = {
    "en": {"the", "and", "with", "from", "will", "for", "news", "report", "government"},
    "es": {"el", "la", "de", "y", "que", "para", "gobierno", "salud", "tecnologia", "economia"},
    "fr": {"le", "la", "de", "et", "que", "pour", "gouvernement", "sante", "technologie", "economie"},
}

TRANSLATION_GLOSSARY = {
    "es": {
        "gobierno": "government",
        "economia": "economy",
        "salud": "health",
        "tecnologia": "technology",
        "cientificos": "scientists",
        "noticias": "news",
        "mercado": "market",
        "ciudad": "city",
        "equipo": "team",
        "voto": "vote",
        "ley": "law",
        "hospital": "hospital",
    },
    "fr": {
        "gouvernement": "government",
        "economie": "economy",
        "sante": "health",
        "technologie": "technology",
        "scientifiques": "scientists",
        "nouvelles": "news",
        "marche": "market",
        "ville": "city",
        "equipe": "team",
        "loi": "law",
        "hopital": "hospital",
    },
}

ENTITY_STOPWORDS = {
    "The",
    "A",
    "An",
    "This",
    "That",
    "These",
    "Those",
    "In",
    "On",
    "At",
    "With",
    "After",
    "Before",
    "New",
    "Today",
    "Week",
    "Month",
    "Year",
}

ENTITY_SUFFIXES = (
    "Inc",
    "Corp",
    "Corporation",
    "Company",
    "Ltd",
    "LLC",
    "Bank",
    "University",
    "Agency",
    "Council",
    "Ministry",
    "Committee",
    "Group",
    "Studio",
    "Laboratories",
    "Systems",
)

QUERY_ENTITY_STOPWORDS = {
    "Find",
    "Search",
    "Show",
    "Compare",
    "Summarize",
    "Tell",
    "Give",
    "Explain",
    "List",
    "What",
    "Who",
    "How",
    "News",
}

RELATION_PATTERNS = [
    (re.compile(r"(?P<source>[A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*)*)\s+(?:said|announced|reported|confirmed)\s+(?P<target>[^.?!]+)", re.IGNORECASE), "reported"),
    (re.compile(r"(?P<source>[A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*)*)\s+(?:acquired|bought|purchased|merged with)\s+(?P<target>[A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*)*)", re.IGNORECASE), "acquired"),
    (re.compile(r"(?P<source>[A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*)*)\s+(?:in|at|near|across)\s+(?P<target>[A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*)*)", re.IGNORECASE), "located"),
]


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _split_tokens(text: str) -> list[str]:
    return tokenize(text)


def _dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _top_n_from_counter(counter: Counter[str], top_k: int) -> list[str]:
    return [item for item, _ in counter.most_common(top_k)]


def _clean_keyword(token: str) -> str:
    return token.lower().strip()


def _article_text(article: dict[str, Any]) -> str:
    if article.get("text"):
        return _coerce_text(article["text"]).strip()
    parts = []
    for key in ("headline", "title", "content", "body"):
        value = article.get(key)
        if value:
            parts.append(_coerce_text(value))
    return " ".join(parts).strip()


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    return value


@dataclass
class NewsTopicModeler:
    n_topics: int = 5
    random_state: int = 42
    vectorizer: TfidfVectorizer | None = None
    model: NMF | None = None
    topic_labels: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.topic_labels is None:
            self.topic_labels = []

    def fit_topics(self, documents: Iterable[str]) -> "NewsTopicModeler":
        docs = [normalize_text(document) for document in documents if _coerce_text(document).strip()]
        if not docs:
            self.vectorizer = None
            self.model = None
            self.topic_labels = []
            return self

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5_000,
            min_df=1,
        )
        matrix = self.vectorizer.fit_transform(docs)
        n_components = max(1, min(self.n_topics, matrix.shape[0], max(1, matrix.shape[1])))
        self.model = NMF(n_components=n_components, init="nndsvda", random_state=self.random_state, max_iter=500)
        self.model.fit(matrix)

        feature_names = self.vectorizer.get_feature_names_out()
        self.topic_labels = []
        for topic_weights in self.model.components_:
            top_indices = topic_weights.argsort()[::-1][:5]
            self.topic_labels.append(", ".join(feature_names[index] for index in top_indices if topic_weights[index] > 0))
        return self

    def get_article_topics(self, article_text: str, top_k: int = 3) -> list[dict[str, Any]]:
        if not self.vectorizer or not self.model:
            return []
        matrix = self.vectorizer.transform([normalize_text(article_text)])
        topic_weights = self.model.transform(matrix)[0]
        if topic_weights.size == 0:
            return []
        feature_names = self.vectorizer.get_feature_names_out()
        rankings = topic_weights.argsort()[::-1][:top_k]
        topics = []
        for index in rankings:
            component = self.model.components_[index]
            token_indices = component.argsort()[::-1][:5]
            topics.append(
                {
                    "topic_id": int(index),
                    "weight": float(topic_weights[index]),
                    "label": self.topic_labels[index] if index < len(self.topic_labels) and self.topic_labels[index] else "",
                    "keywords": [feature_names[token_index] for token_index in token_indices if component[token_index] > 0],
                }
            )
        return topics

    def topic_overview(self) -> list[dict[str, Any]]:
        return [
            {"topic_id": index, "label": label}
            for index, label in enumerate(self.topic_labels)
        ]

    def track_topic_trends(self, articles_with_dates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        trend_counter: dict[str, Counter[int]] = defaultdict(Counter)
        for article in articles_with_dates:
            text = _article_text(article)
            topics = self.get_article_topics(text, top_k=1)
            if not topics:
                continue
            date_value = article.get("published_at") or article.get("timestamp") or article.get("date")
            if isinstance(date_value, datetime):
                date_key = date_value.date().isoformat()
            else:
                date_key = _coerce_text(date_value)[:10] if date_value else "unknown"
            trend_counter[date_key][topics[0]["topic_id"]] += 1

        trend_rows = []
        for date_key, counts in sorted(trend_counter.items()):
            for topic_id, count in counts.most_common():
                topic_label = self.topic_labels[topic_id] if topic_id < len(self.topic_labels) else ""
                trend_rows.append(
                    {
                        "date": date_key,
                        "topic_id": int(topic_id),
                        "topic_label": topic_label,
                        "count": int(count),
                    }
                )
        return trend_rows


@dataclass
class IntelligentSummarizer:
    def summarize_article(self, article_text: str, summary_type: str = "balanced") -> str:
        sentences = split_sentences(article_text)
        if not sentences:
            return _coerce_text(article_text).strip()
        if len(sentences) <= 2:
            return " ".join(sentences)

        target_sentences = {"brief": 1, "balanced": 2, "detailed": 3}.get(summary_type, 2)
        words = [token for token in _split_tokens(article_text) if token not in ENGLISH_STOP_WORDS]
        word_counts = Counter(words)

        scored_sentences = []
        for index, sentence in enumerate(sentences):
            sentence_tokens = [token for token in _split_tokens(sentence) if token not in ENGLISH_STOP_WORDS]
            if not sentence_tokens:
                continue
            raw_score = sum(word_counts[token] for token in sentence_tokens)
            length_penalty = math.sqrt(len(sentence_tokens))
            scored_sentences.append((index, raw_score / max(length_penalty, 1.0), sentence))

        if not scored_sentences:
            return " ".join(sentences[:target_sentences])

        top = sorted(scored_sentences, key=lambda item: item[1], reverse=True)[:target_sentences]
        ordered = [sentence for index, _, sentence in sorted(top, key=lambda item: item[0])]
        return " ".join(ordered)

    def summarize_multiple_articles(self, articles: Iterable[str], focus_topic: str | None = None) -> str:
        texts = [_coerce_text(article) for article in articles if _coerce_text(article).strip()]
        if focus_topic:
            filtered = [text for text in texts if focus_topic.lower() in text.lower()]
            if filtered:
                texts = filtered
        if not texts:
            return ""
        combined = "\n".join(texts)
        return self.summarize_article(combined, summary_type="balanced")

    def generate_headlines(self, article_text: str) -> list[str]:
        keywords = extract_keywords(article_text, top_k=5)
        if keywords:
            return [
                " ".join(word.capitalize() for word in keywords[:3]),
                " ".join(word.capitalize() for word in keywords[:4]),
            ]
        sentences = split_sentences(article_text)
        if sentences:
            first_sentence = sentences[0].strip().rstrip(".?!")
            return [first_sentence[:80]]
        return []

    def assess_summary_quality(self, original_text: str, summary: str) -> dict[str, float]:
        original_terms = set(extract_keywords(original_text, top_k=20))
        summary_terms = set(extract_keywords(summary, top_k=20))
        if not original_terms:
            return {"coverage_ratio": 0.0, "compression_ratio": 0.0}
        coverage_ratio = len(original_terms & summary_terms) / len(original_terms)
        compression_ratio = len(summary) / max(len(original_text), 1)
        return {
            "coverage_ratio": round(float(coverage_ratio), 4),
            "compression_ratio": round(float(compression_ratio), 4),
        }


@dataclass
class NewsSentimentTracker:
    positive_words: set[str] = None  # type: ignore[assignment]
    negative_words: set[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.positive_words is None:
            self.positive_words = set(POSITIVE_WORDS)
        if self.negative_words is None:
            self.negative_words = set(NEGATIVE_WORDS)

    def analyze_sentiment(self, article_text: str) -> dict[str, Any]:
        tokens = [_clean_keyword(token) for token in _split_tokens(article_text)]
        positive_hits = [token for token in tokens if token in self.positive_words]
        negative_hits = [token for token in tokens if token in self.negative_words]
        score = (len(positive_hits) - len(negative_hits)) / max(len(tokens), 1)
        if score > 0.03:
            label = "positive"
        elif score < -0.03:
            label = "negative"
        else:
            label = "neutral"
        confidence = min(1.0, 0.45 + abs(score) * 6)
        return {
            "label": label,
            "score": round(float(score), 4),
            "confidence": round(float(confidence), 4),
            "positive_terms": _dedupe_keep_order(positive_hits),
            "negative_terms": _dedupe_keep_order(negative_hits),
        }

    def track_sentiment_over_time(self, articles_with_dates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        daily_scores: dict[str, list[float]] = defaultdict(list)
        for article in articles_with_dates:
            text = _article_text(article)
            sentiment = self.analyze_sentiment(text)
            date_value = article.get("published_at") or article.get("timestamp") or article.get("date")
            if isinstance(date_value, datetime):
                date_key = date_value.date().isoformat()
            else:
                date_key = _coerce_text(date_value)[:10] if date_value else "unknown"
            daily_scores[date_key].append(sentiment["score"])

        timeline = []
        for date_key, scores in sorted(daily_scores.items()):
            timeline.append(
                {
                    "date": date_key,
                    "average_score": round(float(np.mean(scores)), 4),
                    "article_count": len(scores),
                }
            )
        return timeline

    def detect_sentiment_anomalies(self, sentiment_timeline: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = list(sentiment_timeline)
        if len(rows) < 3:
            return []
        scores = np.array([float(row.get("average_score", 0.0)) for row in rows], dtype=float)
        mean = float(scores.mean())
        std = float(scores.std()) or 1.0
        anomalies = []
        for row, score in zip(rows, scores, strict=False):
            z_score = abs((float(score) - mean) / std)
            if z_score >= 2.0:
                anomalies.append({**row, "z_score": round(float(z_score), 4)})
        return anomalies


@dataclass
class EntityRelationshipMapper:
    entity_pattern: re.Pattern[str] = re.compile(
        r"\b(?:[A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*)*)\b"
    )

    def _classify_entity(self, entity: str) -> str:
        if entity.endswith(ENTITY_SUFFIXES):
            return "organization"
        if entity.isupper() and len(entity) > 1:
            return "organization"
        if " " in entity and entity.split()[-1] in {"City", "State", "Country", "River", "Bay"}:
            return "location"
        if entity in ENTITY_STOPWORDS:
            return "other"
        return "person"

    def extract_entities(self, article_text: str) -> list[dict[str, Any]]:
        candidates = [match.group(0).strip() for match in self.entity_pattern.finditer(article_text)]
        entities = []
        for entity in _dedupe_keep_order(candidates):
            if entity in ENTITY_STOPWORDS or len(entity) < 2:
                continue
            first_token = entity.split()[0]
            if first_token in ENTITY_STOPWORDS:
                continue
            entities.append(
                {
                    "text": entity,
                    "type": self._classify_entity(entity),
                }
            )
        return entities

    def extract_relationships(self, article_text: str) -> list[dict[str, Any]]:
        relationships = []
        for pattern, relation_type in RELATION_PATTERNS:
            for match in pattern.finditer(article_text):
                relationships.append(
                    {
                        "relation": relation_type,
                        "source": _coerce_text(match.group("source")).strip(),
                        "target": _coerce_text(match.group("target")).strip().rstrip("."),
                    }
                )
        return relationships

    def build_knowledge_graph(self, articles: Iterable[dict[str, Any]]) -> dict[str, Any]:
        nodes: Counter[tuple[str, str]] = Counter()
        edges: Counter[tuple[str, str, str]] = Counter()
        for article in articles:
            text = _article_text(article)
            entities = self.extract_entities(text)
            for entity in entities:
                nodes[(entity["text"], entity["type"])] += 1
            relationships = self.extract_relationships(text)
            for relationship in relationships:
                edges[(relationship["source"], relationship["target"], relationship["relation"])] += 1
            entity_names = [entity["text"] for entity in entities]
            for left_index, source in enumerate(entity_names):
                for target in entity_names[left_index + 1 :]:
                    edges[(source, target, "co-mentioned")] += 1

        return {
            "nodes": [
                {"text": text, "type": entity_type, "count": int(count)}
                for (text, entity_type), count in nodes.items()
            ],
            "edges": [
                {"source": source, "target": target, "relation": relation, "count": int(count)}
                for (source, target, relation), count in edges.items()
            ],
        }

    def find_entity_connections(self, entity1: str, entity2: str, knowledge_graph: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        graph = knowledge_graph or {"edges": []}
        matches = []
        for edge in graph.get("edges", []):
            if {edge.get("source"), edge.get("target")} == {entity1, entity2}:
                matches.append(edge)
        return matches


@dataclass
class MultilingualProcessor:
    language_cues: dict[str, set[str]] = None  # type: ignore[assignment]
    translation_glossary: dict[str, dict[str, str]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.language_cues is None:
            self.language_cues = {language: set(words) for language, words in LANGUAGE_CUES.items()}
        if self.translation_glossary is None:
            self.translation_glossary = {language: dict(words) for language, words in TRANSLATION_GLOSSARY.items()}

    def detect_language(self, text: str) -> dict[str, Any]:
        lowered = normalize_text(text)
        tokens = set(_split_tokens(lowered))
        scores = {language: len(tokens & cues) for language, cues in self.language_cues.items()}
        accent_bonus = {
            "es": 1 if any(char in text for char in "ñáéíóú¿¡") else 0,
            "fr": 1 if any(char in text for char in "àâçéèêëîïôùûüÿœ") else 0,
        }
        for language, bonus in accent_bonus.items():
            scores[language] = scores.get(language, 0) + bonus
        language = max(scores, key=scores.get) if scores else "en"
        confidence = 0.35 + min(0.6, scores.get(language, 0) * 0.12)
        if scores.get(language, 0) == 0:
            language = "en"
            confidence = 0.45
        return {
            "language": language,
            "confidence": round(float(min(confidence, 0.99)), 4),
            "signals": scores,
        }

    def _translate_token(self, token: str, source_language: str) -> str:
        glossary = self.translation_glossary.get(source_language, {})
        lower = token.lower()
        translation = glossary.get(lower, token)
        if token[:1].isupper():
            return translation[:1].upper() + translation[1:]
        return translation

    def translate_text(self, text: str, target_language: str = "en") -> dict[str, Any]:
        detection = self.detect_language(text)
        source_language = detection["language"]
        if source_language == target_language:
            return {
                "translated_text": text,
                "source_language": source_language,
                "target_language": target_language,
                "confidence": 1.0,
                "note": "No translation needed.",
            }
        if target_language != "en" or source_language not in self.translation_glossary:
            return {
                "translated_text": text,
                "source_language": source_language,
                "target_language": target_language,
                "confidence": 0.35,
                "note": "Translation fallback returned the original text.",
            }

        tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
        translated_tokens = [self._translate_token(token, source_language) if token.isalpha() else token for token in tokens]
        translated_text = "".join(
            token if index == 0 or token in {",", ".", "!", "?", ":", ";"} else f" {token}"
            for index, token in enumerate(translated_tokens)
        )
        return {
            "translated_text": translated_text,
            "source_language": source_language,
            "target_language": target_language,
            "confidence": round(float(detection["confidence"] * 0.85), 4),
            "note": "Glossary-based translation.",
        }

    def analyze_cross_lingual(self, articles_by_language: dict[str, Iterable[str]]) -> dict[str, Any]:
        overview = []
        for language, articles in articles_by_language.items():
            article_list = list(articles)
            text_blob = " ".join(article_list)
            overview.append(
                {
                    "language": language,
                    "article_count": len(article_list),
                    "keywords": extract_keywords(text_blob, top_k=5),
                }
            )
        return {"languages": overview}

    def extract_cultural_context(self, text: str, source_language: str) -> dict[str, Any]:
        glossary = self.translation_glossary.get(source_language, {})
        matches = [term for term in glossary if term in normalize_text(text)]
        return {
            "source_language": source_language,
            "cultural_terms": _dedupe_keep_order(matches),
            "note": "Lightweight cultural context lookup.",
        }


@dataclass
class SemanticSearchEngine:
    vectorizer: TfidfVectorizer | None = None
    matrix: Any | None = None
    corpus_records: list[dict[str, Any]] = None  # type: ignore[assignment]
    corpus_texts: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.corpus_records is None:
            self.corpus_records = []
        if self.corpus_texts is None:
            self.corpus_texts = []

    def fit(self, corpus: Iterable[dict[str, Any] | str]) -> "SemanticSearchEngine":
        records = []
        texts = []
        for item in corpus:
            if isinstance(item, str):
                record = {"text": item}
            else:
                record = dict(item)
            text = _article_text(record) or _coerce_text(record.get("text"))
            if not text.strip():
                continue
            record["text"] = text
            records.append(record)
            texts.append(normalize_text(text))

        self.corpus_records = records
        self.corpus_texts = texts
        if not texts:
            self.vectorizer = None
            self.matrix = None
            return self

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5_000,
            min_df=1,
        )
        self.matrix = self.vectorizer.fit_transform(texts)
        return self

    def _search_scores(self, query_text: str) -> np.ndarray:
        if not self.vectorizer or self.matrix is None:
            return np.array([])
        query_vector = self.vectorizer.transform([normalize_text(query_text)])
        return cosine_similarity(query_vector, self.matrix).ravel()

    def semantic_search(
        self,
        query_text: str,
        article_database: Iterable[dict[str, Any] | str] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if article_database is not None:
            self.fit(article_database)
        scores = self._search_scores(query_text)
        if scores.size == 0:
            return []

        ranked_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for index in ranked_indices:
            record = self.corpus_records[int(index)]
            text = record.get("text", "")
            snippet = text[:240].rstrip()
            if len(text) > 240:
                snippet += "..."
            result = {
                "score": round(float(scores[int(index)]), 4),
                "snippet": snippet,
                "text": text,
            }
            for key in ("headline", "title", "category", "published_at", "source", "language"):
                if key in record:
                    result[key] = _json_safe_value(record[key])
            results.append(result)
        return results

    def find_similar_articles(self, query_article: str, top_k: int = 5) -> list[dict[str, Any]]:
        return self.semantic_search(query_article, top_k=top_k)

    def cluster_similar_content(self, articles: Iterable[dict[str, Any] | str], similarity_threshold: float = 0.3) -> list[list[dict[str, Any]]]:
        records = list(articles)
        if not records:
            return []
        self.fit(records)
        if not self.vectorizer or self.matrix is None:
            return []
        similarities = cosine_similarity(self.matrix)
        remaining = set(range(len(records)))
        clusters: list[list[dict[str, Any]]] = []
        while remaining:
            seed = remaining.pop()
            cluster_indices = {seed}
            for candidate in list(remaining):
                if similarities[seed, candidate] >= similarity_threshold:
                    cluster_indices.add(candidate)
                    remaining.remove(candidate)
            clusters.append([self.corpus_records[index] for index in sorted(cluster_indices)])
        return clusters


@dataclass
class NewsBotSystem:
    classifier_pipeline: Any | None = None
    articles: list[dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.topic_modeler = NewsTopicModeler()
        self.summarizer = IntelligentSummarizer()
        self.sentiment_tracker = NewsSentimentTracker()
        self.entity_mapper = EntityRelationshipMapper()
        self.multilingual = MultilingualProcessor()
        self.search_engine = SemanticSearchEngine()
        self.conversation = ConversationalInterface(self)
        self.articles = list(self.articles or [])
        self.classifier_labels = []
        if self.classifier_pipeline is not None:
            classifier = self.classifier_pipeline.named_steps.get("classifier")
            self.classifier_labels = list(getattr(classifier, "classes_", []))
        if self.articles:
            self.ingest_articles(self.articles)

    @staticmethod
    def _compose_text(article_text: str | None = None, metadata: dict[str, Any] | None = None) -> str:
        if metadata:
            text_value = metadata.get("text")
            if text_value:
                return _coerce_text(text_value).strip()
        parts = []
        if metadata:
            for key in ("headline", "title", "content", "body", "summary"):
                value = metadata.get(key)
                if value:
                    parts.append(_coerce_text(value))
        if article_text:
            parts.append(_coerce_text(article_text))
        return " ".join(part for part in parts if part).strip()

    def ingest_articles(self, articles: Iterable[dict[str, Any]]) -> None:
        normalized_articles = []
        for article in articles:
            record = dict(article)
            text = self._compose_text(metadata=record)
            record["text"] = text
            if "published_at" in record and isinstance(record["published_at"], str):
                try:
                    parsed = pd.to_datetime(record["published_at"], errors="coerce")
                    record["published_at"] = parsed.isoformat() if not pd.isna(parsed) else record["published_at"]
                except Exception:
                    pass
            for key, value in list(record.items()):
                record[key] = _json_safe_value(value)
            normalized_articles.append(record)
        self.articles = normalized_articles
        self.search_engine.fit(normalized_articles)
        self.topic_modeler.fit_topics(article["text"] for article in normalized_articles)

    def _fallback_category(self, article_text: str) -> dict[str, Any]:
        tokens = set(_split_tokens(article_text))
        scores = {
            category: len(tokens & {keyword.lower() for keyword in keywords})
            for category, keywords in CATEGORY_KEYWORDS.items()
        }
        best_category = max(scores, key=scores.get) if scores else "news"
        confidence = 0.35 + min(0.45, scores.get(best_category, 0) * 0.12)
        return {
            "label": best_category,
            "confidence": round(float(min(confidence, 0.95)), 4),
            "probabilities": {category: round(float(score / max(sum(scores.values()), 1)), 4) for category, score in scores.items()},
            "fallback": True,
        }

    def predict_category(self, article_text: str) -> dict[str, Any]:
        cleaned = normalize_text(article_text)
        if self.classifier_pipeline is None:
            return self._fallback_category(article_text)
        probabilities = self.classifier_pipeline.predict_proba([cleaned])[0]
        best_index = int(np.argmax(probabilities))
        label = self.classifier_labels[best_index] if self.classifier_labels else str(best_index)
        return {
            "label": label,
            "confidence": round(float(probabilities[best_index]), 4),
            "probabilities": {
                self.classifier_labels[index] if self.classifier_labels else str(index): round(float(probability), 4)
                for index, probability in enumerate(probabilities)
            },
            "fallback": False,
        }

    def analyze_article(self, article_text: str | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        text = self._compose_text(article_text=article_text, metadata=metadata)
        if not text.strip():
            raise ValueError("Article text is required for analysis.")

        language = self.multilingual.detect_language(text)
        translated = self.multilingual.translate_text(text) if language["language"] != "en" else {
            "translated_text": text,
            "source_language": "en",
            "target_language": "en",
            "confidence": 1.0,
            "note": "English text used as-is.",
        }
        prediction = self.predict_category(text)
        sentiment = self.sentiment_tracker.analyze_sentiment(text)
        entities = self.entity_mapper.extract_entities(text)
        relationships = self.entity_mapper.extract_relationships(text)
        topics = self.topic_modeler.get_article_topics(text)
        summary = self.summarizer.summarize_article(text, summary_type="balanced")
        keywords = extract_keywords(text, top_k=8)
        related_articles = self.search_engine.semantic_search(text, top_k=3) if self.search_engine.corpus_records else []
        return {
            "text": text,
            "classification": prediction,
            "sentiment": sentiment,
            "entities": entities,
            "relationships": relationships,
            "topics": topics,
            "summary": summary,
            "keywords": keywords,
            "language": language,
            "translation": translated,
            "related_articles": related_articles,
        }

    def generate_insights(self, articles: Iterable[dict[str, Any]] | None = None) -> dict[str, Any]:
        corpus = list(articles or self.articles)
        if not corpus:
            return {
                "article_count": 0,
                "category_distribution": {},
                "sentiment_distribution": {},
                "language_distribution": {},
                "top_keywords": [],
            }

        category_counts: Counter[str] = Counter()
        sentiment_counts: Counter[str] = Counter()
        language_counts: Counter[str] = Counter()
        keyword_counts: Counter[str] = Counter()
        entity_counts: Counter[str] = Counter()
        for article in corpus:
            text = _article_text(article)
            if not text:
                continue
            category_counts[self.predict_category(text)["label"]] += 1
            sentiment_counts[self.sentiment_tracker.analyze_sentiment(text)["label"]] += 1
            language_counts[self.multilingual.detect_language(text)["language"]] += 1
            keyword_counts.update(extract_keywords(text, top_k=6))
            for entity in self.entity_mapper.extract_entities(text):
                entity_counts[entity["text"]] += 1

        return {
            "article_count": len(corpus),
            "category_distribution": dict(category_counts),
            "sentiment_distribution": dict(sentiment_counts),
            "language_distribution": dict(language_counts),
            "top_keywords": [keyword for keyword, _ in keyword_counts.most_common(10)],
            "top_entities": [entity for entity, _ in entity_counts.most_common(10)],
            "topic_overview": self.topic_modeler.topic_overview(),
        }

    def process_query(self, user_query: str, conversation_context: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.conversation.process_query(user_query, conversation_context=conversation_context)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return self.search_engine.semantic_search(query, top_k=top_k)

    def summarize_articles(self, articles: Iterable[dict[str, Any] | str], focus_topic: str | None = None) -> str:
        texts = []
        for article in articles:
            if isinstance(article, str):
                texts.append(article)
            else:
                texts.append(_article_text(article))
        return self.summarizer.summarize_multiple_articles(texts, focus_topic=focus_topic)


@dataclass
class ConversationalInterface:
    newsbot: NewsBotSystem

    def classify_intent(self, user_query: str) -> str:
        lowered = normalize_text(user_query)
        rules = [
            ("translate", ["translate", "translation", "in spanish", "in french", "francais", "espanol"]),
            ("compare", ["compare", "versus", "vs", "difference", "contrast"]),
            ("summarize", ["summarize", "summary", "brief", "tl;dr", "overview"]),
            ("sentiment", ["sentiment", "positive", "negative", "tone", "mood"]),
            ("entities", ["who", "which company", "entity", "people", "organizations"]),
            ("topics", ["topic", "trend", "themes", "theme", "topic model"]),
            ("search", ["find", "search", "show me", "articles", "news about", "coverage"]),
        ]
        for intent, keywords in rules:
            if any(keyword in lowered for keyword in keywords):
                return intent
        return "search"

    def extract_query_entities(self, user_query: str) -> dict[str, Any]:
        lowered = normalize_text(user_query)
        categories = [category for category in CATEGORY_KEYWORDS if category in lowered]
        languages = []
        for language in ("spanish", "espanol", "es", "french", "francais", "fr"):
            if language in lowered:
                languages.append(language)
        timeframes = []
        for token in ("today", "this week", "this month", "latest", "recent", "yesterday"):
            if token in lowered:
                timeframes.append(token)
        named_entities = [match.group(0).strip() for match in EntityRelationshipMapper.entity_pattern.finditer(user_query)]
        named_entities = [entity for entity in named_entities if entity.split()[0] not in QUERY_ENTITY_STOPWORDS]
        return {
            "categories": _dedupe_keep_order(categories),
            "languages": _dedupe_keep_order(languages),
            "timeframes": _dedupe_keep_order(timeframes),
            "entities": _dedupe_keep_order(named_entities),
        }

    def _resolve_articles(self, query: str, conversation_context: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        if conversation_context and conversation_context.get("articles"):
            source_articles = conversation_context["articles"]
        else:
            source_articles = self.newsbot.articles
        if not source_articles:
            return []
        return self.newsbot.search_engine.semantic_search(query, source_articles, top_k=top_k)

    def generate_response(self, query_results: dict[str, Any], intent: str, entities: dict[str, Any]) -> dict[str, Any]:
        if intent == "help":
            message = (
                "I can search the corpus, summarize stories, compare coverage, analyze sentiment, "
                "extract entities, translate text, and surface topic trends."
            )
        elif intent == "translate":
            message = query_results.get("translated_text", "No translation available.")
        elif intent == "summarize":
            message = query_results.get("summary", "I could not generate a summary.")
        elif intent == "compare":
            message = query_results.get("comparison", "Comparison unavailable.")
        elif intent == "sentiment":
            message = f"Sentiment is {query_results.get('sentiment', {}).get('label', 'unknown')}."
        elif intent == "entities":
            message = "I found the most relevant named entities in the selected article set."
        elif intent == "topics":
            message = "I surfaced the strongest topic signals for the selected article set."
        else:
            message = query_results.get("answer", "Here is what I found.")
        return {
            "intent": intent,
            "answer": message,
            "entities": entities,
            "results": query_results,
        }

    def process_query(self, user_query: str, conversation_context: dict[str, Any] | None = None) -> dict[str, Any]:
        if not user_query.strip():
            return {
                "intent": "help",
                "answer": "Please ask about a topic, article, source, or trend.",
                "entities": {},
                "results": {},
            }

        intent = self.classify_intent(user_query)
        entities = self.extract_query_entities(user_query)
        articles = self._resolve_articles(user_query, conversation_context=conversation_context, top_k=5)

        if intent == "translate":
            if articles:
                translation = self.newsbot.multilingual.translate_text(articles[0]["text"])
                return self.generate_response(translation, intent, entities)
            translation = self.newsbot.multilingual.translate_text(user_query)
            return self.generate_response(translation, intent, entities)

        if intent == "summarize":
            summary_source = [article["text"] for article in articles] or [user_query]
            summary = self.newsbot.summarizer.summarize_multiple_articles(summary_source)
            return self.generate_response({"summary": summary, "articles": articles}, intent, entities)

        if intent == "compare":
            comparison = self._compare_articles(articles, entities)
            return self.generate_response({"comparison": comparison, "articles": articles}, intent, entities)

        if intent == "sentiment":
            analyzed = [self.newsbot.sentiment_tracker.analyze_sentiment(article["text"]) for article in articles]
            aggregate = self._aggregate_sentiment(analyzed)
            return self.generate_response({"sentiment": aggregate, "articles": articles}, intent, entities)

        if intent == "entities":
            found_entities = []
            for article in articles[:3]:
                found_entities.extend(self.newsbot.entity_mapper.extract_entities(article["text"]))
            return self.generate_response({"entities": found_entities, "articles": articles}, intent, entities)

        if intent == "topics":
            topics = self.newsbot.topic_modeler.topic_overview()
            trend = self.newsbot.topic_modeler.track_topic_trends(self.newsbot.articles[:20])
            return self.generate_response({"topics": topics, "trend": trend, "articles": articles}, intent, entities)

        search_results = articles or self.newsbot.search(user_query, top_k=5)
        headlines = [result.get("headline") or result.get("title") or result.get("snippet", "") for result in search_results[:5]]
        answer = "\n".join(headlines) if headlines else "I could not find matching articles."
        return self.generate_response({"answer": answer, "results": search_results}, "search", entities)

    def _aggregate_sentiment(self, sentiments: list[dict[str, Any]]) -> dict[str, Any]:
        if not sentiments:
            return {"label": "neutral", "score": 0.0, "confidence": 0.0}
        scores = np.array([float(item.get("score", 0.0)) for item in sentiments], dtype=float)
        average = float(scores.mean())
        label = "positive" if average > 0.03 else "negative" if average < -0.03 else "neutral"
        return {
            "label": label,
            "score": round(average, 4),
            "confidence": round(float(min(1.0, 0.5 + abs(average) * 6)), 4),
            "article_count": len(sentiments),
        }

    def _compare_articles(self, articles: list[dict[str, Any]], entities: dict[str, Any]) -> str:
        if len(articles) < 2:
            return "I need at least two relevant articles to compare coverage."
        sentiments = [self.newsbot.sentiment_tracker.analyze_sentiment(article["text"]) for article in articles[:2]]
        labels = [sentiment["label"] for sentiment in sentiments]
        categories = [article.get("category", "unknown") for article in articles[:2]]
        return (
            f"I found two relevant articles. Their categories are {categories[0]} and {categories[1]}, "
            f"with sentiment leaning {labels[0]} and {labels[1]}."
        )
