"""
retrieval/bm25_search.py
========================
High-performance, dependency-free Okapi BM25 lexical inverted index
for authoritative cricket event records.
Complements dense vector embeddings by providing exact-term keyword recall
for player names, stadium venues, weather conditions, overs, and status flags.
"""

import math
import re
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict

from database.mongo_client import get_mongo_database
from database.schema import CRICKET_EVENTS_COLLECTION

logger = logging.getLogger(__name__)

# Basic English stopwords that do not convey distinct cricket domain meaning
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself",
}

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    """Tokenizes text into lowercase alphanumeric tokens excluding basic stopwords."""
    if not text:
        return []
    tokens = TOKEN_PATTERN.findall(text.lower())
    return [t for t in tokens if t not in STOPWORDS]


class BM25Index:
    """
    Okapi BM25 index over authoritative cricket log records.
    Parameters:
        k1: Controls term frequency saturation (default 1.5).
        b: Controls document length normalization (default 0.75).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = []
        self.doc_lengths: List[int] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_term_freqs: List[Counter] = []
        self.inverted_index: Dict[str, List[int]] = defaultdict(list)
        self.idf: Dict[str, float] = {}
        self.avg_doc_len: float = 0.0
        self.corpus_size: int = 0
        self.is_indexed: bool = False

    def build_from_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Constructs inverted index and precalculates IDFs across all documents."""
        self.doc_ids = []
        self.doc_lengths = []
        self.doc_tokens = []
        self.doc_term_freqs = []
        self.inverted_index.clear()
        self.idf.clear()

        self.corpus_size = len(documents)
        if self.corpus_size == 0:
            self.avg_doc_len = 0.0
            self.is_indexed = True
            return

        total_length = 0
        doc_freq: Counter = Counter()

        for idx, doc in enumerate(documents):
            event_id = doc.get("event_id", f"EVT_{idx:04d}")
            self.doc_ids.append(event_id)

            # Build a rich searchable lexical representation with cricket domain synonyms
            event_type = doc.get("event", "")
            domain_synonyms = []
            if event_type == "wide":
                domain_synonyms.extend(["extra", "extras", "extra runs", "wide delivery", "illegal delivery", "conceded"])
            elif event_type == "no ball":
                domain_synonyms.extend(["extra", "extras", "extra runs", "no ball delivery", "illegal delivery", "conceded"])
            elif event_type == "stumping":
                domain_synonyms.extend(["dismissal", "wicket", "stumping", "stumped", "out"])
            elif event_type == "LBW given":
                domain_synonyms.extend(["lbw", "dismissal", "wicket", "leg before wicket", "out"])
            elif event_type == "catch taken":
                domain_synonyms.extend(["catch", "caught", "dismissal", "wicket", "out"])
            elif event_type == "wicket taken":
                domain_synonyms.extend(["wicket", "dismissal", "bowled", "out"])
            elif event_type == "boundary hit":
                domain_synonyms.extend(["boundary", "four", "4 runs", "boundary hit"])
            elif event_type == "six hit":
                domain_synonyms.extend(["six", "maximum", "6 runs", "six hit"])

            ts = doc.get("timestamp", "")
            ts_str = ""
            if ts:
                ts_str = str(ts)
                try:
                    dt = datetime.fromisoformat(str(ts).replace("Z", "").split()[0])
                    ts_str += f" {dt.strftime('%d %B %Y')} {dt.strftime('%B %d %Y')} {dt.strftime('%d %b %Y')}"
                except Exception:
                    pass

            role_keywords = []
            if re.search(r"Player \d+\s*\(\)", doc.get("commentary", "")):
                role_keywords.append("player role is missing from the commentary empty player role role missing")
            elif not doc.get("player_role"):
                role_keywords.append("missing role unstated role role missing no role unspecified role")

            text_parts = [
                doc.get("match_id", ""),
                doc.get("team", ""),
                doc.get("player", ""),
                " ".join(role_keywords),
                event_type,
                " ".join(domain_synonyms),
                doc.get("status", ""),
                f"over {doc.get('over', '')}",
                f"{doc.get('runs', '')} runs",
                ts_str,
                doc.get("commentary", ""),
            ]
            full_text = " ".join(part for part in text_parts if part)
            tokens = tokenize(full_text)
            tf = Counter(tokens)

            self.doc_tokens.append(tokens)
            self.doc_term_freqs.append(tf)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_length += doc_len

            unique_terms = set(tokens)
            for term in unique_terms:
                self.inverted_index[term].append(idx)
                doc_freq[term] += 1

        self.avg_doc_len = total_length / self.corpus_size if self.corpus_size > 0 else 0.0

        # Calculate Okapi BM25 IDF: ln((N - n + 0.5) / (n + 0.5) + 1)
        for term, freq in doc_freq.items():
            self.idf[term] = math.log(((self.corpus_size - freq + 0.5) / (freq + 0.5)) + 1.0)

        self.is_indexed = True
        logger.info(
            "BM25 index built successfully: %d documents, %d unique terms, avg_len=%.1f",
            self.corpus_size,
            len(self.idf),
            self.avg_doc_len,
        )

    def search(self, query: str, limit: int = 50, allowed_ids: Optional[set] = None) -> List[Tuple[str, float]]:
        """
        Executes BM25 ranking against the index for the given query.
        Returns a ranked list of (event_id, score) tuples.
        """
        if not self.is_indexed or self.corpus_size == 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores: Dict[int, float] = defaultdict(float)

        for token in query_tokens:
            if token not in self.inverted_index:
                continue

            idf = self.idf.get(token, 0.0)
            matching_doc_indices = self.inverted_index[token]

            for doc_idx in matching_doc_indices:
                if allowed_ids is not None and self.doc_ids[doc_idx] not in allowed_ids:
                    continue
                tf = self.doc_term_freqs[doc_idx][token]
                doc_len = self.doc_lengths[doc_idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                scores[doc_idx] += idf * (numerator / denominator)

        if not scores:
            return []

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(self.doc_ids[doc_idx], score) for doc_idx, score in sorted_results[:limit]]


_bm25_instance: Optional[BM25Index] = None


def get_bm25_index(rebuild: bool = False) -> BM25Index:
    """Singleton accessor for BM25Index with automatic build from MongoDB."""
    global _bm25_instance
    if _bm25_instance is None or rebuild:
        index = BM25Index()
        try:
            db = get_mongo_database()
            docs = list(db[CRICKET_EVENTS_COLLECTION].find({}, {"_id": 0}))
            index.build_from_documents(docs)
        except Exception as exc:
            logger.warning("Could not auto-populate BM25 index from MongoDB: %s", exc)
        _bm25_instance = index
    return _bm25_instance


def search_bm25(query: str, limit: int = 50, allowed_ids: Optional[set] = None) -> List[Tuple[str, float]]:
    """Convenience search function returning Top-K (event_id, bm25_score) tuples."""
    index = get_bm25_index()
    return index.search(query=query, limit=limit, allowed_ids=allowed_ids)
