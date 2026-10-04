"""Query expansion module supporting rule-based lexical variations and optional LLM expansion."""

import os
import re
from typing import Dict, List, Optional, Set

from src.claim_analysis.ner import extract_entities
from src.common.types import Entity
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger("query_expansion")

# Synonyms and lexical substitutions for common verification predicates
RELATION_SYNONYMS: Dict[str, List[str]] = {
    "developed": ["formulated", "created", "discovered", "invented"],
    "founded": ["established", "started", "co-founded"],
    "won": ["received", "was awarded", "awarded"],
    "born": ["native of", "birthplace"],
    "directed": ["directed by", "film director"],
    "created": ["developed", "invented", "authored"],
    "wrote": ["authored", "published"],
    "married": ["spouse of", "husband of", "wife of"],
}


class QueryExpander:
    """Generates alternative search queries for improving candidate retrieval recall."""

    def __init__(
        self,
        use_llm: Optional[bool] = None,
        max_expansions: int = 4,
    ) -> None:
        """Initializes QueryExpander.

        Args:
            use_llm: Whether to invoke LLM expansion (defaults to config.yaml).
            max_expansions: Maximum total queries to output (including original).
        """
        if use_llm is None:
            try:
                cfg = load_config()
                self.use_llm = cfg.get("features", {}).get("use_llm_expansion", False)
            except Exception:
                self.use_llm = False
        else:
            self.use_llm = use_llm

        self.max_expansions = max_expansions

    def expand_rules(self, query: str, entities: Optional[List[Entity]] = None) -> List[str]:
        """Generates query variations using entity extraction and predicate substitution."""
        queries: List[str] = [query.strip()]
        ents = entities if entities is not None else extract_entities(query)

        # 1. Entity-focused query: join names of all extracted entities
        if ents:
            ent_query = " ".join(e.text for e in ents)
            if ent_query.lower() != query.lower():
                queries.append(ent_query)

        # 2. Predicate synonym replacement
        lower_q = query.lower()
        for predicate, synonyms in RELATION_SYNONYMS.items():
            if re.search(rf"\b{predicate}\b", lower_q):
                for syn in synonyms[:1]:  # Pick top synonym
                    replaced = re.sub(rf"\b{predicate}\b", syn, query, flags=re.IGNORECASE)
                    if replaced.lower() != query.lower():
                        queries.append(replaced)
                break

        # 3. Entity + Action keywords (stripping functional stopwords)
        tokens = re.findall(r"\b[A-Za-z0-9]+\b", query)
        stopwords = {"the", "a", "an", "is", "was", "are", "were", "in", "of", "to", "at", "by", "for", "with", "that"}
        key_tokens = [t for t in tokens if t.lower() not in stopwords]
        if len(key_tokens) >= 2:
            kw_query = " ".join(key_tokens)
            if kw_query.lower() != query.lower():
                queries.append(kw_query)

        # Deduplicate while preserving order
        unique_queries: List[str] = []
        seen = set()
        for q in queries:
            normalized = q.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_queries.append(q.strip())

        return unique_queries[: self.max_expansions]

    def expand_llm(self, query: str) -> List[str]:
        """Optional LLM-based query generator with safe fallback."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.debug("No OPENAI_API_KEY detected; defaulting to rule-based expansion.")
            return self.expand_rules(query)

        try:
            # Optional lightweight OpenAI client call
            import urllib.request
            import json

            prompt = (
                f"Generate 3 distinct Wikipedia search queries to verify this claim: '{query}'. "
                "Output queries separated by newlines only."
            )
            data = json.dumps({
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 60,
                "temperature": 0.2,
            }).encode("utf-8")

            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=data,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                text_out = result["choices"][0]["message"]["content"]
                llm_queries = [line.strip().lstrip("1234567890.- ") for line in text_out.split("\n") if line.strip()]
                return [query] + [q for q in llm_queries if q.lower() != query.lower()][: self.max_expansions - 1]
        except Exception as e:
            logger.warning("LLM query expansion failed (%s); falling back to rule-based.", e)
            return self.expand_rules(query)

    def expand(self, query: str, entities: Optional[List[Entity]] = None) -> List[str]:
        """Generates alternative search queries.

        Args:
            query: Claim or search query.
            entities: Optional pre-extracted Entity models.

        Returns:
            List of unique query strings, with original query at index 0.
        """
        if not query or not query.strip():
            return []

        if self.use_llm:
            return self.expand_llm(query)
        return self.expand_rules(query, entities=entities)
