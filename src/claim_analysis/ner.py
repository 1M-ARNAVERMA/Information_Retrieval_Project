"""Named Entity Recognition module using spaCy with robust rule-based fallback."""

import re
from typing import List, Optional, Set

from src.common.types import Entity
from src.utils.logger import get_logger

logger = get_logger("entity_extractor")

# Date pattern: e.g., '14 March 1879', '1921', 'March 1879', '1800s'
DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}"
    r"|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}"
    r"|\b(?:1[7-9]\d{2}|20\d{2})s?\b)\b",
    re.IGNORECASE,
)

# Common GPE / Location keywords
KNOWN_GPES: Set[str] = {
    "poland", "france", "germany", "german empire", "united states", "usa", "uk",
    "california", "hawthorne", "mars", "earth", "london", "paris", "berlin",
}

# Common Organizations
KNOWN_ORGS: Set[str] = {
    "spacex", "nasa", "fox broadcasting company", "fox", "tesla", "google",
    "nobel committee", "united nations", "mit", "harvard",
}


class EntityExtractor:
    """Extracts named entities (PERSON, ORG, GPE, DATE, etc.) using spaCy or rules."""

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self.model_name = model_name
        self._nlp = None
        self._load_spacy()

    def _load_spacy(self) -> None:
        """Attempts to load spaCy pipeline."""
        try:
            import spacy
            try:
                self._nlp = spacy.load(self.model_name)
                logger.info("Loaded spaCy model '%s'.", self.model_name)
            except OSError:
                logger.warning(
                    "spaCy model '%s' not installed. Using rule-based fallback extractor.",
                    self.model_name,
                )
                self._nlp = None
        except ImportError:
            logger.warning("spaCy not installed. Using rule-based fallback extractor.")
            self._nlp = None

    def extract_entities(self, text: str) -> List[Entity]:
        """Extracts named entities from claim or evidence text.

        Args:
            text: Input sentence or claim.

        Returns:
            List of Entity instances with text, label, start_char, and end_char.
        """
        if not text or not text.strip():
            return []

        # If spaCy is loaded, use it
        if self._nlp is not None:
            doc = self._nlp(text)
            entities: List[Entity] = []
            for ent in doc.ents:
                entities.append(
                    Entity(
                        text=ent.text.strip(),
                        label=ent.label_,
                        start_char=ent.start_char,
                        end_char=ent.end_char,
                    )
                )
            if entities:
                return entities

        # Rule-based fallback extractor
        return self._extract_fallback(text)

    def _extract_fallback(self, text: str) -> List[Entity]:
        """Deterministic heuristic NER extractor for offline environments."""
        entities: List[Entity] = []
        seen_spans = set()

        # 1. Dates
        for match in DATE_PATTERN.finditer(text):
            span = (match.start(), match.end())
            if span not in seen_spans:
                entities.append(
                    Entity(
                        text=match.group(0),
                        label="DATE",
                        start_char=match.start(),
                        end_char=match.end(),
                    )
                )
                seen_spans.add(span)

        # 2. Known ORGs
        lower_text = text.lower()
        for org in KNOWN_ORGS:
            start = lower_text.find(org)
            while start != -1:
                end = start + len(org)
                span = (start, end)
                if not any(s <= start and end <= e for s, e in seen_spans):
                    entities.append(
                        Entity(
                            text=text[start:end],
                            label="ORG",
                            start_char=start,
                            end_char=end,
                        )
                    )
                    seen_spans.add(span)
                start = lower_text.find(org, end)

        # 3. Known GPEs
        for gpe in KNOWN_GPES:
            start = lower_text.find(gpe)
            while start != -1:
                end = start + len(gpe)
                span = (start, end)
                if not any(s <= start and end <= e for s, e in seen_spans):
                    entities.append(
                        Entity(
                            text=text[start:end],
                            label="GPE",
                            start_char=start,
                            end_char=end,
                        )
                    )
                    seen_spans.add(span)
                start = lower_text.find(gpe, end)

        # 4. Capitalized entity sequences (PERSON / PROPER NOUNS)
        # Matches e.g., 'Albert Einstein', 'Marie Curie', 'SpaceX'
        cap_pattern = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
        for match in cap_pattern.finditer(text):
            matched_text = match.group(0)
            start, end = match.start(), match.end()

            # Skip common initial sentence words like 'The', 'He', 'She', 'It'
            if matched_text in ("The", "He", "She", "It", "They", "We", "This", "That", "There"):
                continue

            if not any(s <= start and end <= e for s, e in seen_spans):
                # Infer label: default to PERSON if 2+ capitalized words, else WORK_OF_ART or PERSON
                label = "PERSON" if " " in matched_text else "PERSON"
                entities.append(
                    Entity(
                        text=matched_text,
                        label=label,
                        start_char=start,
                        end_char=end,
                    )
                )
                seen_spans.add((start, end))

        # Sort entities by start_char
        return sorted(entities, key=lambda e: e.start_char)


# Singleton extractor instance
_default_extractor: Optional[EntityExtractor] = None


def get_entity_extractor() -> EntityExtractor:
    """Retrieves or creates the global default EntityExtractor."""
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = EntityExtractor()
    return _default_extractor


def extract_entities(text: str) -> List[Entity]:
    """Convenience helper to extract named entities from text."""
    return get_entity_extractor().extract_entities(text)
