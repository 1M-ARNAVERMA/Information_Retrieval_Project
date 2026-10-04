"""FEVER claim dataset loader and parser."""

from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Union

from src.common.exceptions import DataIngestionError
from src.common.types import Claim, EvidenceItem, EvidenceSet, VeracityLabel
from src.data_ingestion.normalizer import normalize_fever_text, normalize_wiki_title
from src.utils.file_utils import read_jsonl
from src.utils.logger import get_logger

logger = get_logger("fever_loader")


def parse_fever_record(raw: Dict[str, Union[int, str, List]]) -> Claim:
    """Parses a single raw FEVER JSON dictionary into a typed Claim model.

    Args:
        raw: Dict from FEVER jsonl file.

    Returns:
        Typed Claim instance.
    """
    claim_id = raw.get("id")
    claim_text = normalize_fever_text(str(raw.get("claim", "")))
    raw_label = raw.get("label")

    # Parse veracity label safely
    label = None
    if raw_label:
        try:
            label = VeracityLabel(str(raw_label).strip())
        except ValueError:
            logger.warning("Unknown label '%s' in claim ID %s", raw_label, claim_id)

    # Parse nested evidence sets
    evidence_sets: List[EvidenceSet] = []
    raw_evidence = raw.get("evidence", [])

    if isinstance(raw_evidence, list):
        for raw_set in raw_evidence:
            items: List[EvidenceItem] = []
            if isinstance(raw_set, list):
                for ev in raw_set:
                    # Format: [annotation_id, evidence_id, doc_id, sentence_id]
                    if isinstance(ev, list) and len(ev) >= 4:
                        doc_id = ev[2]
                        sentence_id = ev[3]
                        if doc_id is not None and sentence_id is not None:
                            try:
                                items.append(
                                    EvidenceItem(
                                        doc_id=normalize_wiki_title(str(doc_id)),
                                        sentence_id=int(sentence_id),
                                    )
                                )
                            except (ValueError, TypeError):
                                continue
            if items:
                evidence_sets.append(EvidenceSet(items=items))

    return Claim(
        id=claim_id,
        claim=claim_text,
        label=label,
        evidence=evidence_sets,
    )


def stream_claims(
    file_path: Union[str, Path],
    labels: Optional[Iterable[VeracityLabel]] = None,
    limit: Optional[int] = None,
) -> Generator[Claim, None, None]:
    """Streams and yields Claim objects from a FEVER JSONL file.

    Args:
        file_path: Path to the FEVER JSONL file.
        labels: Optional set of VeracityLabel to filter by.
        limit: Optional maximum number of claims to yield.

    Yields:
        Claim models.
    """
    path = Path(file_path)
    if not path.exists():
        raise DataIngestionError(f"FEVER claims file not found: {path}")

    accepted_labels = set(labels) if labels else None
    yielded = 0

    for raw_record in read_jsonl(path):
        claim = parse_fever_record(raw_record)
        if accepted_labels and claim.label not in accepted_labels:
            continue

        yield claim
        yielded += 1
        if limit is not None and yielded >= limit:
            break


def load_claims(
    file_path: Union[str, Path],
    labels: Optional[Iterable[VeracityLabel]] = None,
    limit: Optional[int] = None,
) -> List[Claim]:
    """Loads all matching claims from a FEVER JSONL file into a list.

    Args:
        file_path: Path to the FEVER JSONL file.
        labels: Optional set of VeracityLabel to filter by.
        limit: Optional maximum number of claims to load.

    Returns:
        List of Claim models.
    """
    return list(stream_claims(file_path=file_path, labels=labels, limit=limit))
