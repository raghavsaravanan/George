"""Live Phase-2 webhook for Project George.

FastAPI endpoint that accepts Vapi custom tool-call webhooks, embeds the
technician query locally with FastEmbed, applies a strict year/make/model
Qdrant metadata filter against the offline-seeded george_specs collection,
and returns a single-line 1-2 sentence spoken result for headset TTS.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ingest import (
    VehicleMetadata,
    build_points,
    collect_spec_chunks,
    delete_points_for_document,
    document_ref_from_path,
    ensure_collection,
    expand_mechanical_units,
    load_api_key,
    parse_pdf_to_markdown,
)

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger("george.main")

QDRANT_PATH = PROJECT_ROOT / "george_mvp_db"
COLLECTION_NAME = "george_specs"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_SHOP_ID = "shop_demo"
# Cosine similarity floor for shop-scoped semantic fallback (Pass 3).
# Tuned so in-corpus AMS/Chevy hits (~0.71–0.75) pass and unrelated
# vehicles (Ford/Honda ~0.63–0.66) refuse → SPEC_NOT_FOUND.
MIN_SEMANTIC_SCORE = 0.68
# Soft vehicle match (Pass 2) may score a bit lower; still refuse junk.
MIN_FILTERED_SCORE = 0.60

# In-memory vehicle context keyed by Vapi callId (single-process demo/prod hop).
_CALL_SESSIONS: dict[str, "CallSession"] = {}
# shop_id → makes/models/years discovered from uploaded Qdrant payloads.
_SHOP_CATALOGS: dict[str, "ShopVehicleCatalog"] = {}

_DIGIT_GAP_RE = re.compile(r"(?<=\d)[.\s]+(?=\d)")
# Dotted part-number groups like "226.020" / "12.13" (not single-digit decimals).
_PART_NUMBER_DOT_RE = re.compile(r"(?<=\d{2})\.(?=\d{2})")
# Contiguous digit runs for TTS digit-spelling (years excluded in helper).
_PART_NUMBER_RUN_RE = re.compile(r"\d{4,}")
_YEAR_LIKE_RE = re.compile(r"^(?:19|20|21)\d{2}$")
_VEHICLE_TYPO_RULES: Sequence[tuple[re.Pattern[str], str]] = (
    (re.compile(r"\bbr[\s\-]?30\b", re.IGNORECASE), "vr30"),
    (re.compile(r"\bdr[\s\-]?30\b", re.IGNORECASE), "vr30"),
    (re.compile(r"\bv[\s\.\-]*r[\s\.\-]*30\b", re.IGNORECASE), "vr30"),
    (re.compile(r"\bfuel[\s\-]?pro\b", re.IGNORECASE), "fel-pro"),
    (re.compile(r"\bfell[\s\-]?pro\b", re.IGNORECASE), "fel-pro"),
    (re.compile(r"\bfelpro\b", re.IGNORECASE), "fel-pro"),
    (re.compile(r"\bchevy\b", re.IGNORECASE), "chevrolet"),
    (re.compile(r"\bsilverad[ao]\b", re.IGNORECASE), "silverado"),
)

_CAUTION_QUERY_HINTS = (
    "warning",
    "caution",
    "safety",
    "tips",
    "tip",
)
_SEQUENCE_QUERY_HINTS = (
    "sequence",
    "pattern",
    "cross pattern",
    "cross-pattern",
    "tightening order",
    "torque sequence",
    "bolt order",
)

_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?(?:\s*:?-+:?\s*\|)+\s*:?-+:?\s*\|?\s*$"
)
_HTML_ENTITY_RE = re.compile(r"&#x[0-9a-fA-F]+;|&[a-zA-Z]+;")
_MARKDOWN_EMPHASIS_RE = re.compile(r"\*+")
_MULTI_SPACE_RE = re.compile(r"\s+")
_UNIT_TOKEN_RE = re.compile(
    r"foot-pounds|inch-pounds|newton-meters",
    re.IGNORECASE,
)
_CAUTION_START_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*{0,2})?(?:WARNING|CAUTION|NOTE|IMPORTANT)"
    r"(?:\*{0,2})?\s*[:\-–]?\s*(.*)$",
    re.IGNORECASE,
)
_QUERY_TORQUE_HINTS = (
    "torque",
    "final",
    "initial",
    "first",
    "pass",
    "fastener",
    "bolt",
    "tighten",
    "spec",
)


@dataclass
class SpokenAnswer:
    tool_call_id: str
    ok: bool
    result: str | None = None
    error: str | None = None

    def to_vapi_result(self) -> dict[str, str]:
        payload: dict[str, str] = {"toolCallId": self.tool_call_id}
        if self.ok and self.result is not None:
            payload["result"] = to_single_line(self.result)
        else:
            payload["error"] = to_single_line(
                self.error or "Unable to resolve specification."
            )
        return payload


@dataclass
class CallSession:
    """Per-call vehicle memory for hands-free follow-up questions."""

    make: str = ""
    model: str = ""
    year: str = ""


@dataclass(frozen=True)
class ShopVehicleCatalog:
    """Vehicles present in a shop's uploaded Qdrant corpus (the living brain)."""

    makes: frozenset[str]
    models: frozenset[str]
    years: frozenset[str]
    # (year, make, model) triples as stored on points
    triples: frozenset[tuple[str, str, str]]

    @property
    def is_empty(self) -> bool:
        return not self.triples and not self.makes and not self.models


def get_shop_vehicle_catalog(
    client: QdrantClient,
    shop_id: str,
    *,
    force_refresh: bool = False,
) -> ShopVehicleCatalog:
    """Build (and cache) make/model/year sets from Qdrant payloads for one shop."""
    tenant = (shop_id or DEFAULT_SHOP_ID).strip().lower() or DEFAULT_SHOP_ID
    if not force_refresh and tenant in _SHOP_CATALOGS:
        return _SHOP_CATALOGS[tenant]

    makes: set[str] = set()
    models: set[str] = set()
    years: set[str] = set()
    triples: set[tuple[str, str, str]] = set()

    shop_filter = qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="shop_id",
                match=qmodels.MatchValue(value=tenant),
            )
        ]
    )
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=shop_filter,
            limit=128,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = getattr(point, "payload", None) or {}
            if not isinstance(payload, dict):
                continue
            year = str(payload.get("year") or "").strip().lower()
            make = str(payload.get("make") or "").strip().lower()
            model = str(payload.get("model") or "").strip().lower()
            if make:
                makes.add(make)
            if model:
                models.add(model)
            if year:
                years.add(year)
            if make or model or year:
                triples.add((year, make, model))
        if offset is None:
            break

    catalog = ShopVehicleCatalog(
        makes=frozenset(makes),
        models=frozenset(models),
        years=frozenset(years),
        triples=frozenset(triples),
    )
    _SHOP_CATALOGS[tenant] = catalog
    logger.info(
        "shop catalog ready shop_id=%r makes=%s models=%s years=%s triples=%d",
        tenant,
        sorted(makes),
        sorted(models),
        sorted(years),
        len(triples),
    )
    return catalog


def invalidate_shop_catalog(shop_id: str | None = None) -> None:
    """Drop cached catalog(s) after a re-ingest so the brain refreshes."""
    if shop_id:
        _SHOP_CATALOGS.pop(
            (shop_id or DEFAULT_SHOP_ID).strip().lower() or DEFAULT_SHOP_ID,
            None,
        )
    else:
        _SHOP_CATALOGS.clear()


def query_normalizer(query: str) -> str:
    """Repair STT digit gaps and common vehicle/brand mishears before search."""
    text = (query or "").strip()
    if not text:
        return ""

    # "20. 19" -> "2019", "226. 042" -> "226042"
    text = _DIGIT_GAP_RE.sub("", text)

    for pattern, replacement in _VEHICLE_TYPO_RULES:
        text = pattern.sub(replacement, text)

    return _MULTI_SPACE_RE.sub(" ", text).strip()


def _extract_call_id(payload: dict[str, Any]) -> str:
    for key in ("callId", "call_id"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    call = payload.get("call")
    if isinstance(call, dict):
        for key in ("id", "callId", "call_id"):
            value = call.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    message = payload.get("message")
    if isinstance(message, dict):
        nested_call = message.get("call")
        if isinstance(nested_call, dict):
            for key in ("id", "callId", "call_id"):
                value = nested_call.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        for key in ("callId", "call_id"):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return ""


def _extract_tool_call_id(payload: dict[str, Any]) -> str:
    """Prefer Vapi toolCall.id; never confuse with session callId."""
    for key in ("toolCallId", "tool_call_id"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    tool_calls = extract_tool_call_list(payload)
    if tool_calls:
        return extract_tool_call_id(tool_calls[0], 0)

    return "vapi-call"


def _get_or_create_session(call_id: str) -> CallSession | None:
    if not call_id:
        return None
    session = _CALL_SESSIONS.get(call_id)
    if session is None:
        session = CallSession()
        _CALL_SESSIONS[call_id] = session
    return session


def _apply_session_vehicle_fields(
    call_id: str,
    make: str,
    model: str,
    year: str,
) -> tuple[str, str, str]:
    """Fill missing make/model/year from the in-memory call session."""
    session = _get_or_create_session(call_id)
    if session is None:
        return make, model, year

    make_out = make or session.make
    model_out = model or session.model
    year_out = year or session.year
    return make_out, model_out, year_out


def _update_session_vehicle_fields(
    call_id: str,
    make: str,
    model: str,
    year: str,
) -> None:
    session = _get_or_create_session(call_id)
    if session is None:
        return
    if make:
        session.make = make
    if model:
        session.model = model
    if year:
        session.year = year


def _should_include_caution(query: str) -> bool:
    lowered = (query or "").lower()
    if any(hint in lowered for hint in _CAUTION_QUERY_HINTS):
        return True
    if any(hint in lowered for hint in _SEQUENCE_QUERY_HINTS):
        return True
    return False


def to_single_line(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\n", " ").replace("\t", " ")
    cleaned = _HTML_ENTITY_RE.sub(" ", cleaned)
    cleaned = _MARKDOWN_EMPHASIS_RE.sub("", cleaned)
    # Collapse dotted part-number sequences: "226.020" -> "226020", "12.13" -> "1213".
    # Requires 2+ digits on each side so values like "1.5" stay intact.
    cleaned = _PART_NUMBER_DOT_RE.sub("", cleaned)
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def format_number_for_speech(text: str) -> str:
    """Spell part-number digit runs individually for TTS clarity.

    Sequences of 4+ digits (except years like 2019) become spaced digits so
    "1213" is spoken as "1 2 1 3" rather than "one thousand two hundred thirteen".
    """

    def _replace_run(match: re.Match[str]) -> str:
        digits = match.group(0)
        if _YEAR_LIKE_RE.fullmatch(digits):
            return digits
        # Prefer explicit 4-5 digit part numbers; longer OEM application
        # codes (e.g. 226042) are also digit-spelled for the same reason.
        if len(digits) < 4:
            return digits
        return " ".join(digits)

    return _PART_NUMBER_RUN_RE.sub(_replace_run, text or "")


def _split_table_cells(row: str) -> list[str]:
    stripped = row.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _is_separator_row(row: str) -> bool:
    return bool(_TABLE_SEPARATOR_RE.match(row.strip()))


def _parse_table_and_caution(chunk_text: str) -> tuple[list[list[str]], str]:
    lines = chunk_text.splitlines()
    table_rows: list[list[str]] = []
    caution_parts: list[str] = []
    past_table = False

    for raw_line in lines:
        line = raw_line.rstrip()
        if not past_table and _TABLE_ROW_RE.match(line):
            if _is_separator_row(line):
                continue
            cells = _split_table_cells(line)
            if cells:
                table_rows.append(cells)
            continue

        if table_rows:
            past_table = True

        if not past_table:
            continue

        if not line.strip():
            continue

        caution_match = _CAUTION_START_RE.match(line)
        if caution_match is not None:
            remainder = caution_match.group(1).strip()
            if remainder:
                caution_parts.append(remainder)
            continue

        caution_parts.append(line.strip())

    caution_text = to_single_line(" ".join(caution_parts)) if caution_parts else ""
    return table_rows, caution_text


def _header_map(header_cells: Sequence[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for index, cell in enumerate(header_cells):
        key = cell.strip().lower()
        if key:
            mapping[key] = index
    return mapping


def _cell_at(row: Sequence[str], mapping: dict[str, int], *names: str) -> str:
    for name in names:
        index = mapping.get(name)
        if index is not None and index < len(row):
            return row[index].strip()
    return ""


def _row_has_unit(row_text: str) -> bool:
    return bool(_UNIT_TOKEN_RE.search(row_text))


def _query_hint_tokens(query: str) -> set[str]:
    lowered = query.lower()
    return {hint for hint in _QUERY_TORQUE_HINTS if hint in lowered}


def _query_content_tokens(query: str) -> set[str]:
    stop = {
        "a", "an", "the", "for", "and", "or", "of", "to", "on", "in", "what",
        "whats", "is", "are", "me", "my", "please", "spec", "specs",
    }
    tokens = set()
    for raw in query.lower().replace("/", " ").replace("-", " ").split():
        token = "".join(ch for ch in raw if ch.isalnum())
        if len(token) >= 3 and token not in stop:
            tokens.add(token)
    return tokens


def _score_row(
    application: str,
    notes: str,
    torque: str,
    hints: set[str],
    query_tokens: set[str] | None = None,
) -> int:
    haystack = f"{application} {notes} {torque}".lower()
    score = 0
    if _row_has_unit(haystack):
        score += 10
    for hint in hints:
        if hint in haystack:
            score += 5
    if query_tokens:
        for token in query_tokens:
            if token in haystack:
                score += 8
    if "final" in haystack:
        score += 3
    if "initial" in haystack or "first" in haystack:
        score += 2
    return score


def _compose_spec_sentence(application: str, torque: str, notes: str) -> str:
    application = to_single_line(application)
    torque = to_single_line(torque)
    notes = to_single_line(notes)

    if torque and application:
        app_lower = application.lower()
        if "final" in app_lower:
            sentence = f"The final torque is {torque}."
        elif "initial" in app_lower or "first" in app_lower:
            sentence = f"The initial torque is {torque}."
        else:
            sentence = f"{application}: {torque}."
    elif torque:
        sentence = f"The specification is {torque}."
    elif application and notes:
        sentence = f"{application}: {notes}."
    elif notes:
        sentence = notes if notes.endswith(".") else f"{notes}."
    elif application:
        sentence = application if application.endswith(".") else f"{application}."
    else:
        sentence = "The matching specification table was found but had no readable values."

    sentence = expand_mechanical_units(sentence)
    if notes and torque and notes.lower() not in sentence.lower():
        if "follow" in notes.lower() or "sequence" in notes.lower():
            note_clause = notes if notes.endswith(".") else f"{notes}."
            sentence = f"{sentence.rstrip('.')}."
            sentence = f"{sentence} {note_clause}"
    return to_single_line(sentence)


def _compose_warning_sentence(caution_text: str) -> str:
    caution = to_single_line(expand_mechanical_units(caution_text))
    caution = re.sub(
        r"^(?:warning|caution|note|important)\b\s*[:\-!]+\s*",
        "",
        caution,
        flags=re.I,
    )
    caution = caution.strip(" \t:-*!")
    # Drop truncated / empty cautions like "!" left over from OCR noise.
    if len(caution) < 12:
        return ""
    body = caution
    if not body.lower().startswith(("warning", "caution", "note", "important")):
        body = f"Caution: {body}"
    if not body.endswith("."):
        body = f"{body}."
    return to_single_line(body)


def format_voice_answer(chunk_text: str, query: str) -> str:
    """Convert a Markdown table chunk into a 1-2 sentence spoken string."""
    expanded_chunk = expand_mechanical_units(chunk_text)
    table_rows, caution_text = _parse_table_and_caution(expanded_chunk)

    if not table_rows:
        fallback = to_single_line(expanded_chunk)
        if not fallback:
            return "No readable specification text was available."
        sentences = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+", fallback)
            if part.strip()
        ]
        return format_number_for_speech(to_single_line(" ".join(sentences[:2])))

    header = table_rows[0]
    body_rows = table_rows[1:] if len(table_rows) > 1 else []
    mapping = _header_map(header)

    looks_like_header = any(
        key in mapping for key in ("application", "torque", "notes", "gasket", "cfm")
    )
    if not looks_like_header:
        body_rows = table_rows
        mapping = {}

    hints = _query_hint_tokens(query)
    query_tokens = _query_content_tokens(query)
    scored: list[tuple[int, str, str, str]] = []

    for row in body_rows:
        if mapping:
            application = _cell_at(row, mapping, "application", "item", "fastener")
            torque = _cell_at(row, mapping, "torque", "spec", "specification", "english units")
            notes = _cell_at(row, mapping, "notes", "note", "gasket", "cfm")
            if not torque and len(row) >= 2:
                torque = row[1].strip()
            if not application and row:
                application = row[0].strip()
            if not notes and len(row) >= 3:
                notes = row[-1].strip()
        else:
            application = row[0].strip() if row else ""
            torque = row[1].strip() if len(row) > 1 else ""
            notes = row[2].strip() if len(row) > 2 else ""

        joined = f"{application} {torque} {notes}"
        score = _score_row(
            application,
            notes,
            torque,
            hints,
            query_tokens=query_tokens,
        )
        if score <= 0 and not _row_has_unit(joined) and not hints:
            score = 1
        scored.append((score, application, torque, notes))

    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        primary = "The matching specification table was found but had no data rows."
    else:
        _, application, torque, notes = scored[0]
        primary = _compose_spec_sentence(application, torque, notes)

        # Prefer a final-pass line as second factual sentence when the top hit is initial.
        if "initial" in application.lower() or "first" in application.lower():
            for score, app2, torque2, notes2 in scored[1:]:
                if score <= 0:
                    break
                if "final" in app2.lower() and torque2:
                    second_fact = _compose_spec_sentence(app2, torque2, notes2)
                    primary = to_single_line(f"{primary} {second_fact}")
                    break

    warning = ""
    if _should_include_caution(query):
        warning = _compose_warning_sentence(caution_text)

    if warning and warning.lower() not in primary.lower():
        combined = to_single_line(f"{primary} {warning}")
    else:
        combined = to_single_line(primary)

    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", combined)
        if part.strip()
    ]
    clipped = " ".join(sentences[:2])
    spoken = to_single_line(expand_mechanical_units(clipped))
    return format_number_for_speech(spoken)


def _payload_text(hit: Any) -> str | None:
    payload = getattr(hit, "payload", None) or {}
    if not isinstance(payload, dict):
        return None
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    return text


def _hit_score(hit: Any) -> float | None:
    score = getattr(hit, "score", None)
    if score is None:
        return None
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def _vehicle_fields_in_catalog(
    catalog: ShopVehicleCatalog,
    year: str,
    make: str,
    model: str,
) -> bool:
    """False when requested make/model/year is not present in uploaded shop data."""
    if catalog.is_empty:
        return False
    if make and make not in catalog.makes:
        return False
    if model and model not in catalog.models:
        return False
    if year and year not in catalog.years:
        return False
    if make and model:
        if not any(t[1] == make and t[2] == model for t in catalog.triples):
            return False
    return True


def _payload_vehicle_matches(
    hit: Any,
    year: str,
    make: str,
    model: str,
) -> bool:
    """Reject hits whose payload vehicle tags contradict explicit request fields."""
    payload = getattr(hit, "payload", None) or {}
    if not isinstance(payload, dict):
        return False
    if make:
        hit_make = str(payload.get("make") or "").strip().lower()
        if hit_make and hit_make != make:
            return False
    if model:
        hit_model = str(payload.get("model") or "").strip().lower()
        if hit_model and hit_model != model:
            return False
    if year:
        hit_year = str(payload.get("year") or "").strip().lower()
        if hit_year and hit_year != year:
            return False
    return True


def _accept_hit(
    hit: Any,
    *,
    min_score: float | None,
    year: str,
    make: str,
    model: str,
    pass_name: str,
) -> str | None:
    text = _payload_text(hit)
    if not text:
        return None
    if not _payload_vehicle_matches(hit, year, make, model):
        logger.info(
            "reject %s: vehicle payload mismatch year=%r make=%r model=%r",
            pass_name,
            year or None,
            make or None,
            model or None,
        )
        return None
    if min_score is not None:
        score = _hit_score(hit)
        if score is None or score < min_score:
            logger.info(
                "reject %s: score=%s below min=%.3f",
                pass_name,
                score,
                min_score,
            )
            return None
        logger.info("accept %s: score=%.4f", pass_name, score)
    return text


def _query_vector_for(
    embedder: TextEmbedding,
    query: str,
    year: str,
    make: str,
    model: str,
) -> list[float] | None:
    # Fold vehicle context into the embedding so semantic fallback still
    # prefers chunks that mention related applications / torque language.
    vehicle_bits = [part for part in (year, make, model) if part]
    embed_text = query
    if vehicle_bits:
        embed_text = f"{query} {' '.join(vehicle_bits)}".strip()

    vectors = list(embedder.embed([embed_text]))
    if not vectors:
        return None
    return list(vectors[0])


def _search_points(
    client: QdrantClient,
    query_vector: list[float],
    query_filter: qmodels.Filter | None,
    limit: int = 1,
) -> list[Any]:
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )
    if response is None or not response.points:
        return []
    return list(response.points)


def _shop_id_condition(shop_id: str) -> qmodels.FieldCondition:
    tenant = (shop_id or DEFAULT_SHOP_ID).strip().lower() or DEFAULT_SHOP_ID
    return qmodels.FieldCondition(
        key="shop_id",
        match=qmodels.MatchValue(value=tenant),
    )


def _search_top_chunk_sync(
    client: QdrantClient,
    embedder: TextEmbedding,
    query: str,
    year: str,
    make: str,
    model: str,
    shop_id: str = DEFAULT_SHOP_ID,
) -> str | None:
    """Hybrid retrieval: exact metadata first, then soft match, then semantic.

    ``shop_id`` is a mandatory must-filter on every pass so tenants cannot
    retrieve each other's manuals. Pass 3 (and soft Pass 2) refuse low-score
    nearest neighbors so unknown vehicles do not inherit AMS/Chevy leftovers.
    Sync — call via ``search_top_chunk`` so FastAPI never blocks the event loop.
    """
    tenant = (shop_id or DEFAULT_SHOP_ID).strip().lower() or DEFAULT_SHOP_ID
    shop_must = [_shop_id_condition(tenant)]
    catalog = get_shop_vehicle_catalog(client, tenant)

    # Empty shop brain (post-purge / no uploads) → hard miss, no semantic gambling.
    if catalog.is_empty:
        logger.info("refuse search: empty shop catalog shop_id=%r", tenant)
        return None

    # Explicit vehicle fields must exist in this shop's uploaded brain.
    if (make or model or year) and not _vehicle_fields_in_catalog(
        catalog, year, make, model
    ):
        logger.info(
            "refuse search: fields not in shop catalog shop_id=%r "
            "year=%r make=%r model=%r",
            tenant,
            year or None,
            make or None,
            model or None,
        )
        return None

    query_vector = _query_vector_for(embedder, query, year, make, model)
    if query_vector is None:
        return None

    # Pass 1 — strict AND filter (character-for-character metadata match).
    if year and make and model:
        strict_filter = qmodels.Filter(
            must=[
                *shop_must,
                qmodels.FieldCondition(
                    key="year",
                    match=qmodels.MatchValue(value=year),
                ),
                qmodels.FieldCondition(
                    key="make",
                    match=qmodels.MatchValue(value=make),
                ),
                qmodels.FieldCondition(
                    key="model",
                    match=qmodels.MatchValue(value=model),
                ),
            ]
        )
        hits = _search_points(client, query_vector, strict_filter, limit=1)
        if hits:
            # Strict metadata already scoped the vehicle; score floor is not required.
            text = _accept_hit(
                hits[0],
                min_score=None,
                year=year,
                make=make,
                model=model,
                pass_name="strict",
            )
            if text:
                return text

    # Pass 2 — soft OR on vehicle fields, still locked to shop_id.
    soft_should: list[qmodels.FieldCondition] = []
    if year:
        soft_should.append(
            qmodels.FieldCondition(
                key="year",
                match=qmodels.MatchValue(value=year),
            )
        )
    if make:
        soft_should.append(
            qmodels.FieldCondition(
                key="make",
                match=qmodels.MatchValue(value=make),
            )
        )
    if model:
        soft_should.append(
            qmodels.FieldCondition(
                key="model",
                match=qmodels.MatchValue(value=model),
            )
        )
    if soft_should:
        soft_filter = qmodels.Filter(
            must=shop_must,
            min_should=qmodels.MinShould(
                conditions=soft_should,
                min_count=1,
            ),
        )
        hits = _search_points(client, query_vector, soft_filter, limit=1)
        if hits:
            text = _accept_hit(
                hits[0],
                min_score=MIN_FILTERED_SCORE,
                year=year,
                make=make,
                model=model,
                pass_name="soft",
            )
            if text:
                return text

    # Pass 3 — semantic fallback still restricted to this shop's corpus.
    tenant_filter = qmodels.Filter(must=shop_must)
    hits = _search_points(client, query_vector, query_filter=tenant_filter, limit=1)
    if not hits:
        return None
    return _accept_hit(
        hits[0],
        min_score=MIN_SEMANTIC_SCORE,
        year=year,
        make=make,
        model=model,
        pass_name="semantic",
    )


async def search_top_chunk(
    client: QdrantClient,
    embedder: TextEmbedding,
    query: str,
    year: str,
    make: str,
    model: str,
    shop_id: str = DEFAULT_SHOP_ID,
) -> str | None:
    """Run hybrid retrieval off the event loop (embed + Qdrant are sync)."""
    return await asyncio.to_thread(
        _search_top_chunk_sync,
        client,
        embedder,
        query,
        year,
        make,
        model,
        shop_id,
    )


def _coerce_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        parts = [_coerce_string(item) for item in value]
        return " ".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        return ""
    return str(value).strip()


def extract_tool_arguments(tool_call: dict[str, Any]) -> dict[str, str]:
    """Pull query/make/model/year/shop_id from a Vapi toolCallList entry safely."""
    raw_args: Any = tool_call.get("parameters")
    if raw_args is None:
        raw_args = tool_call.get("arguments")

    if isinstance(raw_args, str):
        import json

        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            parsed = {}
        raw_args = parsed

    if not isinstance(raw_args, dict):
        function_block = tool_call.get("function")
        if isinstance(function_block, dict):
            nested = function_block.get("arguments") or function_block.get("parameters")
            if isinstance(nested, str):
                import json

                try:
                    nested = json.loads(nested)
                except json.JSONDecodeError:
                    nested = {}
            raw_args = nested if isinstance(nested, dict) else {}
        else:
            raw_args = {}

    return {
        "query": _coerce_string(raw_args.get("query")),
        "make": _coerce_string(raw_args.get("make")).lower(),
        "model": _coerce_string(raw_args.get("model")).lower(),
        "year": _coerce_string(raw_args.get("year")).lower(),
        "shop_id": _coerce_string(raw_args.get("shop_id")).lower(),
    }


def normalize_vapi_lookup_payload(payload: dict[str, Any]) -> dict[str, str]:
    """Accept flat apiRequest bodies or nested message.toolCallList webhooks."""
    query = str(payload.get("query") or "").strip()
    make = str(payload.get("make") or "").strip().lower()
    model = str(payload.get("model") or "").strip().lower()
    year = str(payload.get("year") or "").strip().lower()
    shop_id = str(payload.get("shop_id") or "").strip().lower()
    tool_call_id = _extract_tool_call_id(payload)

    if not query:
        tool_calls = extract_tool_call_list(payload)
        if tool_calls:
            first = tool_calls[0]
            tool_call_id = extract_tool_call_id(first, 0)
            args = extract_tool_arguments(first)
            query = args.get("query") or ""
            make = make or args.get("make") or ""
            model = model or args.get("model") or ""
            year = year or args.get("year") or ""
            shop_id = shop_id or args.get("shop_id") or ""

    # Some apiRequest configs nest fields under "parameters".
    if not query:
        nested = payload.get("parameters")
        if isinstance(nested, str):
            import json

            try:
                nested = json.loads(nested)
            except json.JSONDecodeError:
                nested = {}
        if isinstance(nested, dict):
            query = str(nested.get("query") or "").strip()
            make = make or str(nested.get("make") or "").strip().lower()
            model = model or str(nested.get("model") or "").strip().lower()
            year = year or str(nested.get("year") or "").strip().lower()
            shop_id = shop_id or str(nested.get("shop_id") or "").strip().lower()

    return {
        "query": query,
        "make": make,
        "model": model,
        "year": year,
        "shop_id": shop_id or DEFAULT_SHOP_ID,
        "tool_call_id": tool_call_id or "vapi-call",
    }


def extract_tool_call_id(tool_call: dict[str, Any], index: int) -> str:
    tool_call_id = tool_call.get("id") or tool_call.get("toolCallId")
    if isinstance(tool_call_id, str) and tool_call_id.strip():
        return tool_call_id.strip()
    return f"tool-call-{index}"


def extract_tool_call_list(payload: dict[str, Any]) -> list[dict[str, Any]]:
    message = payload.get("message")
    if not isinstance(message, dict):
        return []

    tool_call_list = message.get("toolCallList")
    if isinstance(tool_call_list, list):
        return [item for item in tool_call_list if isinstance(item, dict)]

    tool_with_list = message.get("toolWithToolCallList")
    if isinstance(tool_with_list, list):
        recovered: list[dict[str, Any]] = []
        for item in tool_with_list:
            if not isinstance(item, dict):
                continue
            nested = item.get("toolCall")
            if isinstance(nested, dict):
                merged = dict(nested)
                if "name" not in merged and "name" in item:
                    merged["name"] = item["name"]
                if "parameters" not in merged and "parameters" in item:
                    merged["parameters"] = item["parameters"]
                recovered.append(merged)
            else:
                recovered.append(item)
        return recovered

    return []


def handle_tool_call(
    tool_call: dict[str, Any],
    index: int,
    client: QdrantClient,
    embedder: TextEmbedding,
) -> SpokenAnswer:
    tool_call_id = extract_tool_call_id(tool_call, index)
    args = extract_tool_arguments(tool_call)

    missing = [
        name
        for name in ("query", "make", "model", "year")
        if not args.get(name)
    ]
    if missing:
        return SpokenAnswer(
            tool_call_id=tool_call_id,
            ok=False,
            error=(
                "Missing required tool parameters: "
                + ", ".join(missing)
                + "."
            ),
        )

    try:
        chunk_text = _search_top_chunk_sync(
            client=client,
            embedder=embedder,
            query=args["query"],
            year=args["year"],
            make=args["make"],
            model=args["model"],
            shop_id=DEFAULT_SHOP_ID,
        )
    except Exception as exc:  # noqa: BLE001 - surface as Vapi tool error string
        return SpokenAnswer(
            tool_call_id=tool_call_id,
            ok=False,
            error=f"Specification lookup failed: {exc}.",
        )

    if chunk_text is None:
        return SpokenAnswer(
            tool_call_id=tool_call_id,
            ok=False,
            error="No specification found for that year make and model.",
        )

    spoken = format_voice_answer(chunk_text, args["query"])
    if not spoken:
        return SpokenAnswer(
            tool_call_id=tool_call_id,
            ok=False,
            error="Specification found but could not be formatted for speech.",
        )

    return SpokenAnswer(tool_call_id=tool_call_id, ok=True, result=spoken)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
    logger.setLevel(logging.INFO)
    logger.info("George webhook starting up")

    # On Render, cloud Qdrant is mandatory — never fall back to a missing local path.
    on_render = bool((os.getenv("RENDER") or "").strip())
    qdrant_url = (os.getenv("QDRANT_URL") or "").strip()
    qdrant_api_key = (os.getenv("QDRANT_API_KEY") or "").strip()

    if on_render and not (qdrant_url and qdrant_api_key):
        raise RuntimeError(
            "Render deploy requires QDRANT_URL and QDRANT_API_KEY environment variables."
        )

    embedder = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)

    if qdrant_url and qdrant_api_key:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=10.0)
        collection_location = qdrant_url
    else:
        if not QDRANT_PATH.exists():
            raise RuntimeError(
                f"Qdrant path does not exist: {QDRANT_PATH}. Run ingest.py first."
            )
        client = QdrantClient(path=str(QDRANT_PATH), timeout=10.0)
        collection_location = str(QDRANT_PATH)

    if not client.collection_exists(COLLECTION_NAME):
        client.close()
        raise RuntimeError(
            f"Qdrant collection '{COLLECTION_NAME}' was not found at "
            f"{collection_location}."
        )

    logger.info("Qdrant ready at %s collection=%s", collection_location, COLLECTION_NAME)
    invalidate_shop_catalog()
    app.state.embedder = embedder
    app.state.qdrant = client
    try:
        yield
    finally:
        client.close()
        invalidate_shop_catalog()
        logger.info("George webhook shut down")


app = FastAPI(
    title="George Vapi Tool Webhook",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness endpoint for Render / load balancers."""
    return {"status": "ok", "service": "george"}


async def _ingest_upload_job(
    temp_path: Path,
    year: str,
    make: str,
    model: str,
    shop_id: str,
    embedder: TextEmbedding,
    client: QdrantClient,
) -> None:
    """Background: LlamaParse → chunks → embed → Qdrant upsert, then delete temp."""
    tenant = (shop_id or DEFAULT_SHOP_ID).strip().lower() or DEFAULT_SHOP_ID
    try:
        api_key = load_api_key()
        logger.info("upload job start path=%s shop_id=%s", temp_path.name, tenant)
        markdown = await parse_pdf_to_markdown(
            temp_path,
            api_key,
            force_parse=True,
            premium_mode=False,
        )
        markdown = expand_mechanical_units(markdown)
        chunks = collect_spec_chunks(markdown)
        if not chunks:
            logger.error(
                "upload job produced no chunks for %s — nothing upserted",
                temp_path.name,
            )
            return

        metadata = VehicleMetadata(
            year=year,
            make=make,
            model=model,
            document_ref=document_ref_from_path(temp_path),
            shop_id=tenant,
        )

        def _sync_upsert() -> int:
            ensure_collection(client)
            delete_points_for_document(
                client,
                document_ref=metadata.document_ref,
                shop_id=tenant,
            )
            points = build_points(chunks, metadata, embedder)
            if points:
                client.upsert(collection_name=COLLECTION_NAME, points=list(points))
            invalidate_shop_catalog(tenant)
            return len(points)

        count = await asyncio.to_thread(_sync_upsert)
        logger.info(
            "upload job done document_ref=%s shop_id=%s points=%d",
            metadata.document_ref,
            tenant,
            count,
        )
    except SystemExit as exc:
        logger.error(
            "upload job aborted path=%s shop_id=%s err=%s",
            temp_path,
            tenant,
            exc,
        )
    except Exception:
        logger.exception("upload job failed path=%s shop_id=%s", temp_path, tenant)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
            parent = temp_path.parent
            if parent.name.startswith("george_upload_"):
                parent.rmdir()
        except OSError:
            logger.warning("could not delete temp upload %s", temp_path)


@app.post("/upload")
async def upload_manual(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
    year: str = Form(...),
    make: str = Form(...),
    model: str = Form(...),
    shop_id: str = Form(DEFAULT_SHOP_ID),
) -> JSONResponse:
    """Accept a PDF and ingest it into Qdrant in the background."""
    if not year.strip() or not make.strip() or not model.strip():
        return JSONResponse(
            status_code=400,
            content={"detail": "year, make, and model are required."},
        )

    filename = (file.filename or "upload.pdf").strip()
    if not filename.lower().endswith(".pdf"):
        return JSONResponse(
            status_code=400,
            content={"detail": "Only PDF uploads are supported."},
        )

    safe_name = Path(filename).name
    tmp_dir = Path(tempfile.mkdtemp(prefix="george_upload_"))
    temp_path = tmp_dir / safe_name
    try:
        with temp_path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    finally:
        await file.close()

    client: QdrantClient = request.app.state.qdrant
    embedder: TextEmbedding = request.app.state.embedder
    background_tasks.add_task(
        _ingest_upload_job,
        temp_path,
        year.strip(),
        make.strip(),
        model.strip(),
        (shop_id or DEFAULT_SHOP_ID).strip() or DEFAULT_SHOP_ID,
        embedder,
        client,
    )
    return JSONResponse(
        status_code=202,
        content={
            "status": "accepted",
            "message": "Manual queued for ingest.",
            "filename": safe_name,
            "shop_id": (shop_id or DEFAULT_SHOP_ID).strip().lower() or DEFAULT_SHOP_ID,
            "year": year.strip().lower(),
            "make": make.strip().lower(),
            "model": model.strip().lower(),
        },
    )


def _infer_vehicle_fields(
    query: str,
    make: str,
    model: str,
    year: str,
    catalog: ShopVehicleCatalog | None = None,
) -> tuple[str, str, str]:
    """Fill missing make/model/year only from tags present in the shop catalog."""
    q = (query or "").lower()
    make_out, model_out, year_out = make, model, year
    if catalog is None or catalog.is_empty:
        return make_out, model_out, year_out

    if not model_out:
        for candidate in sorted(catalog.models, key=len, reverse=True):
            if candidate and candidate in q:
                model_out = candidate
                break

    if not make_out:
        for candidate in sorted(catalog.makes, key=len, reverse=True):
            if candidate and candidate in q:
                make_out = candidate
                break

    # Spelling aliases — only if the canonical tag exists in this shop's brain.
    if not make_out and "chevy" in q and "chevrolet" in catalog.makes:
        make_out = "chevrolet"
    if not model_out and "ams" in q and "vr30" in catalog.models:
        model_out = "vr30"
    if not make_out and model_out == "vr30" and "nissan" in catalog.makes:
        make_out = "nissan"

    if not year_out:
        for candidate in sorted(catalog.years, key=len, reverse=True):
            if candidate and candidate in q:
                year_out = candidate
                break

    # If model is known, fill make/year from the only matching uploaded triple(s).
    if model_out:
        matches = [t for t in catalog.triples if t[2] == model_out]
        if matches:
            if not make_out:
                makes = {t[1] for t in matches if t[1]}
                if len(makes) == 1:
                    make_out = next(iter(makes))
            if not year_out:
                years = {t[0] for t in matches if t[0]}
                if len(years) == 1:
                    year_out = next(iter(years))

    if make_out and not model_out:
        matches = [t for t in catalog.triples if t[1] == make_out]
        if len(matches) == 1:
            y, _m, mo = matches[0]
            if mo and not model_out:
                model_out = mo
            if y and not year_out:
                year_out = y

    return make_out, model_out, year_out


@app.post("/vapi-tool")
async def vapi_tool(request: Request) -> JSONResponse:
    """Accept flat apiRequest body: {query, make, model, year, callId?}.

    Only ``query`` is required. Missing vehicle fields are filled from the
    in-memory call session when ``callId`` is present, then inferred from the
    query when possible so the voice agent does not need to interrogate the
    tech for year/make/model on every turn.
    """
    started = time.perf_counter()
    outcome = "error"

    webhook_secret = (os.getenv("VAPI_WEBHOOK_SECRET") or "").strip()
    if webhook_secret:
        incoming_secret = request.headers.get("x-vapi-secret")
        if incoming_secret != webhook_secret:
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized"},
            )

    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001 - still return Vapi-shaped 200
        return JSONResponse(
            status_code=200,
            content={
                "results": [
                    {
                        "toolCallId": "vapi-call",
                        "error": "Request body must be valid JSON.",
                    }
                ]
            },
        )

    if not isinstance(payload, dict):
        return JSONResponse(
            status_code=200,
            content={
                "results": [
                    {
                        "toolCallId": "vapi-call",
                        "error": "Request JSON must be an object.",
                    }
                ]
            },
        )

    fields = normalize_vapi_lookup_payload(payload)
    query = fields["query"]
    make = fields["make"]
    model = fields["model"]
    year = fields["year"]
    shop_id = fields["shop_id"] or DEFAULT_SHOP_ID
    call_id = _extract_call_id(payload)
    tool_call_id = fields["tool_call_id"]

    if not query:
        return JSONResponse(
            status_code=200,
            content={
                "results": [
                    {
                        "toolCallId": tool_call_id,
                        "error": "Missing required field: query.",
                    }
                ]
            },
        )

    try:
        query = query_normalizer(query)
        make, model, year = _apply_session_vehicle_fields(call_id, make, model, year)

        if not make or not model or not year:
            logger.warning(
                "context-less lookup call_id=%r query=%r make=%r model=%r year=%r",
                call_id or None,
                query,
                make or None,
                model or None,
                year or None,
            )

        client: QdrantClient = request.app.state.qdrant
        embedder: TextEmbedding = request.app.state.embedder
        catalog = await asyncio.to_thread(
            get_shop_vehicle_catalog, client, shop_id
        )
        make, model, year = _infer_vehicle_fields(
            query, make, model, year, catalog
        )
        _update_session_vehicle_fields(call_id, make, model, year)

        try:
            chunk_text = await search_top_chunk(
                client=client,
                embedder=embedder,
                query=query,
                year=year,
                make=make,
                model=model,
                shop_id=shop_id,
            )
        except Exception as exc:  # noqa: BLE001 - surface as tool error string
            outcome = "lookup_failed"
            return JSONResponse(
                status_code=200,
                content={
                    "results": [
                        {
                            "toolCallId": tool_call_id,
                            "error": f"Specification lookup failed: {exc}.",
                        }
                    ]
                },
            )

        if chunk_text is None:
            # Catch-all miss: empty DB, unknown vehicle, or below score floors.
            # toolCallId is fixed to "vapi-call" for the stable empty-corpus contract
            # (Vapi ignores mismatch when the tool was invoked as a single call).
            outcome = "not_found"
            return JSONResponse(
                status_code=200,
                content={
                    "results": [
                        {
                            "toolCallId": "vapi-call",
                            "result": (
                                "I don't see that specification anywhere "
                                "in my current manuals."
                            ),
                        }
                    ]
                },
            )

        spoken_result = await asyncio.to_thread(
            format_voice_answer, chunk_text, query
        )
        if not spoken_result:
            outcome = "format_failed"
            return JSONResponse(
                status_code=200,
                content={
                    "results": [
                        {
                            "toolCallId": tool_call_id,
                            "error": (
                                "Specification found but could not be formatted "
                                "for speech."
                            ),
                        }
                    ]
                },
            )

        outcome = "ok"
        return JSONResponse(
            status_code=200,
            content={
                "results": [
                    {
                        "toolCallId": tool_call_id,
                        "result": format_number_for_speech(
                            to_single_line(spoken_result)
                        ),
                    }
                ]
            },
        )
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "vapi_tool outcome=%s elapsed_ms=%.1f query=%r year=%r make=%r "
            "model=%r shop_id=%r",
            outcome,
            elapsed_ms,
            query,
            year or None,
            make or None,
            model or None,
            shop_id,
        )


if __name__ == "__main__":
    port = int((os.getenv("PORT") or "8000").strip() or "8000")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
