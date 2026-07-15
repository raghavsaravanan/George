"""Live Phase-2 webhook for Project George.

FastAPI endpoint that accepts Vapi custom tool-call webhooks, embeds the
technician query locally with FastEmbed, applies a strict year/make/model
Qdrant metadata filter against the offline-seeded george_specs collection,
and returns a single-line 1-2 sentence spoken result for headset TTS.
"""

from __future__ import annotations

import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

PROJECT_ROOT = Path(__file__).resolve().parent
QDRANT_PATH = PROJECT_ROOT / "george_mvp_db"
COLLECTION_NAME = "george_specs"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

_UNIT_EXPANSION_RULES: Sequence[tuple[re.Pattern[str], str]] = (
    (re.compile(r"\blb\.\s*ft\.", re.IGNORECASE), "foot-pounds"),
    (re.compile(r"\blb\.\s*ft\b", re.IGNORECASE), "foot-pounds"),
    (re.compile(r"\blb\.\-ft\.?", re.IGNORECASE), "foot-pounds"),
    (re.compile(r"\blb\s*-\s*ft\.?\b", re.IGNORECASE), "foot-pounds"),
    (re.compile(r"\bft\.\s*lb\.?", re.IGNORECASE), "foot-pounds"),
    (re.compile(r"\bft\s*-\s*lb\.?\b", re.IGNORECASE), "foot-pounds"),
    (re.compile(r"\bft\s+lbs?\b", re.IGNORECASE), "foot-pounds"),
    (re.compile(r"(?<=\d)ft\s*lbs?\b", re.IGNORECASE), " foot-pounds"),
    (re.compile(r"\blb\.\s*in\.", re.IGNORECASE), "inch-pounds"),
    (re.compile(r"\blb\.\s*in\b", re.IGNORECASE), "inch-pounds"),
    (re.compile(r"\blb\.\-in\.?", re.IGNORECASE), "inch-pounds"),
    (re.compile(r"\blb\s*-\s*in\.?\b", re.IGNORECASE), "inch-pounds"),
    (re.compile(r"\bin\.\s*lb\.?", re.IGNORECASE), "inch-pounds"),
    (re.compile(r"\bin\s*-\s*lb\.?\b", re.IGNORECASE), "inch-pounds"),
    (re.compile(r"\bin\s+lbs?\b", re.IGNORECASE), "inch-pounds"),
    (re.compile(r"\bN\s*[·•\.]\s*m\b"), "newton-meters"),
    (re.compile(r"\bN\s*-\s*m\b"), "newton-meters"),
    (re.compile(r"\bNm\b"), "newton-meters"),
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


@dataclass(frozen=True)
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


def to_single_line(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\n", " ").replace("\t", " ")
    cleaned = _HTML_ENTITY_RE.sub(" ", cleaned)
    cleaned = _MARKDOWN_EMPHASIS_RE.sub("", cleaned)
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def expand_mechanical_units(text: str) -> str:
    expanded = text
    for pattern, replacement in _UNIT_EXPANSION_RULES:
        expanded = pattern.sub(replacement, expanded)
    return expanded


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
        return to_single_line(" ".join(sentences[:2]))

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
    return to_single_line(expand_mechanical_units(clipped))


def _payload_text(hit: Any) -> str | None:
    payload = getattr(hit, "payload", None) or {}
    if not isinstance(payload, dict):
        return None
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
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


def search_top_chunk(
    client: QdrantClient,
    embedder: TextEmbedding,
    query: str,
    year: str,
    make: str,
    model: str,
) -> str | None:
    """Hybrid retrieval: exact metadata first, then soft match, then semantic.

    Exact year/make/model matches are preferred when present in Qdrant, but
    mismatched or fuzzy vehicle strings still return the closest technical
    chunk via unfiltered vector similarity.
    """
    query_vector = _query_vector_for(embedder, query, year, make, model)
    if query_vector is None:
        return None

    # Pass 1 — strict AND filter (character-for-character metadata match).
    if year and make and model:
        strict_filter = qmodels.Filter(
            must=[
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
        text = _payload_text(hits[0]) if hits else None
        if text:
            return text

    # Pass 2 — soft OR filter: any provided vehicle field may match.
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
            min_should=qmodels.MinShould(
                conditions=soft_should,
                min_count=1,
            ),
        )
        hits = _search_points(client, query_vector, soft_filter, limit=1)
        text = _payload_text(hits[0]) if hits else None
        if text:
            return text
    # Pass 3 — pure semantic fallback (no metadata lock).
    hits = _search_points(client, query_vector, query_filter=None, limit=1)
    if not hits:
        return None
    return _payload_text(hits[0])


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
    """Pull query/make/model/year from a Vapi toolCallList entry safely."""
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
        chunk_text = search_top_chunk(
            client=client,
            embedder=embedder,
            query=args["query"],
            year=args["year"],
            make=args["make"],
            model=args["model"],
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
    if not QDRANT_PATH.exists():
        raise RuntimeError(
            f"Qdrant path does not exist: {QDRANT_PATH}. Run ingest.py first."
        )

    embedder = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
    client = QdrantClient(path=str(QDRANT_PATH))

    if not client.collection_exists(COLLECTION_NAME):
        client.close()
        raise RuntimeError(
            f"Qdrant collection '{COLLECTION_NAME}' was not found at {QDRANT_PATH}."
        )

    app.state.embedder = embedder
    app.state.qdrant = client
    try:
        yield
    finally:
        client.close()


app = FastAPI(
    title="George Vapi Tool Webhook",
    version="1.0.0",
    lifespan=lifespan,
)


def _infer_vehicle_fields(
    query: str,
    make: str,
    model: str,
    year: str,
) -> tuple[str, str, str]:
    """Fill missing make/model/year from query keywords for known manuals."""
    q = query.lower()
    make_out, model_out, year_out = make, model, year

    if any(token in q for token in ("vr30", "ams", "q50", "q60", "infiniti")):
        if not make_out:
            make_out = "nissan"
        if not model_out:
            model_out = "vr30"
        if not year_out:
            year_out = "2016"

    if "silverado" in q or ("chevy" in q) or ("chevrolet" in q):
        if not make_out:
            make_out = "chevrolet"
        if not model_out and "silverado" in q:
            model_out = "silverado"
        if not year_out and "2019" in q:
            year_out = "2019"
        # Default Chevy demo vehicle when Silverado is named without a year.
        if not year_out and model_out == "silverado":
            year_out = "2019"

    return make_out, model_out, year_out


@app.post("/vapi-tool")
async def vapi_tool(request: Request) -> JSONResponse:
    """Accept flat apiRequest body: {query, make, model, year}.

    Only ``query`` is required. Missing vehicle fields are inferred from the
    query when possible so the voice agent does not need to interrogate the
    tech for year/make/model on every turn.
    """
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

    query = str(payload.get("query") or "").strip()
    make = str(payload.get("make") or "").strip().lower()
    model = str(payload.get("model") or "").strip().lower()
    year = str(payload.get("year") or "").strip().lower()

    if not query:
        return JSONResponse(
            status_code=200,
            content={
                "results": [
                    {
                        "toolCallId": "vapi-call",
                        "error": "Missing required field: query.",
                    }
                ]
            },
        )

    make, model, year = _infer_vehicle_fields(query, make, model, year)

    client: QdrantClient = request.app.state.qdrant
    embedder: TextEmbedding = request.app.state.embedder

    try:
        chunk_text = search_top_chunk(
            client=client,
            embedder=embedder,
            query=query,
            year=year,
            make=make,
            model=model,
        )
    except Exception as exc:  # noqa: BLE001 - surface as tool error string
        return JSONResponse(
            status_code=200,
            content={
                "results": [
                    {
                        "toolCallId": "vapi-call",
                        "error": f"Specification lookup failed: {exc}.",
                    }
                ]
            },
        )

    if chunk_text is None:
        return JSONResponse(
            status_code=200,
            content={
                "results": [
                    {
                        "toolCallId": "vapi-call",
                        "error": (
                            "No specification found for that year make and model."
                        ),
                    }
                ]
            },
        )

    spoken_result = format_voice_answer(chunk_text, query)
    if not spoken_result:
        return JSONResponse(
            status_code=200,
            content={
                "results": [
                    {
                        "toolCallId": "vapi-call",
                        "error": (
                            "Specification found but could not be formatted "
                            "for speech."
                        ),
                    }
                ]
            },
        )

    return JSONResponse(
        status_code=200,
        content={
            "results": [
                {
                    "toolCallId": "vapi-call",
                    "result": to_single_line(spoken_result),
                }
            ]
        },
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
