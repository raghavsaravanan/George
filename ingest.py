"""Offline Phase-1 seeder for Project George.

Parses a binary FSM/service PDF via LlamaParse into Markdown, expands
mechanical unit abbreviations for voice-safe storage, isolates Markdown
tables as atomic chunks, embeds them locally with FastEmbed, and upserts
into a file-based Qdrant collection at ./george_mvp_db.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from dotenv import load_dotenv
from fastembed import TextEmbedding
from llama_parse import LlamaParse
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PDF_PATH = PROJECT_ROOT / "intake_manifold_guide.pdf.pdf"
QDRANT_PATH = PROJECT_ROOT / "george_mvp_db"
PARSE_CACHE_DIR = PROJECT_ROOT / ".george_parse_cache"
COLLECTION_NAME = "george_specs"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384

DEFAULT_YEAR = "2019"
DEFAULT_MAKE = "chevrolet"
DEFAULT_MODEL = "silverado"

# Longer / period-spaced patterns first so partial matches cannot corrupt text.
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
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?(?:\s*:?-+:?\s*\|)+\s*:?-+:?\s*\|?\s*$")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")
_CAUTION_LABELS = ("warning", "caution", "note", "important")
_CAUTION_START_RE = re.compile(
    r"^\s*(?:\*{0,2})(?:WARNING|CAUTION|NOTE|IMPORTANT)(?:\*{0,2})?\s*[:\-–]?",
    re.IGNORECASE,
)
_DAMAGE_LINE_RE = re.compile(
    r"damage can occur|cracking or breaking|do not confuse",
    re.IGNORECASE,
)
_LLAMAPARSE_SYSTEM_PROMPT = (
    "OCR all pages carefully when the source is scanned or image-based. "
    "Extract all technical content as Markdown. "
    "Convert every fastener, torque, and specification listing into a strict "
    "Markdown pipe table using | column | syntax with a separator row. "
    "Prefer columns such as Application, Torque, and Notes when present. "
    "Preserve WARNING, CAUTION, and NOTE text immediately after the related "
    "table. Do not omit numeric torque values or unit abbreviations such as "
    "lb. ft., lb-ft, ft lbs, lb. in., lb-in, Nm, or N·m."
)


@dataclass(frozen=True)
class VehicleMetadata:
    year: str
    make: str
    model: str
    document_ref: str

    def normalized(self) -> "VehicleMetadata":
        return VehicleMetadata(
            year=self.year.strip().lower(),
            make=self.make.strip().lower(),
            model=self.model.strip().lower(),
            document_ref=self.document_ref.strip().lower(),
        )


@dataclass(frozen=True)
class SpecChunk:
    text: str
    section: str


def load_api_key() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    import os

    raw = os.getenv("LLAMA_CLOUD_API_KEY")
    if raw is None:
        raise SystemExit(
            "LLAMA_CLOUD_API_KEY is missing from the environment / .env file."
        )
    api_key = raw.strip().strip('"').strip("'")
    if not api_key:
        raise SystemExit(
            "LLAMA_CLOUD_API_KEY is empty after stripping whitespace. "
            "Set a valid LlamaParse API key in .env."
        )
    return api_key


def expand_mechanical_units(text: str) -> str:
    """Rewrite torque / unit abbreviations into spelled-out plain text."""
    expanded = text
    for pattern, replacement in _UNIT_EXPANSION_RULES:
        expanded = pattern.sub(replacement, expanded)
    return expanded


def document_ref_from_path(path: Path) -> str:
    name = path.name
    while name.lower().endswith(".pdf"):
        name = name[: -len(".pdf")]
        if name.endswith("."):
            name = name[:-1]
    ref = name.strip().lower().replace(" ", "_")
    return ref or path.stem.lower()


def _is_table_row(line: str) -> bool:
    stripped = line.rstrip()
    if not stripped:
        return False
    if _TABLE_ROW_RE.match(stripped):
        return True
    if _TABLE_SEPARATOR_RE.match(stripped):
        return True
    return False


def _extract_heading_text(line: str) -> str | None:
    match = _HEADING_RE.match(line.strip())
    if match:
        return match.group(1).strip()
    return None


def _is_caution_heading(heading: str) -> bool:
    label = heading.strip().lower().rstrip(":").strip()
    return label in _CAUTION_LABELS


def _is_caution_body_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _is_table_row(stripped):
        return False
    heading = _extract_heading_text(stripped)
    if heading is not None:
        return False
    if _CAUTION_START_RE.match(stripped):
        return True
    if _DAMAGE_LINE_RE.search(stripped):
        return True
    if stripped.startswith("*") and not stripped.startswith("**"):
        return True
    return True


def _lookahead_nonempty(lines: Sequence[str], start: int) -> int | None:
    index = start
    while index < len(lines):
        if lines[index].strip():
            return index
        index += 1
    return None


def _should_attach_after_table(lines: Sequence[str], index: int) -> bool:
    look = _lookahead_nonempty(lines, index)
    if look is None:
        return False
    candidate = lines[look].strip()
    if _is_table_row(candidate):
        return False
    heading = _extract_heading_text(candidate)
    if heading is not None:
        return _is_caution_heading(heading)
    if _CAUTION_START_RE.match(candidate):
        return True
    if _DAMAGE_LINE_RE.search(candidate):
        return True
    if candidate.startswith("*") and not candidate.startswith("**"):
        return True
    return False


def isolate_markdown_tables(markdown: str) -> list[SpecChunk]:
    """Isolate each Markdown table plus immediately following caution lines.

    Never splits a contiguous table row run across chunk boundaries.
    WARNING/CAUTION/NOTE headings that immediately follow a table stay attached.
    """
    lines = markdown.splitlines()
    chunks: list[SpecChunk] = []
    current_section = ""
    index = 0
    total = len(lines)

    while index < total:
        line = lines[index]
        heading = _extract_heading_text(line)
        if heading is not None:
            if not _is_caution_heading(heading):
                current_section = heading
            index += 1
            continue

        if not _is_table_row(line):
            index += 1
            continue

        table_lines: list[str] = []
        while index < total and _is_table_row(lines[index]):
            table_lines.append(lines[index].rstrip())
            index += 1

        caution_lines: list[str] = []
        while index < total and _should_attach_after_table(lines, index):
            candidate = lines[index]
            if not candidate.strip():
                caution_lines.append("")
                index += 1
                continue

            heading = _extract_heading_text(candidate)
            if heading is not None and _is_caution_heading(heading):
                caution_lines.append(candidate.rstrip())
                index += 1
                while index < total:
                    body = lines[index]
                    if not body.strip():
                        next_idx = _lookahead_nonempty(lines, index + 1)
                        if next_idx is None:
                            break
                        next_line = lines[next_idx]
                        next_heading = _extract_heading_text(next_line)
                        if next_heading is not None:
                            break
                        if _is_table_row(next_line):
                            break
                        caution_lines.append("")
                        index += 1
                        continue
                    if _is_table_row(body):
                        break
                    next_heading = _extract_heading_text(body)
                    if next_heading is not None:
                        break
                    if not _is_caution_body_line(body):
                        break
                    caution_lines.append(body.rstrip())
                    index += 1
                continue

            if _is_caution_body_line(candidate):
                caution_lines.append(candidate.rstrip())
                index += 1
                continue
            break

        while caution_lines and not caution_lines[-1].strip():
            caution_lines.pop()

        parts = table_lines + ([""] + caution_lines if caution_lines else [])
        chunk_text = "\n".join(parts).strip()
        if chunk_text:
            chunks.append(SpecChunk(text=chunk_text, section=current_section))

    return chunks


_TORQUE_SPEC_LINE_RE = re.compile(
    r"^\s*(?:\*{0,2})\s*(.+?)\s*Torque\s+Spec\s*:\s*(.+?)\s*(?:\*{0,2})\s*$",
    re.IGNORECASE,
)
_INLINE_TORQUE_RE = re.compile(
    r"(?P<label>[A-Za-z][A-Za-z0-9 /_-]{2,80}?)\s*[:\-–]\s*"
    r"(?P<value>\d+(?:\s*-\s*\d+)?\s*(?:N\s*[·•\.]?\s*m|Nm|ft\.?\s*-?\s*lbs?|"
    r"lb\.?\s*-?\s*fts?|inch-pounds|foot-pounds|newton-meters)"
    r"(?:[^.\n]{0,60})?)",
    re.IGNORECASE,
)


def isolate_torque_spec_chunks(markdown: str) -> list[SpecChunk]:
    """Fallback for scanned manuals: synthesize a Markdown torque table.

    Used when LlamaParse returns prose torque lines without pipe tables.
    """
    table_rows: list[str] = [
        "| Application | Torque | Notes |",
        "| --- | --- | --- |",
    ]
    caution_lines: list[str] = []
    current_section = "Torque Specifications"
    seen: set[str] = set()

    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        heading = _extract_heading_text(line)
        if heading is not None and not _is_caution_heading(heading):
            if any(
                token in heading.lower()
                for token in ("torque", "fastener", "install", "fuel", "intake")
            ):
                current_section = heading
            continue

        if _CAUTION_START_RE.match(line) or _DAMAGE_LINE_RE.search(line):
            caution_lines.append(line)
            continue

        spec_match = _TORQUE_SPEC_LINE_RE.match(line)
        if spec_match is not None:
            application = spec_match.group(1).strip(" :-*") or "Torque Specification"
            torque = spec_match.group(2).strip(" *-")
            key = f"{application.lower()}::{torque.lower()}"
            if key not in seen and torque:
                seen.add(key)
                table_rows.append(f"| {application} | {torque} | |")
            continue

        inline = _INLINE_TORQUE_RE.search(line)
        if inline is not None and "torque" in line.lower():
            application = inline.group("label").strip(" :-*")
            torque = inline.group("value").strip(" *-")
            key = f"{application.lower()}::{torque.lower()}"
            if key not in seen and torque:
                seen.add(key)
                table_rows.append(f"| {application} | {torque} | |")

    if len(table_rows) <= 2:
        return []

    parts = table_rows
    if caution_lines:
        parts = table_rows + [""] + caution_lines[:3]
    return [SpecChunk(text="\n".join(parts).strip(), section=current_section)]


def collect_spec_chunks(markdown: str) -> list[SpecChunk]:
    """Prefer atomic Markdown tables; fall back to synthesized torque specs."""
    chunks = isolate_markdown_tables(markdown)
    if chunks:
        return chunks
    return isolate_torque_spec_chunks(markdown)


def parse_cache_path(document_ref: str) -> Path:
    return PARSE_CACHE_DIR / f"{document_ref}.md"


def load_cached_markdown(document_ref: str) -> str | None:
    cache_file = parse_cache_path(document_ref)
    if not cache_file.is_file():
        return None
    content = cache_file.read_text(encoding="utf-8").strip()
    return content or None


def save_cached_markdown(document_ref: str, markdown: str) -> Path:
    PARSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = parse_cache_path(document_ref)
    cache_file.write_text(markdown.strip() + "\n", encoding="utf-8")
    return cache_file


async def parse_pdf_to_markdown(
    path: Path,
    api_key: str,
    *,
    force_parse: bool = False,
    premium_mode: bool = False,
) -> str:
    """Async LlamaParse extraction to a single Markdown document string.

    Absolute credit protection:
    1. When force_parse is False and a local cache file exists, read it from disk
       and return immediately — zero LlamaParse HTTP calls.
    2. Remote calls use tier=cost_effective (never agentic), use_vendor_cache for
       LlamaCloud's 48-hour dedupe window, and premium_mode=False unless --premium.
    """
    if not path.is_file():
        raise SystemExit(f"PDF not found: {path}")
    if not path.name.lower().endswith(".pdf"):
        raise SystemExit(
            f"Input must be a PDF path (name ending in .pdf). Got: {path.name}"
        )

    document_ref = document_ref_from_path(path)
    cache_file = parse_cache_path(document_ref)

    # Local filesystem handshake: never open a LlamaParse client when the
    # cache already exists and the caller did not ask to force a re-parse.
    if not force_parse and cache_file.is_file():
        cached_markdown = cache_file.read_text(encoding="utf-8")
        if cached_markdown.strip():
            print(
                "Using local parse cache (no LlamaParse network request): "
                f"{cache_file}"
            )
            return cached_markdown

    # Enforce defaults: cost_effective tier only; premium only via --premium.
    allow_premium = bool(premium_mode)

    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
        verbose=True,
        language="en",
        tier="cost_effective",
        use_vendor_cache=True,
        invalidate_cache=False,
        do_not_cache=False,
        aggressive_table_extraction=True,
        outlined_table_extraction=True,
        premium_mode=allow_premium,
        system_prompt=_LLAMAPARSE_SYSTEM_PROMPT,
    )
    print(
        "Calling LlamaParse "
        f"(tier=cost_effective, use_vendor_cache=True, "
        f"premium_mode={allow_premium}, force_parse={force_parse})..."
    )
    documents = await parser.aload_data(str(path))
    if not documents:
        raise SystemExit(f"LlamaParse returned no documents for {path}")

    pages: list[str] = []
    for document in documents:
        content = (document.get_content() or "").strip()
        if content:
            pages.append(content)

    if not pages:
        raise SystemExit(f"LlamaParse returned empty Markdown for {path}")

    markdown = "\n\n".join(pages)
    saved = save_cached_markdown(document_ref, markdown)
    print(f"Saved parse cache for reuse: {saved}")
    return markdown


def build_points(
    chunks: Sequence[SpecChunk],
    metadata: VehicleMetadata,
    embedding_model: TextEmbedding,
) -> list[qmodels.PointStruct]:
    if not chunks:
        return []

    texts = [chunk.text for chunk in chunks]
    vectors = list(embedding_model.embed(texts))
    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"Embedding count mismatch: {len(vectors)} vectors for {len(chunks)} chunks."
        )

    meta = metadata.normalized()
    points: list[qmodels.PointStruct] = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        points.append(
            qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=list(vector),
                payload={
                    "text": chunk.text,
                    "year": meta.year,
                    "make": meta.make,
                    "model": meta.model,
                    "document_ref": meta.document_ref,
                    "section": chunk.section,
                },
            )
        )
    return points


def ensure_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION_NAME):
        return
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qmodels.VectorParams(
            size=VECTOR_SIZE,
            distance=qmodels.Distance.COSINE,
        ),
    )


def delete_points_for_document(client: QdrantClient, document_ref: str) -> None:
    """Remove prior points for one manual so re-ingest is idempotent."""
    if not client.collection_exists(COLLECTION_NAME):
        return
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="document_ref",
                        match=qmodels.MatchValue(value=document_ref.strip().lower()),
                    )
                ]
            )
        ),
    )


def seed_qdrant(
    points: Sequence[qmodels.PointStruct],
    document_ref: str,
) -> None:
    """Append/replace one document's points without wiping other manuals."""
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_PATH))
    try:
        ensure_collection(client)
        delete_points_for_document(client, document_ref)
        if points:
            client.upsert(collection_name=COLLECTION_NAME, points=list(points))
    finally:
        client.close()


async def run_ingest(
    pdf_path: Path,
    year: str,
    make: str,
    model: str,
    embedding_model: TextEmbedding | None = None,
    *,
    force_parse: bool = False,
    premium_mode: bool = False,
) -> int:
    api_key = load_api_key()
    print(f"Preparing Markdown for: {pdf_path}")
    markdown = await parse_pdf_to_markdown(
        pdf_path,
        api_key,
        force_parse=force_parse,
        premium_mode=premium_mode,
    )

    print("Expanding mechanical units...")
    markdown = expand_mechanical_units(markdown)

    print("Isolating atomic specification chunks...")
    chunks = collect_spec_chunks(markdown)
    if not chunks:
        raise SystemExit(
            "No Markdown tables or torque specifications were isolated from "
            f"LlamaParse output for {pdf_path.name}. Refuse to seed empty points."
        )

    metadata = VehicleMetadata(
        year=year,
        make=make,
        model=model,
        document_ref=document_ref_from_path(pdf_path),
    )

    print(f"Embedding {len(chunks)} chunk(s) with FastEmbed ({EMBEDDING_MODEL_NAME})...")
    model_instance = embedding_model or TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
    points = build_points(chunks, metadata, model_instance)

    print(
        f"Upserting {len(points)} point(s) for document_ref="
        f"{metadata.document_ref!r} into {QDRANT_PATH} / {COLLECTION_NAME}..."
    )
    seed_qdrant(points, document_ref=metadata.document_ref)

    print(f"Seeded {len(points)} point(s).")
    print(f"Metadata: year={metadata.year} make={metadata.make} model={metadata.model}")
    print(f"document_ref={metadata.document_ref}")
    for i, chunk in enumerate(chunks, start=1):
        preview = chunk.text.replace("\n", " | ")
        if len(preview) > 160:
            preview = preview[:157] + "..."
        section_label = chunk.section or "(none)"
        print(f"  [{i}] section={section_label!r} :: {preview}")

    return len(points)


async def run_ingest_many(
    pdf_paths: Sequence[Path],
    year: str,
    make: str,
    model: str,
    *,
    force_parse: bool = False,
    premium_mode: bool = False,
) -> int:
    """Ingest one or more PDFs with the same vehicle metadata for this invocation."""
    if not pdf_paths:
        raise SystemExit("No PDF paths provided.")

    embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
    total = 0
    for pdf_path in pdf_paths:
        print(f"\n=== Ingesting {pdf_path.name} ===")
        total += await run_ingest(
            pdf_path=pdf_path,
            year=year,
            make=make,
            model=model,
            embedding_model=embedding_model,
            force_parse=force_parse,
            premium_mode=premium_mode,
        )
    return total


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest one or more binary PDFs via LlamaParse into George's "
            "local Qdrant store. Each document is replaced by document_ref; "
            "other manuals in the collection are preserved. "
            "Parse results are cached under .george_parse_cache/ to avoid "
            "burning LlamaParse credits on every run."
        )
    )
    parser.add_argument(
        "pdf_paths",
        nargs="*",
        default=[str(DEFAULT_PDF_PATH)],
        help=(
            "One or more PDF paths "
            f"(default: {DEFAULT_PDF_PATH.name})"
        ),
    )
    parser.add_argument("--year", default=DEFAULT_YEAR, help="Vehicle year metadata")
    parser.add_argument("--make", default=DEFAULT_MAKE, help="Vehicle make metadata")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Vehicle model metadata")
    parser.add_argument(
        "--force-parse",
        action="store_true",
        help=(
            "Ignore local/server parse cache and spend LlamaParse credits again. "
            "Use only when the PDF content changed."
        ),
    )
    parser.add_argument(
        "--premium",
        action="store_true",
        help="Enable LlamaParse premium_mode (costs more pages; for hard scans).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    raw_paths = args.pdf_paths if args.pdf_paths else [str(DEFAULT_PDF_PATH)]
    pdf_paths = [Path(p).expanduser().resolve() for p in raw_paths]
    for path in pdf_paths:
        if not path.is_file():
            raise SystemExit(f"PDF not found: {path}")
        if not path.name.lower().endswith(".pdf"):
            raise SystemExit(
                f"Input must be a PDF path (name ending in .pdf). Got: {path.name}"
            )

    try:
        count = asyncio.run(
            run_ingest_many(
                pdf_paths=pdf_paths,
                year=args.year,
                make=args.make,
                model=args.model,
                force_parse=args.force_parse,
                premium_mode=args.premium,
            )
        )
    except KeyboardInterrupt:
        print("Ingest interrupted.", file=sys.stderr)
        raise SystemExit(130) from None

    print(f"\nTotal points upserted this run: {count}")
    if count <= 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
