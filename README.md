# George

Hands-free voice RAG for automotive technicians. Ask torque and install specs out loud; George answers in one or two clear sentences from shop manuals — not from guesswork.

```text
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│ Manual PDFs │ ──► │  ingest.py   │ ──► │ george_mvp_db   │
│ (offline)   │     │ LlamaParse + │     │ Qdrant vectors  │
└─────────────┘     │ FastEmbed    │     │ year/make/model │
                    └──────────────┘     └────────┬────────┘
                                                  │
Mechanic ──► Vapi ──► ngrok ──► main.py /vapi-tool ◄┘
                         │              FastEmbed search
                         └──────── spoken 1–2 sentences ──► headset
```

## Why two Python files?

| File | Role | When it runs |
| --- | --- | --- |
| `ingest.py` | Offline seeder | You add/update PDFs |
| `main.py` | Live Vapi webhook | Every Call / tool hit |

Decoupled on purpose: eating manuals is slow and credit-sensitive; answering must stay sub-second.

## Repository map

```text
George/
├── ingest.py                 # PDF → chunks → Qdrant
├── main.py                   # FastAPI POST /vapi-tool
├── requirements.txt
├── DEMO.md                   # Shop-demo runbook
├── intake_manifold_guide.pdf.pdf
├── AMS Performance VR30 Guide.pdf
├── .env                      # LLAMA_CLOUD_API_KEY (gitignored)
├── george_mvp_db/            # Local Qdrant (gitignored)
└── .george_parse_cache/      # Parsed Markdown cache (gitignored)
```

## Phase 1 — Ingest (build the brain)

```text
PDF
 │
 ├─ local cache hit? ──yes──► skip LlamaParse (no credits)
 │
 └─no──► LlamaParse (tier=cost_effective)
           │
           ▼
        Markdown tables (+ torque fallback for scans)
           │
           ▼
        Expand units (lb. ft. → foot-pounds)
           │
           ▼
        Atomic table chunks (never split mid-table)
           │
           ▼
        FastEmbed (BAAI/bge-small-en-v1.5, 384-d)
           │
           ▼
        Qdrant upsert by document_ref
        payload: text, year, make, model, section
```

**Current corpus**

| PDF | Tags |
| --- | --- |
| Intake manifold guide | 2019 · chevrolet · silverado |
| AMS VR30 guide | 2016 · nissan · vr30 |

```bash
python ingest.py "intake_manifold_guide.pdf.pdf" --year 2019 --make chevrolet --model silverado
python ingest.py "AMS Performance VR30 Guide.pdf" --year 2016 --make nissan --model vr30
# PDF changed? add --force-parse   | hard scan? add --premium
```

## Phase 2 — Live voice (use the brain)

```text
"What's the lower intake torque on a VR30?"
                │
                ▼
         Vapi STT + LLM
                │
                ▼
   POST /vapi-tool
   { query, make, model, year }
                │
                ▼
   Hybrid search in Qdrant
   1) exact year+make+model
   2) soft OR match
   3) semantic fallback
                │
                ▼
   format_voice_answer → 1–2 sentences
                │
                ▼
   { "results": [{ "toolCallId": "vapi-call", "result": "..." }] }
                │
                ▼
         Vapi TTS → mechanic
```

## Quick start (Call test)

```bash
# Terminal 1
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2
ngrok http 8000
```

Vapi tool `lookup_spec` Server URL: `https://YOUR-NGROK-HOST/vapi-tool`

See [DEMO.md](DEMO.md) for smoke questions and curl examples.

## Stack

| Layer | Choice |
| --- | --- |
| Parse | LlamaParse (`cost_effective`) + local `.george_parse_cache/` |
| Embed | FastEmbed local CPU |
| Store | File Qdrant `./george_mvp_db` / `george_specs` |
| API | FastAPI |
| Voice | Vapi (WebRTC + STT/TTS) |
| Tunnel | Ngrok (dev) |

## Design rules

- **Verbal:** ≤2 spoken sentences, no filler
- **Units:** always `foot-pounds` / `inch-pounds` / `newton-meters`
- **Tables:** never clip across chunk boundaries
- **Credits:** cache-first; never force-parse unless the PDF changed

## What’s next

Frontend bay picker, hosted API + managed Qdrant, persistent public URL — after multi-PDF Call demos stay green.
