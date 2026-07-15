# George shop-demo runbook

Keep this path green before building a frontend.

## Current corpus

| PDF | document_ref | year | make | model |
| --- | --- | --- | --- | --- |
| `intake_manifold_guide.pdf.pdf` | `intake_manifold_guide` | 2019 | chevrolet | silverado |
| `AMS Performance VR30 Guide.pdf` | `ams_performance_vr30_guide` | 2016 | nissan | vr30 |

Parse Markdown is cached in `.george_parse_cache/` (gitignored). Re-ingest does **not** call LlamaParse unless you pass `--force-parse`.

## Start for a Call test (3 terminals)

```bash
cd /Users/raghav.s18/Documents/Projects/George

# Terminal 1 — API (only one process; holds local Qdrant lock)
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2 — tunnel
ngrok http 8000
```

Vapi → George assistant → tool `lookup_spec` → Server URL:

```text
https://YOUR-NGROK-HOST/vapi-tool
```

## Vapi setup (stops “what year?” loops)

### Tool `lookup_spec` parameters

| Param | Required? | Notes |
| --- | --- | --- |
| `query` | **Yes** | What they’re asking |
| `make` | No | e.g. chevrolet / nissan |
| `model` | No | e.g. silverado / vr30 |
| `year` | **No** | Uncheck required in Vapi |

If the tech says “AMS VR30”, the API infers `nissan` / `vr30` / `2016`. If they say “2019 Silverado”, it infers Chevy tags.

### Paste into Assistant **System Prompt**

```text
You are George, a shop-floor automotive voice assistant.

RULES:
1. For any torque, fastener, gasket, or install-spec question, ALWAYS call lookup_spec first.
2. Pass query = the mechanic's question. Pass make/model/year only if the mechanic said them. Do NOT ask for year if they already named the vehicle (Silverado, VR30, AMS, etc.).
3. After lookup_spec returns a result, speak that result almost word-for-word. Do not ask clarifying questions after a successful lookup.
4. Never shorten units. Keep "foot-pounds", "inch-pounds", "newton-meters" exactly.
5. Keep answers to 1-2 short sentences. No filler.
6. Only ask for year/make/model if the tool returns an error AND the vehicle is truly unknown.
```

### Tool description (paste on the tool)

```text
Looks up torque and fastener specs from shop manuals. Pass query always. make, model, and year are optional — omit year rather than asking the user when the platform is clear (e.g. VR30/AMS → Nissan VR30; Silverado → Chevrolet). Speak the returned result string exactly; do not invent values.
```

## Re-seed (credit-safe)

```bash
.venv/bin/python ingest.py "intake_manifold_guide.pdf.pdf" \
  --year 2019 --make chevrolet --model silverado

.venv/bin/python ingest.py "AMS Performance VR30 Guide.pdf" \
  --year 2016 --make nissan --model vr30
```

After ingest, restart uvicorn.

## Smoke questions

1. “Intake manifold bolt torque for a 2019 Silverado?”
2. “Fel-Pro gasket for application 226042?”
3. “Lower intake manifold torque on the AMS VR30 guide?”
4. “Fuel nut torque for AMS VR30?”
5. “Upper intake manifold torque for Nissan VR30?”

## Curl

```bash
curl -s -X POST "http://127.0.0.1:8000/vapi-tool" \
  -H "Content-Type: application/json" \
  -d '{"query":"lower intake manifold torque AMS VR30"}'
```

## Credit rules

- Default LlamaParse tier is `cost_effective`; `premium_mode` only with `--premium`.
- Local cache short-circuits all remote parse calls.
- Do not use `--force-parse` unless the PDF bytes changed.
