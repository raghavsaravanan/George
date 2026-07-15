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

Current local tunnel when started from this machine often looks like:

```text
https://unmanned-rundown-denim.ngrok-free.dev/vapi-tool
```

(Free ngrok hostnames can change — always copy from the running ngrok UI/logs.)

## Re-seed (credit-safe)

```bash
.venv/bin/python ingest.py "intake_manifold_guide.pdf.pdf" \
  --year 2019 --make chevrolet --model silverado

.venv/bin/python ingest.py "AMS Performance VR30 Guide.pdf" \
  --year 2016 --make nissan --model vr30
```

Only if a PDF file itself changed:

```bash
.venv/bin/python ingest.py "path/to/file.pdf" --year YYYY --make MAKE --model MODEL --force-parse
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
  -d '{"query":"lower intake manifold torque","make":"Nissan","model":"vr30","year":"2016"}'
```

## Credit rules

- Default LlamaParse tier is `cost_effective`; `premium_mode` only with `--premium`.
- Local cache short-circuits all remote parse calls.
- Do not use `--force-parse` unless the PDF bytes changed.
