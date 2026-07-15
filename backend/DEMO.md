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
cd /Users/raghav.s18/Documents/Projects/George/backend

# Terminal 1 — API (only one process; holds local Qdrant lock)
../.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2 — tunnel
ngrok http 8000
```

Vapi → George assistant → tool `lookup_spec` → Server URL:

```text
# Local demo
https://YOUR-NGROK-HOST/vapi-tool

# Production (Render)
https://YOUR-SERVICE.onrender.com/vapi-tool
```

## Vapi webhook secret (`x-vapi-secret`)

`backend/.env` must have matching `VAPI_WEBHOOK_SECRET=...` (restart uvicorn after changes).
This is **not** on the **API Keys** page (Private/Public keys are for calling Vapi).
This is **not** **Encryption Settings**.

### Preferred: HTTP header on `lookup_spec`

1. **Build → Tools → `lookup_spec`**
2. Keep Request URL: `https://YOUR-NGROK-HOST/vapi-tool`
3. Find **Headers** (often near Auth / Request / Advanced — not Encryption)
4. Add header:
   - Name: `X-Vapi-Secret`
   - Value: same string as `VAPI_WEBHOOK_SECRET` in `backend/.env`
5. Save

### Alternate: Custom Credential

1. Dashboard → **Custom Credentials** (org settings; not API Keys)
2. Create **Bearer Token**:
   - Header Name: `X-Vapi-Secret`
   - Include Bearer Prefix: **off**
   - Token: same as `VAPI_WEBHOOK_SECRET`
3. On `lookup_spec`, attach that credential under **Authentication** / `credentialId`

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
You are George, a hands-free shop-floor automotive voice assistant for technicians wearing headphones.

MISSION
Answer torque, fastener, gasket, and install-spec questions ONLY from the lookup_spec tool. You do not invent specs. The tool result is the source of truth.

WHEN TO CALL lookup_spec
- Call lookup_spec on EVERY question about torque, bolts, nuts, fasteners, gaskets, torque sequences, install notes, or service-manual specs.
- Call it again for EVERY new question in the same call, even if the vehicle is the same.
- Never answer a numbers/specs question from memory, prior turns, or general car knowledge.

HOW TO CALL lookup_spec
- query: the mechanic's request in their words (required).
- make / model / year: only if the mechanic clearly stated them. All are optional.
- Do NOT interrogate for year if they already named the platform (Silverado, Chevy, Chevrolet, AMS, VR30, Q50, Q60, Infiniti, Nissan VR30, etc.).
- If speech-to-text looks like BR30 / DR30 / V R 30 / AMS-BR30, treat it as VR30 / AMS VR30 and still call the tool. Prefer spelling VR30 in your spoken reply only if you must name the engine yourself; otherwise read the tool result as-is.
- Never skip the tool because a similar question was asked earlier.

AFTER THE TOOL RETURNS (CRITICAL — VERBATIM MODE)
- If result is present: speak the result string EXACTLY. Character-for-character intent. No paraphrase. No rewrite. No synonym swap.
- Forbidden rewrites include: "ft", "ft.", "ft-lb", "ft-lbs", "lb-ft", "lb.ft", "in-lb", "N·m", "Nm", "N.m", or any abbreviation of foot-pounds / inch-pounds / newton-meters.
- Keep every unit word exactly as returned: "foot-pounds", "inch-pounds", "newton-meters".
- Do not add greetings, emojis, "sure", "absolutely", or extra shop chatter.
- Do not merge this answer with a previous answer.
- Do not reuse numbers, cautions, or sentences from earlier tool results.
- Do not substitute a Chevy/Silverado answer for an AMS/VR30 answer or vice versa.
- If the tool returns an error string: say that you do not have that spec in the manuals, briefly. Only then ask for year/make/model if the vehicle is still unknown.

CONVERSATION STYLE
- Voice-first: 1–2 short sentences max, matching the tool string.
- Confirm the working vehicle only if it helps; do not block the lookup.
- If the mechanic is clearly chatting (not asking a spec), reply briefly without calling the tool.
- Never invent Fel-Pro numbers, torque values, sequences, or warnings.

FAILURE MODES YOU MUST AVOID
- Saying "15-18 ft" instead of "15-18 foot-pounds".
- Saying "15 N·m or 11 ft" instead of "15 newton-meters or 11 foot-pounds".
- Answering AMS lower-intake with the Silverado intake-bolt reply.
- Using an earlier result after a new lookup_spec completed.
```

### Tool description (paste on the tool)

```text
Looks up torque and fastener specs from shop manuals. ALWAYS call for every new torque/spec question in a call. Pass query always. make, model, and year are optional — omit year when the platform is clear (VR30/AMS/BR30/DR30 mishears → Nissan VR30; Silverado → Chevrolet). The API returns a ready-to-speak result string. The assistant MUST speak that result EXACTLY with no paraphrasing, no unit abbreviations, and no reuse of prior-turn answers. Never invent values.
```

## Re-seed (credit-safe)

```bash
cd /Users/raghav.s18/Documents/Projects/George/backend

../.venv/bin/python ingest.py "intake_manifold_guide.pdf.pdf" \
  --year 2019 --make chevrolet --model silverado

../.venv/bin/python ingest.py "AMS Performance VR30 Guide.pdf" \
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
# Without secret → 401 if VAPI_WEBHOOK_SECRET is set
curl -s -X POST "http://127.0.0.1:8000/vapi-tool" \
  -H "Content-Type: application/json" \
  -H "x-vapi-secret: YOUR_SECRET_HERE" \
  -d '{"query":"lower intake manifold torque AMS VR30"}'
```

Run the API from `backend/` so `.env` and local paths resolve correctly.

## Credit rules

- Default LlamaParse tier is `cost_effective`; `premium_mode` only with `--premium`.
- Local cache short-circuits all remote parse calls.
- Do not use `--force-parse` unless the PDF bytes changed.
