#!/usr/bin/env bash
# Validate George prerequisites for cloud-first demos.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

if [[ -f .env ]]; then
  # Prefer python-dotenv style KEY=value; tolerate KEY = value via xargs trim below.
  set -a
  # shellcheck disable=SC1091
  source .env || true
  set +a
fi

QDRANT_URL="${QDRANT_URL:-}"
QDRANT_API_KEY="${QDRANT_API_KEY:-}"

# Trim accidental spaces around values from KEY = value style .env files
QDRANT_URL="$(echo "${QDRANT_URL}" | xargs)"
QDRANT_API_KEY="$(echo "${QDRANT_API_KEY}" | xargs)"

if [[ -n "${QDRANT_URL}" && -n "${QDRANT_API_KEY}" ]]; then
  echo "Cloud Qdrant configured: ${QDRANT_URL}"
  echo "Runtime DB is Qdrant Cloud. Local george_mvp_db/ is optional."
  echo "Empty wipe:  ../.venv/bin/python purge_db.py"
  echo "Re-seed:     ../.venv/bin/python ingest.py <pdf> --shop-id shop_demo ..."
  exit 0
fi

DB_PATH="${ROOT}/george_mvp_db"
if [[ ! -d "${DB_PATH}" ]]; then
  echo "Database missing. Run \`python ingest.py\` first."
  echo "(Or set QDRANT_URL + QDRANT_API_KEY in backend/.env for cloud.)"
  exit 1
fi

echo "Local george_mvp_db found at ${DB_PATH}"
exit 0
