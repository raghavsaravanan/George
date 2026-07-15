#!/usr/bin/env bash
# Validate local George prerequisites for demo/dev.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DB_PATH="${ROOT}/george_mvp_db"

if [[ ! -d "${DB_PATH}" ]]; then
  echo "Database missing. Run \`python ingest.py\` first."
  exit 1
fi

echo "george_mvp_db found at ${DB_PATH}"
exit 0
