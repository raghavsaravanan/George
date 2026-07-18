"""Render / local shim: repo-root ``uvicorn main:app`` loads backend/main.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_BACKEND_MAIN = Path(__file__).resolve().parent / "backend" / "main.py"
if not _BACKEND_MAIN.is_file():
    raise SystemExit(f"backend entrypoint missing: {_BACKEND_MAIN}")

_BACKEND_DIR = str(_BACKEND_MAIN.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_spec = importlib.util.spec_from_file_location("george_backend_main", _BACKEND_MAIN)
if _spec is None or _spec.loader is None:
    raise SystemExit(f"unable to load module spec for {_BACKEND_MAIN}")

_module = importlib.util.module_from_spec(_spec)
sys.modules["george_backend_main"] = _module
_spec.loader.exec_module(_module)

app = _module.app
