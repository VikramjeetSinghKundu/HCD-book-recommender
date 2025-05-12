#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# devsetup.sh  –  one‑shot bootstrap for HCD‑book‑recommender
#   ▸ creates .venv
#   ▸ installs Python deps (streamlit, torch, faiss‑cpu)
#   ▸ makes sure a single, system OpenMP (libomp) is visible
#   ▸ launches the Streamlit app
# Tested on: macOS 14 (Apple Silicon), Ubuntu 22.04
# ─────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

PY=python3.12    
VENV=".venv"

echo "▶ creating virtual‑env …"
$PY -m venv "$VENV"

echo "▶ activating virtual‑env …"
# shellcheck disable=SC1090
source "$VENV/bin/activate"

echo "▶ upgrading pip & installing packages …"
python -m pip install --upgrade pip wheel
python -m pip install streamlit torch faiss-cpu

# ── macOS + Homebrew: ensure ONE canonical libomp ─────────────────────────────
if [[ "$(uname -s)" == "Darwin" ]]; then
  if ! brew ls --versions libomp >/dev/null; then
    echo "▶ installing Homebrew’s libomp …"
    brew install libomp
  fi
  export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:${DYLD_LIBRARY_PATH:-}"
fi

# ── delete wheel‑bundled libomp duplicates from torch / faiss ─────────────────
python - <<'PY'
import site, pathlib, sys
site_pkgs = pathlib.Path(next(p for p in site.getsitepackages() if "site-packages" in p))
for p in site_pkgs.rglob("libomp*.dylib"):
    if any(parent in ("torch", "faiss") for parent in p.parts):
        print("  removing stray", p.relative_to(site_pkgs))
        p.unlink()
PY

echo "▶ sanity‑checking imports …"
python - <<'PY'
import importlib, sys
for m in ("streamlit", "torch", "faiss"):
    spec = importlib.util.find_spec(m)
    print(f"  {m:<9} ✓  ({spec.origin})")
PY

echo -e "\n  Environment ready – launching the app!  (Ctrl‑C to quit)\n"
exec python -m streamlit run streamlit_app.py
