#!/usr/bin/env bash
# -----------------------------------------------------------------------------
#  devsetup.sh – bootstrap HCD‑book‑recommender in one command
#
#  Usage:
#     git clone https://github.com/VikramjeetSinghKundu/HCD-book-recommender.git
#     cd HCD-book-recommender
#     chmod +x devsetup.sh          # only needed once if the bit isn’t set
#     ./devsetup.sh                 # ← runs everything
#     source .venv/bin/activate
#     python -m streamlit run streamlit_app.py
# -----------------------------------------------------------------------------
set -euo pipefail

### 0️⃣  Prep
PROJECT_ROOT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"        # override with PYTHON_BIN=python3.11 …

echo "▸ Creating / re‑using virtual‑env at .venv/"
if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

### 1️⃣  Python deps
echo "▸ Installing Python packages (this may take a minute)…"
python -m pip install --upgrade pip wheel >/dev/null
# Streamlit pins packaging<25; install it first to avoid resolver hiccup
python -m pip install "packaging<25" >/dev/null

# main requirements
python -m pip install -r requirements.txt >/dev/null
python -m pip install streamlit torch faiss-cpu >/dev/null

### 2️⃣  macOS / Apple‑silicon OpenMP fix
if [[ "$(uname)" == "Darwin" ]]; then
  echo "▸ macOS detected – ensuring single system‑wide libomp"
  if ! command -v brew &>/dev/null; then
    echo "  ✗ Homebrew not found – please install Homebrew first." >&2
    exit 1
  fi
  brew list libomp &>/dev/null || brew install libomp

  # prepend once to .zprofile if not already present
  OMP_PATH='/opt/homebrew/opt/libomp/lib'
  if ! grep -qs "$OMP_PATH" "$HOME/.zprofile"; then
    echo "export DYLD_LIBRARY_PATH=\"$OMP_PATH:\${DYLD_LIBRARY_PATH:-}\"" >> "$HOME/.zprofile"
    echo "  • Added DYLD_LIBRARY_PATH to ~/.zprofile (open a new shell to pick it up)"
    # also export for this _current_ shell:
    export DYLD_LIBRARY_PATH="$OMP_PATH:${DYLD_LIBRARY_PATH:-}"
  fi

  echo "  • Removing wheel‑bundled libomp duplicates…"
  python - <<'PY'
import site, pathlib, os
site_pkgs = pathlib.Path(next(p for p in site.getsitepackages() if "site-packages" in p))
for p in site_pkgs.rglob("libomp*.dylib"):
    if any(parent in ("torch", "faiss") for parent in p.parts):
        print("    – nixing", p.relative_to(site_pkgs))
        try:
            p.unlink()
        except FileNotFoundError:
            pass
PY
fi

### 3️⃣  Cache Hugging Face dataset locally
echo "▸ Downloading parquet data from Hugging Face (first run only)…"
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="VikramjeetSingh/books-recs",
    repo_type="dataset",
    allow_patterns=[
        "books_processed.parquet",
        "books_metadata_small.parquet",
    ],
    local_dir="hf_cache",                 # caches inside the repo
    local_dir_use_symlinks=False,
    resume_download=False
)
print("    ✓ parquet files cached in hf_cache/")
PY

### 4️⃣  Sanity‑check imports
echo "▸ Verifying critical imports…"
python - <<'PY'
import importlib.util, pathlib, sys
for m in ("streamlit", "torch", "faiss"):
    spec = importlib.util.find_spec(m)
    if spec is None:
        sys.exit(f"✗ {m} failed to import")
    print("  ✓", m.ljust(9), "→", pathlib.Path(spec.origin).parent)
print("✓ All good – you can now 'source .venv/bin/activate' and launch Streamlit!")
PY
