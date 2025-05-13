#!/usr/bin/env bash
###############################################################################
# devsetup.sh  –  one‑shot bootstrap for HCD‑book‑recommender on macOS
#
# Usage (inside a fresh clone):
#   chmod +x devsetup.sh      # once per clone
#   ./devsetup.sh             # sets up .venv, deps, libomp, sanity‑checks
#
# After it says “✓ imports OK”:
#   source .venv/bin/activate
#   python -m streamlit run streamlit_app.py
###############################################################################
set -euo pipefail

echo "▶ creating / activating Python 3.12 virtual‑env …"
if [[ ! -d .venv ]]; then
  python3.12 -m venv .venv        # fails if 3.12 isn’t available
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "▶ upgrading pip / wheel …"
python -m pip install --upgrade --quiet pip wheel

echo "▶ installing project dependencies …"
python -m pip install --quiet -r requirements.txt
python -m pip install --quiet streamlit torch faiss-cpu

###############################################################################
# macOS only ‑‑ ensure *one* canonical libomp.dylib comes from Homebrew
###############################################################################
if [[ "$(uname)" == "Darwin" ]]; then
  echo "▶ verifying Homebrew presence …"
  if ! command -v brew >/dev/null; then
    echo "✖ Homebrew not found. Install it from https://brew.sh first." >&2
    exit 1
  fi
  BREW_PREFIX=$(brew --prefix)

  if ! brew list libomp >/dev/null 2>&1; then
    echo "• installing libomp …"
    brew install libomp
  fi

  OMP_LIB="$BREW_PREFIX/opt/libomp/lib"
  export DYLD_LIBRARY_PATH="$OMP_LIB:${DYLD_LIBRARY_PATH:-}"
  # persist for future shells
  if ! grep -q "$OMP_LIB" "${HOME}/.zprofile" 2>/dev/null; then
    echo "export DYLD_LIBRARY_PATH=\"$OMP_LIB:\$DYLD_LIBRARY_PATH\"" \
      >> "${HOME}/.zprofile"
  fi

  echo "▶ removing wheel‑bundled copies of libomp.dylib …"
  python - <<'PY'
import pathlib, site, sys, os
site_pkgs = pathlib.Path(next(p for p in site.getsitepackages()
                              if "site-packages" in p))
removed = 0
for lib in site_pkgs.rglob("libomp*.dylib"):
    if any(parent in ("torch", "faiss") for parent in lib.parts):
        print("  removing", lib.relative_to(site_pkgs))
        lib.unlink()
        removed += 1
print(f"  → {removed} duplicate(s) removed")
PY
fi  # Darwin

###############################################################################
# Final sanity‑check
###############################################################################
echo "▶ sanity‑checking key imports …"
python - <<'PY'
import importlib.util, pathlib, sys
for mod in ("streamlit", "torch", "faiss"):
    spec = importlib.util.find_spec(mod)
    if spec is None:
        print(f"✖ {mod} NOT importable", file=sys.stderr)
        sys.exit(1)
    print(f"✓ {mod:9} →", pathlib.Path(spec.origin).parent)
print("\n✓  imports OK – now run:\n   source .venv/bin/activate && "
      "python -m streamlit run streamlit_app.py")
PY

