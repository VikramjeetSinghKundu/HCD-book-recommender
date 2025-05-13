#!/usr/bin/env bash
# devsetup.sh – one‑shot bootstrap for HCD‑book‑recommender
# ---------------------------------------------------------
set -euo pipefail
IFS=$'\n\t'

banner()  { printf "\n\033[1;36m▶ %s\033[0m\n" "$*"; }
die()     { printf "\033[1;31m✖ %s\033[0m\n" "$*" ; exit 1; }

################################################################################
# 0) picking a recent Python (prefers 3.12, falls back to `python3`)
################################################################################
PYBIN=$(command -v python3.12 || command -v python3 || die "Need Python ≥3.10")
banner "using $($PYBIN --version)"

################################################################################
# 1) create / reusing the venv
################################################################################
[[ -d .venv ]] || { banner "creating venv"; "$PYBIN" -m venv .venv; }
source .venv/bin/activate
python -m pip install --upgrade pip wheel >/dev/null

################################################################################
# 2) installing dependencies
################################################################################
banner "installing Python deps"
python -m pip install -r requirements.txt streamlit torch faiss-cpu >/dev/null

################################################################################
# 3) macOS‑only: ensures a single, system‑wide OpenMP (libomp)
################################################################################
if [[ "$(uname)" == "Darwin" ]]; then
  banner "checking Homebrew libomp"
  if ! brew list libomp &>/dev/null; then brew install libomp; fi

  # Choose correct Homebrew prefix (Intel vs Apple Silicon)
  OMP_PREFIX="/opt/homebrew/opt/libomp/lib"
  [[ -d $OMP_PREFIX ]] || OMP_PREFIX="/usr/local/opt/libomp/lib"
  export DYLD_LIBRARY_PATH="$OMP_PREFIX:${DYLD_LIBRARY_PATH-}"
fi

################################################################################
# 4) removing wheel‑bundled libomp duplicates (torch / faiss)
################################################################################
banner "removing wheel‑bundled libomp copies"
python - <<'PY'
import pathlib, site, sys, os
site_pkgs = pathlib.Path(next(p for p in site.getsitepackages() if "site-packages" in p))
for dylib in site_pkgs.rglob("libomp*.dylib"):
    if any(parent in ("torch", "faiss") for parent in dylib.parts):
        print("  deleting", dylib.relative_to(site_pkgs))
        dylib.unlink()
PY

################################################################################
# 5) sanity‑check imports (and guard against cwd shadowing)
################################################################################
banner "sanity‑checking imports …"
export PYTHONPATH=""            # avoid accidentally shadowing std‑lib modules
python - <<'PY'
import importlib.util, pathlib, sys
for m in ("streamlit", "torch", "faiss"):
    spec = importlib.util.find_spec(m)
    rel  = pathlib.Path(spec.origin).relative_to(pathlib.Path.cwd())
    print(f"  {m:10} → {rel}")
print("\nsetup looks \033[1;32mOK ✓\033[0m")
PY

