#!/bin/bash
# ============================================================
# build_submission.sh — Build NMI Submission Package
#
# Runs experiments, compiles main manuscript + supplementary,
# checks for unresolved references, and assembles a clean
# dist/NMI_SUBMISSION_PACKAGE/ directory inside the repo.
#
# Prerequisites:
#   - pdflatex + bibtex (or latexmk)
#   - Python 3.x with numpy, scipy, networkx, matplotlib
#
# Usage:
#   bash scripts/build_submission.sh          # full build
#   bash scripts/build_submission.sh --smoke  # smoke experiments only
# ============================================================

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SMOKE=""
[[ "${1:-}" == "--smoke" ]] && SMOKE="--smoke"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; ERRORS=$((ERRORS + 1)); }
ERRORS=0

echo "============================================="
echo "  NMI Submission Package Builder"
echo "============================================="
echo ""

# ----- 1. Run experiments -----
echo "--- Step 1: Run experiments ---"
cd "$ROOT"
mkdir -p results

echo "[1/4] Logic Maze..."
(cd code/logic_maze && python3 run_logic_maze.py --grid_size 3 --num_runs 2 --num_timesteps 30 && cp results/* "$ROOT/results/") && pass "Logic Maze" || fail "Logic Maze"

echo "[2/4] Bellman Consistency..."
if [[ -n "$SMOKE" ]]; then
    (cd code/safety_gym && python3 train_safety_gym.py --smoke && cp results/* "$ROOT/results/") && pass "Bellman Consistency (smoke)" || fail "Bellman Consistency"
else
    (cd code/safety_gym && python3 train_safety_gym.py --grid_size 8 --num_seeds 10 --episodes 500 && cp results/* "$ROOT/results/") && pass "Bellman Consistency" || fail "Bellman Consistency"
fi

echo "[3/4] Multi-Robot..."
(cd code/multi_robot && python3 run_multi_robot.py --num_robots 5 --num_timesteps 30 --num_runs 2 && cp results/* "$ROOT/results/") && pass "Multi-Robot" || fail "Multi-Robot"

echo "[4/4] Scalability..."
(cd code/scalability && python3 run_scalability.py --max_agents 500 --num_trials 2 && cp results/* "$ROOT/results/") && pass "Scalability" || fail "Scalability"

echo ""

# ----- 2. Copy experiment figures into paper/ so LaTeX can find them -----
echo "--- Step 2: Copy figures to paper/ ---"
cp "$ROOT/results/"*.png "$ROOT/paper/" 2>/dev/null && pass "Figures copied to paper/" || echo "  (no new PNGs to copy)"
echo ""

# ----- 3. Build main manuscript -----
echo "--- Step 3: Compile main manuscript ---"
cd "$ROOT/paper"

# Check for required figures before compiling
MISSING_FIG=0
for fig in figure1_timeseries.png figure2_barchart.png figure4_scalability.png; do
    if [[ ! -f "$fig" ]]; then
        fail "Missing figure: $fig"
        MISSING_FIG=1
    fi
done
if [[ $MISSING_FIG -eq 0 ]]; then pass "All required figures present"; fi

if command -v latexmk &>/dev/null; then
    latexmk -pdf -interaction=nonstopmode main_manuscript.tex && pass "main_manuscript.pdf" || fail "main_manuscript.pdf"
else
    pdflatex -interaction=nonstopmode main_manuscript.tex || true
    bibtex main_manuscript || true
    pdflatex -interaction=nonstopmode main_manuscript.tex || true
    pdflatex -interaction=nonstopmode main_manuscript.tex && pass "main_manuscript.pdf" || fail "main_manuscript.pdf"
fi
echo ""

# ----- 4. Build supplementary -----
echo "--- Step 4: Compile supplementary ---"
SI_DIR="$ROOT/supplementary"
if [[ -d "$SI_DIR" ]]; then
    cd "$SI_DIR"
    if command -v latexmk &>/dev/null; then
        latexmk -pdf -interaction=nonstopmode supplementary_information.tex && pass "supplementary_information.pdf" || fail "supplementary_information.pdf"
    else
        pdflatex -interaction=nonstopmode supplementary_information.tex || true
        bibtex supplementary_information || true
        pdflatex -interaction=nonstopmode supplementary_information.tex || true
        pdflatex -interaction=nonstopmode supplementary_information.tex && pass "supplementary_information.pdf" || fail "supplementary_information.pdf"
    fi
else
    fail "supplementary/ directory not found at $SI_DIR"
fi
echo ""

# ----- 5. Quality gate — unresolved references -----
echo "--- Step 5: Quality gate — unresolved references ---"
cd "$ROOT/paper"
if [[ -f main_manuscript.log ]]; then
    UNDEF=$(grep -c "LaTeX Warning.*undefined" main_manuscript.log 2>/dev/null || echo 0)
    if [[ "$UNDEF" -gt 0 ]]; then
        fail "main_manuscript has $UNDEF undefined reference(s)"
        grep "LaTeX Warning.*undefined" main_manuscript.log
    else
        pass "main_manuscript: 0 undefined references"
    fi
fi

if [[ -f "$SI_DIR/supplementary_information.log" ]]; then
    UNDEF_SI=$(grep -c "LaTeX Warning.*undefined" "$SI_DIR/supplementary_information.log" 2>/dev/null || echo 0)
    if [[ "$UNDEF_SI" -gt 0 ]]; then
        fail "supplementary has $UNDEF_SI undefined reference(s)"
        grep "LaTeX Warning.*undefined" "$SI_DIR/supplementary_information.log"
    else
        pass "supplementary: 0 undefined references"
    fi
fi
echo ""

# ----- 6. Assemble submission package -----
echo "--- Step 6: Assemble dist/NMI_SUBMISSION_PACKAGE ---"
PKG="$ROOT/dist/NMI_SUBMISSION_PACKAGE"
rm -rf "$PKG"
mkdir -p "$PKG/manuscript" "$PKG/supplementary" "$PKG/code" "$PKG/figures"

cp "$ROOT/paper/main_manuscript.pdf" "$PKG/manuscript/" 2>/dev/null && pass "Copied manuscript PDF" || fail "Manuscript PDF missing"
cp "$SI_DIR/supplementary_information.pdf" "$PKG/supplementary/" 2>/dev/null && pass "Copied SI PDF" || fail "SI PDF missing"
cp "$ROOT/results/"* "$PKG/figures/" 2>/dev/null || true
cp -r "$ROOT/code" "$PKG/" 2>/dev/null || true
cp "$ROOT/REPRODUCIBILITY.md" "$PKG/" 2>/dev/null || true
pass "Package assembled at dist/NMI_SUBMISSION_PACKAGE/"

# Create zip
cd "$ROOT/dist"
if command -v zip &>/dev/null; then
    zip -r NMI_SUBMISSION_PACKAGE.zip NMI_SUBMISSION_PACKAGE/ && pass "Created NMI_SUBMISSION_PACKAGE.zip"
elif command -v tar &>/dev/null; then
    tar czf NMI_SUBMISSION_PACKAGE.tar.gz NMI_SUBMISSION_PACKAGE/ && pass "Created NMI_SUBMISSION_PACKAGE.tar.gz"
fi

echo ""

# ----- Summary -----
echo "============================================="
if [[ $ERRORS -eq 0 ]]; then
    echo -e "${GREEN}  ALL GATES PASSED${NC}"
else
    echo -e "${RED}  $ERRORS GATE(S) FAILED${NC}"
fi
echo "============================================="
exit $ERRORS
