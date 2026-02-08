#!/bin/bash
# ============================================================
# build_submission.sh — Build NMI Submission Package
#
# Compiles the main manuscript and supplementary PDFs, copies
# them into NMI_SUBMISSION_PACKAGE/, and checks for unresolved
# LaTeX references (??) and missing citations ([?]).
#
# Prerequisites:
#   - latexmk or pdflatex + bibtex
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

# ----- 2. Build main manuscript -----
echo "--- Step 2: Compile main manuscript ---"
cd "$ROOT/paper"
if command -v latexmk &>/dev/null; then
    latexmk -pdf -interaction=nonstopmode main_manuscript.tex && pass "main_manuscript.pdf" || fail "main_manuscript.pdf"
else
    pdflatex -interaction=nonstopmode main_manuscript.tex
    bibtex main_manuscript || true
    pdflatex -interaction=nonstopmode main_manuscript.tex
    pdflatex -interaction=nonstopmode main_manuscript.tex && pass "main_manuscript.pdf" || fail "main_manuscript.pdf"
fi

# ----- 3. Build supplementary -----
echo "--- Step 3: Compile supplementary ---"
SI_DIR="$(dirname "$ROOT")/supplementary"
if [[ -d "$SI_DIR" ]]; then
    cd "$SI_DIR"
    if command -v latexmk &>/dev/null; then
        latexmk -pdf -interaction=nonstopmode supplementary_information.tex && pass "supplementary_information.pdf" || fail "supplementary_information.pdf"
    else
        pdflatex -interaction=nonstopmode supplementary_information.tex
        bibtex supplementary_information || true
        pdflatex -interaction=nonstopmode supplementary_information.tex
        pdflatex -interaction=nonstopmode supplementary_information.tex && pass "supplementary_information.pdf" || fail "supplementary_information.pdf"
    fi
else
    fail "supplementary directory not found at $SI_DIR"
fi

echo ""

# ----- 4. Check for unresolved references -----
echo "--- Step 4: Quality gate — unresolved references ---"
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

# ----- 5. Copy into submission package -----
echo "--- Step 5: Assemble NMI_SUBMISSION_PACKAGE ---"
PKG="$ROOT/../NMI_SUBMISSION_PACKAGE"
mkdir -p "$PKG/manuscript" "$PKG/supplementary" "$PKG/code" "$PKG/figures"

cp "$ROOT/paper/main_manuscript.pdf" "$PKG/manuscript/" 2>/dev/null && pass "Copied manuscript PDF" || fail "Manuscript PDF missing"
cp "$SI_DIR/supplementary_information.pdf" "$PKG/supplementary/" 2>/dev/null && pass "Copied SI PDF" || fail "SI PDF missing"
cp "$ROOT/results/"* "$PKG/figures/" 2>/dev/null || true
cp -r "$ROOT/code" "$PKG/" 2>/dev/null || true
pass "Package assembled at $PKG"

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
