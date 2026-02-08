#!/bin/bash
# ============================================================
# Docker: Run ALL experiments + compile LaTeX + validate
# ============================================================
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0

log_pass() { echo -e "  ${GREEN}✓ PASS${NC}: $1"; PASS=$((PASS+1)); }
log_fail() { echo -e "  ${RED}✗ FAIL${NC}: $1"; FAIL=$((FAIL+1)); }

echo "========================================================"
echo " Phronesis Index — Full Reproducibility Pipeline"
echo " $(date)"
echo "========================================================"
echo ""

# ──────────────────────────────────────────────────────────────
# PHASE 1 : Run experiments
# ──────────────────────────────────────────────────────────────
echo "═══ PHASE 1: EXPERIMENTS ═══"
echo ""

# Global results dir
mkdir -p /repo/results

# 1. Logic Maze
echo "[1/4] Logic Maze..."
cd /repo/code/logic_maze
python run_logic_maze.py --grid_size 5 --num_runs 10 --num_timesteps 150
if [ -f results/phi_timeseries.csv ] && [ -f results/figure1_timeseries.png ]; then
    log_pass "Logic Maze — CSV + PNG produced"
    cp results/* /repo/results/
else
    log_fail "Logic Maze — missing output files"
fi
echo ""

# 2. Bellman Consistency / Safety
echo "[2/4] Bellman Consistency (Safety)..."
cd /repo/code/safety_gym
python train_safety_gym.py --grid_size 8 --num_seeds 10 --episodes 500
if [ -f results/training_curves.csv ] && [ -f results/phi_trajectory.csv ]; then
    log_pass "Bellman Consistency — CSVs produced"
    cp results/* /repo/results/
else
    log_fail "Bellman Consistency — missing output files"
fi
if [ -f results/figure2_barchart.png ]; then
    log_pass "Bellman Consistency — PNG produced"
else
    log_fail "Bellman Consistency — figure2_barchart.png missing"
fi
echo ""

# 3. Multi-Robot
echo "[3/4] Multi-Robot Coordination..."
cd /repo/code/multi_robot
python run_multi_robot.py --num_robots 10 --num_timesteps 100 --num_runs 10
if [ -f results/consistency_metrics.csv ] && [ -f results/robot_trajectories.png ]; then
    log_pass "Multi-Robot — CSV + PNG produced"
    cp results/* /repo/results/
else
    log_fail "Multi-Robot — missing output files"
fi
echo ""

# 4. Scalability
echo "[4/4] Scalability Test..."
cd /repo/code/scalability
python run_scalability.py --max_agents 2000 --num_trials 2
if [ -f results/scalability_data.csv ] && [ -f results/figure4_scalability.png ]; then
    log_pass "Scalability — CSV + PNG produced"
    cp results/* /repo/results/
else
    log_fail "Scalability — missing output files"
fi
echo ""

# ──────────────────────────────────────────────────────────────
# PHASE 2 : Copy fresh figures into paper/ for LaTeX
# ──────────────────────────────────────────────────────────────
echo "═══ PHASE 2: COPY FIGURES ═══"
cp /repo/results/*.png /repo/paper/ 2>/dev/null && log_pass "Copied experiment figures to paper/" || echo "  (no new PNGs)"
echo ""

# ──────────────────────────────────────────────────────────────
# PHASE 3 : Compile LaTeX
# ──────────────────────────────────────────────────────────────
echo "═══ PHASE 3: LaTeX COMPILATION ═══"
echo ""

# Main manuscript
echo "[LaTeX] Compiling main_manuscript.tex ..."
cd /repo/paper
pdflatex -interaction=nonstopmode main_manuscript.tex > /tmp/latex1.log 2>&1 || true
bibtex main_manuscript > /tmp/bibtex1.log 2>&1 || true
pdflatex -interaction=nonstopmode main_manuscript.tex > /tmp/latex2.log 2>&1 || true
pdflatex -interaction=nonstopmode main_manuscript.tex > /tmp/latex3.log 2>&1 || true

if [ -f main_manuscript.pdf ]; then
    log_pass "main_manuscript.pdf compiled"
    # Check for undefined references (match actual LaTeX warnings only)
    UNDEF=$(grep -cE "Warning.*undefined" main_manuscript.log 2>/dev/null) || true
    UNDEF=${UNDEF:-0}
    if [ "$UNDEF" -eq 0 ]; then
        log_pass "main_manuscript — 0 undefined references"
    else
        log_fail "main_manuscript — $UNDEF undefined reference warnings"
        grep -E "Warning.*undefined" main_manuscript.log || true
    fi
else
    log_fail "main_manuscript.pdf not produced"
    echo "--- Last 50 lines of LaTeX log ---"
    tail -50 /tmp/latex3.log
fi
echo ""

# Supplementary
echo "[LaTeX] Compiling supplementary_information.tex ..."
cd /repo/supplementary
if [ -f supplementary_information.tex ]; then
    pdflatex -interaction=nonstopmode supplementary_information.tex > /tmp/slatex1.log 2>&1 || true
    bibtex supplementary_information > /tmp/sbibtex1.log 2>&1 || true
    pdflatex -interaction=nonstopmode supplementary_information.tex > /tmp/slatex2.log 2>&1 || true
    pdflatex -interaction=nonstopmode supplementary_information.tex > /tmp/slatex3.log 2>&1 || true

    if [ -f supplementary_information.pdf ]; then
        log_pass "supplementary_information.pdf compiled"
        UNDEF=$(grep -cE "Warning.*undefined" supplementary_information.log 2>/dev/null) || true
        UNDEF=${UNDEF:-0}
        if [ "$UNDEF" -eq 0 ]; then
            log_pass "supplementary — 0 undefined references"
        else
            log_fail "supplementary — $UNDEF undefined reference warnings"
            grep -E "Warning.*undefined" supplementary_information.log || true
        fi
    else
        log_fail "supplementary_information.pdf not produced"
        tail -50 /tmp/slatex3.log
    fi
else
    echo "  (no supplementary_information.tex found — skipping)"
fi
echo ""

# ──────────────────────────────────────────────────────────────
# PHASE 4 : Summary
# ──────────────────────────────────────────────────────────────
echo "═══ PHASE 4: SUMMARY ═══"
echo ""
echo "  Results directory contents:"
ls -lh /repo/results/ 2>/dev/null || echo "  (no results)"
echo ""
echo "──────────────────────────────────────────"
echo -e "  ${GREEN}PASSED${NC}: $PASS"
echo -e "  ${RED}FAILED${NC}: $FAIL"
echo "──────────────────────────────────────────"

if [ $FAIL -gt 0 ]; then
    echo -e "\n  ${RED}SOME CHECKS FAILED — see above for details${NC}"
    exit 1
else
    echo -e "\n  ${GREEN}ALL CHECKS PASSED — submission package is valid!${NC}"
    exit 0
fi
