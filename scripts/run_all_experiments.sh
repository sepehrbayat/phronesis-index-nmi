#!/bin/bash

# Run All Experiments for Phronesis Index Paper
# This script reproduces all results from the paper

set -e  # Exit on error

echo "=========================================="
echo "Phronesis Index - Full Experiment Suite"
echo "=========================================="
echo ""

# Create results directory
mkdir -p results

# 1. Logic Maze Experiment
echo "[1/4] Running Logic Maze experiment..."
cd code/logic_maze
python3 run_logic_maze.py --grid_size 5 --num_runs 10 --num_timesteps 150
cp results/* ../../results/
cd ../..
echo "✓ Logic Maze complete"
echo ""

# 2. Safety / Bellman Consistency Experiment
echo "[2/4] Running Bellman Consistency experiment..."
cd code/safety_gym
python3 train_safety_gym.py --grid_size 8 --num_seeds 10 --episodes 500
cp results/* ../../results/
cd ../..
echo "✓ Bellman Consistency complete"
echo ""

# 3. Multi-Robot Experiment
echo "[3/4] Running Multi-Robot experiment..."
cd code/multi_robot
python3 run_multi_robot.py --num_robots 10 --num_timesteps 100 --num_runs 10
cp results/* ../../results/
cd ../..
echo "✓ Multi-Robot complete"
echo ""

# 4. Scalability Test
echo "[4/4] Running Scalability test..."
cd code/scalability
python3 run_scalability.py --max_agents 50000 --num_trials 5
cp results/* ../../results/
cd ../..
echo "✓ Scalability complete"
echo ""

echo "=========================================="
echo "All experiments completed successfully!"
echo "Results saved to: results/"
echo "=========================================="
