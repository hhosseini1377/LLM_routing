#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=0-12:00:00

# Activate your environment
# source ~/envs/llm-routing/bin/activate

# Run training
python tempt.py
