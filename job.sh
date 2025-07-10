#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=0-12:00:00
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=24G

# Activate your environment
# source ~/envs/llm-routing/bin/activate

# Run training
module purge
module load GCCcore/11.3.0
module load Python/3.10.4
source ./vnv/bin/activate

python train_BERT.py --model deberta