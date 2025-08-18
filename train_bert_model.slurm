#!/bin/bash

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

#SBATCH --job-name="BERT_train"

#SBATCH --account="punim2662"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --nodes 1
#SBATCH --partition gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8  
#SBATCH --mem=80GB
#SBATCH -A punim2662 
#SBATCH --output=outputs/myjob_%j.out

# Use this email address:
#SBATCH --mail-user=shosseinikas@student.unimelb.edu.au


# The modules to load:
module load GCCcore/11.3.0
module load Python/3.11.3

# Activating the VirEn
source env/bin/activate

# The job command(s):
python main.py --model_name distilbert  --num_epochs 5 --batch_size 32 --context_window 512 --data_size None --strategy cls
deactivate

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s