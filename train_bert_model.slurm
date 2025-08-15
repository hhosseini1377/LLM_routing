#!/bin/bash

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

#SBATCH --job-name="BERT_train"

#SBATCH --account="punim2662"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --nodes 1
#SBATCH --partition gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=80GB
#SBATCH -A punim2662 
#SBATCH --output=outputs/myjob_%j.out

# Use this email address:
#SBATCH --mail-user=shosseinikas@student.unimelb.edu.au


# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

# The modules to load:
module load GCCcore/11.3.0
module load Python/3.11.3

# Change the home directory for huggingface
export HF_HOME=/data/gpfs/projects/punim2662/.cache/huggingface

# Activating the VirEn
source env/bin/activate

# The job command(s):
python main.py --model_name distilbert  --num_epochs 200 --batch_size 32 --context_window 512 --data_size None --strategy cls
deactivate

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s