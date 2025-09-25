#!/bin/bash

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

#SBATCH --job-name="BERT_train"

#SBATCH --account="punim2662"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --nodes 1
#SBATCH --partition gpu-h100
#SBATCH --gres=gpu:3
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=80GB
#SBATCH -A punim2662 
#SBATCH --output=outputs/train_mistral_cpx_%j.out

# Use this email address:
#SBATCH --mail-user=shosseinikas@student.unimelb.edu.au


# The modules to load:

# Activating the VirEn
source env/bin/activate

# The job command(s):
# python main.py --model_name distilbert  --num_epochs 5 --batch_size 32 --context_window 512 --data_size None --strategy cls
python3 -m cpx_model.cpxmistral.main  --num_epochs 10 --batch_size 24 --context_window 8192 --data_size 'None' --evaluation_size 'None'

deactivate

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s