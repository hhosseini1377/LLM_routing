#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpu-h100
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH -A punim2662 
#SBATCH --output=outputs/myjob_%j.out

module load GCCcore/11.3.0
module load Python/3.11.3

source env/bin/activate

jupyter notebook --no-browser --port=8890 --ip=127.0.0.1

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s