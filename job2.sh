#!/bin/bash
#SBATCH --job-name=missing_imputation_test    # Job name
#SBATCH --mem=15G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5                 # Number of CPU cores (adjust as needed)
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:0:0    
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=yjkweon24@berkeley.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=methylation_%x_%j.out        # Output file (%j is job ID)
#SBATCH --error=methylation_%x_%j.err         # Error file
#SBATCH --account=def-rtc

module load cuda/12.2

# Activate your virtual environment (if you have one)
source ~/jinenv/bin/activate

# Change to project directory
cd /home/yjkweon2/projects/def-rtc/yjkweon2/Missing_imputation

python test2.py
