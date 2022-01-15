#!/bin/bash

# Example of running python script with a job array

#SBATCH -N 1                     # Number of nodes
#SBATCH --ntasks-per-node 1
#SBATCH -J DMTcamus              # Job name
#SBATCH -t 14-0:00               # Time limit. Here I set maximum 2 weeks
#SBATCH --array=0-55             # How many tasks in the array
#SBATCH -c 2                     # Number of CPU core per task
#SBATCH -o run_%j_%a.out

# Load software

# Run python script with a command line argument
srun python3 wrapper_run.py $SLURM_ARRAY_TASK_ID
