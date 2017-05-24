#!/bin/bash

# Name of the job: PBS can read the command after #, which shell ignores them
#PBS -N train_3D

# Job asks for 1 node (computer) and 1 processing number per node (total 4 GPUs each node)
# Now we want to use 2 GPUs, then we set ppn=2.
#PBS -l nodes=1:ppn=1

# Redirect standard error to standard output
#PBS -j oe

### Copy the completed output to the specified folder
#PBS -o /share/data/joblog/${PBS_JOBID}.OU


#Remap the free gpus to make the IDs always start from 0
source /share/data/script/util/remap_free_gpus.sh

# Enter the job's working directory.
[ "$PBS_O_WORKDIR" != "" ] && cd $PBS_O_WORKDIR

#Our own shell commands, copy the above lines before your real commands
./experiments/NYU/train_3D.sh
