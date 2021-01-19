#!/bin/bash -l
#SBATCH --time=00:15:00
#SBATCH -C gpu

#SBATCH --account=m1759

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=10
#SBATCH -o sout/%j.out

#SBATCH --image=nersc/pytorch:ngc-20.08-v0
#SBATCH --volume="/dev/infiniband:/sys/class/infiniband_verbs"

ln -s /global/cscratch1/sd/mustafa/SDSS/data_20200901/valid.h5 /tmp

srun shifter --env PYTHONUSERBASE=$HOME/cori/pytorch_ngc-20.08-tf2-v0_env \
             python eval_specz.py --config=baseline
