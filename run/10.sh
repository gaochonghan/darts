#!/bin/bash
#SBATCH --partition=dell
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:1
#SBATCH -J 2a10
#SBATCH -o log/job-%j.log
#SBATCH -e log/job-%j.err
# shellcheck disable=SC2046
echo Time is `date`
echo Directory is $PWD
echo This job runs on the following nodes:
echo "$SLURM_JOB_NODELIST"
eval "$(conda shell.bash hook)"
source /home/LAB/gaoch/miniconda2/bin/activate base
# conda --version
# which python
conda activate gchmini
cd ..
echo Python:
which python
export NCCL_IB_DISABLE=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
export MKL_THREADING_LAYER=GNU
export CUDA_HOME=/usr/local/cuda-10.2
# sugon does not support infiniband
srun python search.py --name cifar10 --dataset cifar10 --epochs 30