#!/bin/bash
#SBATCH --partition=dell
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:1
#SBATCH -J 2a100
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
cd ../..
echo Python:
which python
export NCCL_IB_DISABLE=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
export MKL_THREADING_LAYER=GNU
export CUDA_HOME=/usr/local/cuda-10.2
# sugon does not support infiniband
srun python ./augment.py --name 2a100 --dataset cifar100 --batch_size 128 --epochs 2000 --genotype \
"Genotype(normal=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)], [('max_pool_3x3', 0), ('sep_conv_5x5', 2)], [('max_pool_3x3', 0), ('max_pool_3x3', 1)], [('sep_conv_5x5', 4), ('max_pool_3x3', 0)]], normal_concat=range(2, 6), reduce=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)], [('max_pool_3x3', 2), ('max_pool_3x3', 0)], [('max_pool_3x3', 0), ('max_pool_3x3', 2)], [('max_pool_3x3', 0), ('max_pool_3x3', 2)]], reduce_concat=range(2, 6))"