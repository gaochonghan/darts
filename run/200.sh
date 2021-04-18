#!/bin/bash
#SBATCH --partition=dell
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:1
#SBATCH -J e2h3-a200
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
srun python ./augment.py --name a200 --dataset imagenet --batch_size 196 --epochs 2000 --genotype \
"genotype = Genotype(normal=[[('dil_conv_3x3', 0), ('dil_conv_5x5', 1)], [('sep_conv_3x3', 0), ('dil_conv_5x5', 2)], [('dil_conv_3x3', 3), ('sep_conv_3x3', 1)], [('dil_conv_3x3', 0), ('sep_conv_3x3', 1)]], normal_concat=range(2, 6), reduce=[[('dil_conv_5x5', 0), ('skip_connect', 1)], [('dil_conv_5x5', 2), ('dil_conv_5x5', 1)], [('dil_conv_5x5', 3), ('skip_connect', 2)], [('dil_conv_5x5', 4), ('dil_conv_5x5', 3)]], reduce_concat=range(2, 6))"
# e2h2 "Genotype(normal=[[('sep_conv_5x5', 1), ('sep_conv_3x3', 0)], [('dil_conv_5x5', 2), ('sep_conv_5x5', 1)], [('dil_conv_5x5', 2), ('dil_conv_3x3', 0)], [('dil_conv_3x3', 4), ('dil_conv_5x5', 2)]], normal_concat=range(2, 6), reduce=[[('dil_conv_5x5', 0), ('sep_conv_3x3', 1)], [('max_pool_3x3', 1), ('dil_conv_5x5', 2)], [('skip_connect', 3), ('max_pool_3x3', 1)], [('skip_connect', 4), ('skip_connect', 2)]], reduce_concat=range(2, 6))"
# e2h1 "Genotype(normal=[[('dil_conv_5x5', 0), ('dil_conv_5x5', 1)], [('sep_conv_5x5', 0), ('dil_conv_3x3', 2)], [('dil_conv_3x3', 0), ('dil_conv_3x3', 3)], [('dil_conv_5x5', 0), ('dil_conv_3x3', 1)]], normal_concat=range(2, 6), reduce=[[('max_pool_3x3', 0), ('skip_connect', 1)], [('avg_pool_3x3', 0), ('skip_connect', 2)], [('avg_pool_3x3', 0), ('skip_connect', 2)], [('skip_connect', 2), ('avg_pool_3x3', 0)]], reduce_concat=range(2, 6))"