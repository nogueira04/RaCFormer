#!/bin/bash
#SBATCH --job-name=racformer_nightaug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

CONFIG="configs/racformer_r50_nuimg_704x256_f8_nightaug.py"
WORK_DIR="outputs/racformer_r50_nuimg_704x256_f8_nightaug"

mkdir -p slurm_logs

source /srv/nfs/shared/gnmp/miniconda3/etc/profile.d/conda.sh
conda activate racformerfix

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=^lo,docker,cali
export NCCL_DEBUG=INFO
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_IB_DISABLE=1

echo "Master node: $MASTER_ADDR"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Job ID: $SLURM_JOB_ID"

srun python3 train.py --config $CONFIG
