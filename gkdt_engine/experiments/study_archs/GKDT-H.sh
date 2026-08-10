#!/bin/bash
#SBATCH --job-name=gkd           # create a short name for your job
#SBATCH --nodes=2                # node count
#SBATCH --gpus-per-node=8        # number of GPUs per node(only valid under large/normal partition)
#SBATCH --cpus-per-task=224      # number of CPUs (28, 56, 112, 224 for 1, 2, 4, 8 GPUs)
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=yours        # partition(preempt/large/normal/cpu) where you submit
#SBATCH --account=yours          # only require for multiple projects

module purge  # clear environment modules inherited from submission
module load slurm cuda11.8/blas/11.8.0 cuda11.8/fft/11.8.0 cuda11.8/toolkit/11.8.0  # need Pytorch with at least cuda11.8 
source /your/path/to/anaconda3/bin/activate gkd_env

cd '/your/path/to/General-Keypoint-Detection'  # modify this to your path of General-Keypoint-Detection
echo $(pwd)

echo "========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "========================================="

# Set up distributed training parameters
# MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
# MASTER_PORT=${MASTER_PORT:-29502}
MASTER_ADDR=$(scontrol show hostnames $SLURM_STEP_NODELIST | head -n 1)
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo "========================================="
echo "Distributed Training Configuration:"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
echo "Total GPUs: $((SLURM_NNODES * SLURM_GPUS_PER_NODE))"
echo "========================================="

# Launch distributed training
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    main_gkd.py --cfg_file experiments/configs/gkd.yaml \
    OUTPUT_DIR output/GKDT-H \
    MODEL.ENCODER.DINOv3.VISUAL_ENCODER dinov3_vith16plus \
    MODEL.DETECTION_HEAD.IM_FEAT_UPSAMPLER.TYPE 'bilinear' \
    TRAIN.MIX_MODAL_TRAINING.TYPE 'episode' \
    TRAIN.MIX_MODAL_TRAINING.RANGE 'tvm' \
    DATASET.SAMPLING_STRATEGY.SHUFFLE True \
    DATASET.SAMPLING_STRATEGY.DROP_LAST True \
    DATASET.SAMPLING_STRATEGY.TEST 'importance' \
    TRAIN.NUM_ROLL_OUT 1 \
    TRAIN.LR 0.0001 \
    TB_WRITER True \
    RESUME False