#!/bin/bash
# Evaluation script for generating images from a trained checkpoint
# This script submits a SLURM job to run train.py in eval-only mode

cd /data/user_data/kkuo2/flow_matching/examples/image

# Configuration - modify these parameters as needed
DATASET="cat_faces,celeba"  # Must match the training data path
FID_SAMPLES=16  # Number of samples to generate for FID computation
SAVE_FID_SAMPLES=true  # Set to true to save all generated images individually

# SLURM job parameters
NODES=1
NGPUS=1
PARTITION="general"
TIMEOUT=120  # Time limit in minutes
IMAGE_SIZE=64  # Must match training configuration
BATCH_SIZE=4  # Batch size for generation
MAX_EXAMPLES=4000

user=kkuo2
CHECKPOINT_DIR="checkpoint/${user}/experiments/cat_faces_celeba_${MAX_EXAMPLES}_${IMAGE_SIZE}"  # Directory containing checkpoint and args.json
CHECKPOINT_NAME="checkpoint-699.pth"  # Name of the checkpoint file to evaluate

# Run evaluation using submitit
python submitit_train.py \
  --dataset="${DATASET}" \
  --image_size="${IMAGE_SIZE}" \
  --batch_size="${BATCH_SIZE}" \
  --nodes="${NODES}" \
  --ngpus="${NGPUS}" \
  --partition="${PARTITION}" \
  --timeout="${TIMEOUT}" \
  --job_dir="${CHECKPOINT_DIR}/eval_%j" \
  --resume="${CHECKPOINT_DIR}/${CHECKPOINT_NAME}" \
  --eval_only \
  --fid_samples="${FID_SAMPLES}" \
  --swap_steps="swap_schedules/50_50.txt" \
  $([ "${SAVE_FID_SAMPLES}" = "true" ] && echo "--save_fid_samples")
