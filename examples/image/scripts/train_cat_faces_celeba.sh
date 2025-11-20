#!/bin/bash
cd /data/user_data/kkuo2/flow_matching/examples/image
export PYTHONPATH=$PYTHONPATH:/data/user_data/kkuo2/flow_matching
base_folder=/data/user_data/kkuo2/flow_matching/examples/image
image_size=128
max_examples=4000
user=kkuo2
# image_size=64
# max_examples=1000
python $base_folder/submitit_train.py \
  --data_path=cat_faces,celeba \
  --image_size=${image_size} \
  --batch_size=4 \
  --nodes=1 \
  --ngpus=4 \
  --partition=general \
  --timeout=1440 \
  --class_drop_prob=0.1 \
  --cfg_scale=1.0 \
  --eval_frequency=100 \
  --epochs=1800 \
  --accum_iter=1 \
  --num_classes=2 \
  --max_examples_per_dataset=${max_examples} \
  --fid_samples=32 \
  --captions=false \
  --shared_dir=$base_folder/checkpoint \
  --job_dir=$base_folder/checkpoint/${user}/experiments/cat_faces_celeba_${max_examples}_${image_size} \
  # --resume=$base_folder/checkpoint/${user}/experiments/cat_faces_celeba_${max_examples}_${image_size}/checkpoint.pth