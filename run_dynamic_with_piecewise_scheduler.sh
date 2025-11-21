#!/bin/bash

# Dynamic resampling run with Piecewise Constant + Tail Decay scheduler
# This scheduler keeps LR constant for most of each 800-epoch block, then gently decays at the end

nohup bash -c '
CUDA_VISIBLE_DEVICES=0 /home/kha98/Desktop/.venv/bin/python /home/kha98/Desktop/rs-imle/train.py \
    --hps fewshot \
    --save_dir /home/kha98/Desktop/rs-imle/runs/resample_teacher/resample-dynamic-800-piecewise/ \
    --data_root /home/kha98/Desktop/rs-imle/dataset/ffhq/20_steps_seed_9 \
    --condition_path /home/kha98/Desktop/rs-imle/dataset/ffhq/x0_20_seed_9.pt \
    --model_type unet \
    --force_factor 20 \
    --imle_force_resample 10 \
    --teacher_force_resample 800 \
    --use_comet True \
    --latent_dim 4096 \
    --comet_name teacher-resample-dynamic-800-piecewise-scheduler \
    --comet_api_key esx5iX53IbgtEtr4Zj1tkxpYB \
    --use_teacher_resample True \
    --teacher_checkpoint_path /home/kha98/Desktop/flow-model-chirag/output_flow/flow-ffhq-debugfm/fm_cifar10_weights_step_84000.pt \
    --teacher_resample_steps 20 \
    --teacher_num_samples 100 \
    --imle_db_size 2000 \
    --imle_batch 50 \
    --dynamic_scheduler_type piecewise \
    --lr_min 1e-5 \
    --lr_tail_epochs 200 \
    --lr_tail_decay 0.99
' > /home/kha98/Desktop/rs-imle/runs/resample_teacher/resample-dynamic-800-piecewise/train.log 2>&1 &

echo "Started training with Piecewise scheduler (PID: $!)"
echo "Log file: /home/kha98/Desktop/rs-imle/runs/resample_teacher/resample-dynamic-800-piecewise/train.log"
echo "Monitor with: tail -f /home/kha98/Desktop/rs-imle/runs/resample_teacher/resample-dynamic-800-piecewise/train.log"

