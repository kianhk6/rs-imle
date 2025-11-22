#!/bin/bash

# RS-IMLE Training with Teacher-Based Dynamic Resampling
# ========================================================
# 
# This script trains RS-IMLE where:
# 1. Initial dataset is generated from teacher model (not loaded from disk)
# 2. During resampling, NEW data is generated from teacher with NEW conditions
# 3. Student (IMLE) learns to match the dynamically refreshed teacher outputs
#
# Key differences from standard RS-IMLE:
# - No fixed dataset: data is continuously regenerated
# - Conditions are updated alongside data
# - --data_root is ignored (data comes from teacher)

nohup bash -c '
CUDA_VISIBLE_DEVICES=1 /home/kha98/Desktop/rs-imle/train.py \
    --hps fewshot \
    --save_dir /home/kha98/Desktop/rs-imle/runs/resample_teacher/resample-dynamic-800/ \
    --data_root /home/kha98/Desktop/rs-imle/dataset/ffhq/20_steps_seed_9 \
    --condition_path /home/kha98/Desktop/rs-imle/dataset/ffhq/x0_20_seed_9.pt \
    --model_type unet \
    --force_factor 20 \
    --imle_force_resample 10 \
    --teacher_force_resample 800 \
    --use_comet True \
    --latent_dim 4096 \
    --comet_name teacher-resample-dynamic-data-every-800 \
    --comet_api_key esx5iX53IbgtEtr4Zj1tkxpYB \
    --use_teacher_resample True \
    --teacher_checkpoint_path /home/kha98/Desktop/flow-model-chirag/output_flow/flow-ffhq-debugfm/fm_cifar10_weights_step_84000.pt \
    --teacher_resample_steps 20 \
    --teacher_num_samples 4000 \
    --imle_db_size 2000 \
    --imle_batch 50 
    
' > /home/kha98/Desktop/rs-imle/runs/resample_teacher/resample-dynamic-800/train.log 2>&1 &

echo "Training started in background with teacher-based dynamic resampling!"
echo "Initial dataset: /home/kha98/Desktop/rs-imle/dataset/ffhq/20_steps_seed_9"
echo "Initial conditions: /home/kha98/Desktop/rs-imle/dataset/ffhq/x0_20_seed_9.pt"
echo "During resampling: 4000 NEW samples will be generated from teacher"
echo "Logs: /home/kha98/Desktop/rs-imle/runs/teacher-resample-dynamic/train.log"
echo "To monitor: tail -f /home/kha98/Desktop/rs-imle/runs/teacher-resample-dynamic/train.log"
echo "To check process: ps aux | grep train.py"

nohup bash -c '
CUDA_VISIBLE_DEVICES=0 /home/kha98/Desktop/.venv/bin/python /home/kha98/Desktop/rs-imle/train.py \
    --hps fewshot \
    --save_dir /home/kha98/Desktop/rs-imle/runs/resample_teacher/resample-dynamic-2000/ \
    --data_root /home/kha98/Desktop/rs-imle/dataset/ffhq/20_steps_seed_9 \
    --condition_path /home/kha98/Desktop/rs-imle/dataset/ffhq/x0_20_seed_9.pt \
    --model_type unet \
    --force_factor 20 \
    --imle_force_resample 10 \
    --teacher_force_resample 2000 \
    --use_comet True \
    --latent_dim 4096 \
    --comet_name teacher-resample-dynamic-data-every-2000 \
    --comet_api_key esx5iX53IbgtEtr4Zj1tkxpYB \
    --use_teacher_resample True \
    --teacher_checkpoint_path /home/kha98/Desktop/flow-model-chirag/output_flow/flow-ffhq-debugfm/fm_cifar10_weights_step_84000.pt \
    --teacher_resample_steps 20 \
    --teacher_num_samples 100 \
    --imle_db_size 2000 \
    --imle_batch 50 \
    --eps_radius 0.06
' > /home/kha98/Desktop/rs-imle/runs/resample_teacher/resample-dynamic-2000/train.log 2>&1 &

#   --teacher_force_resample 800 -> does regular resampling at 800 epochs
# TEACHER MODE: When using teacher_generate_initial_data=True, you can:
# 1. Keep data_root and condition_path (they will be ignored for training, only condition_path size matters)
# 2. OR remove them and use fid_real_dir to specify FID reference images
#
# Current setup: data_root is used as FID fallback (if fid_real_dir not set)
# Recommended: Add --fid_real_dir /path/to/real/images and remove data_root/condition_path

nohup bash -c '
CUDA_VISIBLE_DEVICES=0 python /home/kha98/Desktop/rs-imle/train.py \
    --hps fewshot \
    --save_dir /home/kha98/Desktop/rs-imle/runs/resample_teacher/resample-scheduled/ \
    --fid_real_dir /home/kha98/Desktop/rs-imle/teacher/clean/img \
    --model_type unet \
    --force_factor 20 \
    --imle_force_resample 10 \
    --use_comet True \
    --latent_dim 4096 \
    --comet_name teacher-resample-scheduled \
    --comet_api_key esx5iX53IbgtEtr4Zj1tkxpYB \
    --use_teacher_resample True \
    --teacher_checkpoint_path /home/kha98/Desktop/rs-imle/teacher/fm_cifar10_weights_step_84000.pt \
    --teacher_resample_steps 20 \
    --teacher_num_samples 4000 \
    --imle_db_size 2000 \
    --imle_batch 50 \
    --teacher_force_resample 20 \
    --subset_len -1 \
    --lr 0.0002 \
    --n_batch 12 \
    --teacher_generate_initial_data True
' > /home/kha98/Desktop/rs-imle/runs/resample_teacher/resample-scheduled/train.log 2>&1 &

echo "Training started with LIST-BASED scheduling!"
echo "Initial data: Load from disk (--teacher_generate_initial_data False)"
echo "Schedule:"
echo "  every_n_epochs_resample_data=[800, 200, 100, 50]"
echo "  change_schedule_of_data_resampling_every_n_epoch=[800, 2000, 6000]"
echo ""
echo "Breakdown:"
echo "  Phase 1 (Epoch 0-800): Resample every 800 epochs → 1 resample at epoch 800"
echo "  Phase 2 (Epoch 800-2000): Resample every 200 epochs → ~6 resamples"
echo "  Phase 3 (Epoch 2000-6000): Resample every 100 epochs → ~40 resamples"
echo "  Phase 4 (Epoch 6000-15000): Resample every 50 epochs → ~180 resamples"
echo "  Total: ~227 resamples over 15,000 epochs"
echo ""
echo "Logs: /home/kha98/Desktop/rs-imle/runs/resample_teacher/resample-scheduled/train.log"

nohup bash -c '
CUDA_VISIBLE_DEVICES=0 python /home/kha98/Desktop/rs-imle/train.py \
    --hps fewshot \
    --save_dir /home/kha98/Desktop/rs-imle/runs/resample_teacher/resample-scheduled-1/ \
    --fid_real_dir /home/kha98/Desktop/rs-imle/teacher/clean/img \
    --model_type unet \
    --force_factor 20 \
    --imle_force_resample 10 \
    --use_comet True \
    --latent_dim 4096 \
    --comet_name teacher-resample-scheduled-1 \
    --comet_api_key esx5iX53IbgtEtr4Zj1tkxpYB \
    --use_teacher_resample True \
    --teacher_checkpoint_path /home/kha98/Desktop/rs-imle/teacher/fm_cifar10_weights_step_84000.pt \
    --teacher_resample_steps 20 \
    --teacher_num_samples 4000 \
    --imle_db_size 2000 \
    --imle_batch 50 \
    --teacher_force_resample 20 \
    --subset_len -1 \
    --lr 0.0002 \
    --n_batch 4 \
    --teacher_generate_initial_data True
' > /home/kha98/Desktop/rs-imle/runs/resample_teacher/resample-scheduled-1/train.log 2>&1 &

















nohup bash -c '
CUDA_VISIBLE_DEVICES=0 python /home/kha98/Desktop/rs-imle/train.py \
    --hps fewshot \
    --save_dir /home/kha98/Desktop/rs-imle/runs/resample_teacher/resample-scheduled-lr-4-resample-20/ \
    --fid_real_dir /home/kha98/Desktop/rs-imle/teacher/clean/img \
    --model_type unet \
    --force_factor 20 \
    --imle_force_resample 10 \
    --use_comet True \
    --latent_dim 4096 \
    --comet_name resample-scheduled-lr-4-resample-20 \
    --comet_api_key esx5iX53IbgtEtr4Zj1tkxpYB \
    --use_teacher_resample True \
    --teacher_checkpoint_path /home/kha98/Desktop/rs-imle/teacher/fm_cifar10_weights_step_84000.pt \
    --teacher_resample_steps 20 \
    --teacher_num_samples 4000 \
    --imle_db_size 2000 \
    --imle_batch 50 \
    --teacher_force_resample 20 \
    --subset_len -1 \
    --lr 0.0004 \
    --n_batch 12 \
    --teacher_generate_initial_data True
' > /home/kha98/Desktop/rs-imle/runs/resample_teacher/resample-scheduled-lr-4-resample-20/train.log 2>&1 &


nohup bash -c '
CUDA_VISIBLE_DEVICES=1 python /home/kha98/Desktop/rs-imle/train.py \
    --hps fewshot \
    --save_dir /home/kha98/Desktop/rs-imle/runs/resample_teacher/resample-scheduled-lr-4-resample-40/ \
    --fid_real_dir /home/kha98/Desktop/rs-imle/teacher/clean/img \
    --model_type unet \
    --force_factor 20 \
    --imle_force_resample 10 \
    --use_comet True \
    --latent_dim 4096 \
    --comet_name resample-scheduled-lr-4-resample-40 \
    --comet_api_key esx5iX53IbgtEtr4Zj1tkxpYB \
    --use_teacher_resample True \
    --teacher_checkpoint_path /home/kha98/Desktop/rs-imle/teacher/fm_cifar10_weights_step_84000.pt \
    --teacher_resample_steps 20 \
    --teacher_num_samples 4000 \
    --imle_db_size 2000 \
    --imle_batch 50 \
    --teacher_force_resample 40 \
    --subset_len -1 \
    --lr 0.0004 \
    --n_batch 12 \
    --teacher_generate_initial_data True
' > /home/kha98/Desktop/rs-imle/runs/resample_teacher/resample-scheduled-lr-4-resample-40/train.log 2>&1 &

nohup bash -c '
CUDA_VISIBLE_DEVICES=2 python /home/kha98/Desktop/rs-imle/train.py \
    --hps fewshot \
    --save_dir /home/kha98/Desktop/rs-imle/runs/resample_teacher/resample-scheduled-lr-2-resample-40/ \
    --fid_real_dir /home/kha98/Desktop/rs-imle/teacher/clean/img \
    --model_type unet \
    --force_factor 20 \
    --imle_force_resample 10 \
    --use_comet True \
    --latent_dim 4096 \
    --comet_name resample-scheduled-lr-2-resample-40 \
    --comet_api_key esx5iX53IbgtEtr4Zj1tkxpYB \
    --use_teacher_resample True \
    --teacher_checkpoint_path /home/kha98/Desktop/rs-imle/teacher/fm_cifar10_weights_step_84000.pt \
    --teacher_resample_steps 20 \
    --teacher_num_samples 4000 \
    --imle_db_size 2000 \
    --imle_batch 50 \
    --teacher_force_resample 40 \
    --subset_len -1 \
    --lr 0.0002 \
    --n_batch 12 \
    --teacher_generate_initial_data True
' > /home/kha98/Desktop/rs-imle/runs/resample_teacher/resample-scheduled-lr-2-resample-40/train.log 2>&1 &


nohup bash -c '
CUDA_VISIBLE_DEVICES=3 python /home/kha98/Desktop/rs-imle/train.py \
    --hps fewshot \
    --save_dir /home/kha98/Desktop/rs-imle/runs/resample_teacher/resample-scheduled-lr-2-resample-20/ \
    --fid_real_dir /home/kha98/Desktop/rs-imle/teacher/clean/img \
    --model_type unet \
    --force_factor 20 \
    --imle_force_resample 10 \
    --use_comet True \
    --latent_dim 4096 \
    --comet_name resample-scheduled-lr-2-resample-20 \
    --comet_api_key esx5iX53IbgtEtr4Zj1tkxpYB \
    --use_teacher_resample True \
    --teacher_checkpoint_path /home/kha98/Desktop/rs-imle/teacher/fm_cifar10_weights_step_84000.pt \
    --teacher_resample_steps 20 \
    --teacher_num_samples 4000 \
    --imle_db_size 2000 \
    --imle_batch 50 \
    --teacher_force_resample 20 \
    --subset_len -1 \
    --lr 0.0002 \
    --n_batch 12 \
    --teacher_generate_initial_data True
' > /home/kha98/Desktop/rs-imle/runs/resample_teacher/resample-scheduled-lr-2-resample-20/train.log 2>&1 &


nohup bash -c '
CUDA_VISIBLE_DEVICES=4 python /home/kha98/Desktop/rs-imle/train.py \
    --hps fewshot \
    --save_dir /home/kha98/Desktop/rs-imle/runs/resample_teacher/resample-scheduled-lr-2-resample-20-data-100/ \
    --fid_real_dir /home/kha98/Desktop/rs-imle/teacher/clean/img \
    --model_type unet \
    --force_factor 20 \
    --imle_force_resample 10 \
    --use_comet True \
    --latent_dim 4096 \
    --comet_name resample-scheduled-lr-2-resample-20-data-100 \
    --comet_api_key esx5iX53IbgtEtr4Zj1tkxpYB \
    --use_teacher_resample True \
    --teacher_checkpoint_path /home/kha98/Desktop/rs-imle/teacher/fm_cifar10_weights_step_84000.pt \
    --teacher_resample_steps 20 \
    --teacher_num_samples 100 \
    --imle_db_size 2000 \
    --imle_batch 50 \
    --teacher_force_resample 20 \
    --subset_len -1 \
    --lr 0.0002 \
    --n_batch 12 \
    --teacher_generate_initial_data True
' > /home/kha98/Desktop/rs-imle/runs/resample_teacher/resample-scheduled-lr-2-resample-20-data-100/train.log 2>&1 &

nohup bash -c '
CUDA_VISIBLE_DEVICES=5 python /home/kha98/Desktop/rs-imle/train.py \
    --hps fewshot \
    --save_dir /home/kha98/Desktop/rs-imle/runs/resample_teacher/resample-scheduled-5/ \
    --fid_real_dir /home/kha98/Desktop/rs-imle/teacher/clean/img \
    --model_type unet \
    --force_factor 20 \
    --imle_force_resample 10 \
    --use_comet True \
    --latent_dim 4096 \
    --comet_name teacher-resample-scheduled-5 \
    --comet_api_key esx5iX53IbgtEtr4Zj1tkxpYB \
    --use_teacher_resample True \
    --teacher_checkpoint_path /home/kha98/Desktop/rs-imle/teacher/fm_cifar10_weights_step_84000.pt \
    --teacher_resample_steps 20 \
    --teacher_num_samples 1000 \
    --imle_db_size 2000 \
    --imle_batch 50 \
    --teacher_force_resample 20 \
    --subset_len -1 \
    --lr 0.0004 \
    --n_batch 24 \
    --teacher_generate_initial_data True
' > /home/kha98/Desktop/rs-imle/runs/resample_teacher/resample-scheduled-5/train.log 2>&1 &
