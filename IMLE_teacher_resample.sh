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










# \
#     --restore_path /localscratch/kian/resample_teacher/resample-scheduled-lr-4-resample-20/resample-scheduled-lr-4-resample-20/train/latest-model.th \
#     --restore_ema_path /localscratch/kian/resample_teacher/resample-scheduled-lr-4-resample-20/resample-scheduled-lr-4-resample-20/train/latest-model-ema.th \
#     --restore_log_path /localscratch/kian/resample_teacher/resample-scheduled-lr-4-resample-20/resample-scheduled-lr-4-resample-20/train/log.jsonl \
#     --restore_optimizer_path /localscratch/kian/resample_teacher/resample-scheduled-lr-4-resample-20/resample-scheduled-lr-4-resample-20/train/latest-opt.th \
#     --restore_scheduler_path /localscratch/kian/resample_teacher/resample-scheduled-lr-4-resample-20/resample-scheduled-lr-4-resample-20/train/latest-sched.th \
#     --comet_experiment_key '9981680daa774033a67daa2fb6d1a6f9'




nohup bash -c '
CUDA_VISIBLE_DEVICES=6 python /home/kha98/Desktop/rs-imle/train.py \
    --hps fewshot \
    --save_dir  /localscratch/kian/resample_teacher/regression-noise-as-input-4000/ \
    --fid_real_dir /home/kha98/Desktop/rs-imle/teacher/clean/img \
    --model_type unet \
    --force_factor 1 \
    --imle_force_resample 1 \
    --use_comet True \
    --latent_dim 4096 \
    --comet_name regression-noise-as-input-4000 \
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
    --teacher_generate_initial_data True \
    --use_teacher_noise_as_input True \
    --fid_freq 30 \
    --iters_per_save 5000 \
    --restore_path /localscratch/kian/resample_teacher/regression-noise-as-input-4000/train/latest-model.th \
    --restore_ema_path /localscratch/kian/resample_teacher/regression-noise-as-input-4000/train/latest-model-ema.th \
    --restore_log_path /localscratch/kian/resample_teacher/regression-noise-as-input-4000/train/log.jsonl \
    --restore_optimizer_path /localscratch/kian/resample_teacher/regression-noise-as-input-4000/train/latest-opt.th \
    --restore_scheduler_path /localscratch/kian/resample_teacher/regression-noise-as-input-4000/train/latest-sched.th \
    --comet_experiment_key '3d30adf8bfa94197a480fa1016aad68b'
' > /localscratch/kian/resample_teacher/regression-noise-as-input-4000/train-1.log 2>&1 &




# Example: Pure Regression Mode - Teacher noise as input (unconditional UNet)
# This mode uses the teacher's input noise (x0) directly as input to the student.
# The UNet will be unconditional and learns: teacher_noise → teacher_output
# Use force_factor=1 for pure regression (no IMLE sampling)
# Latent codes are ignored in this mode.
#


nohup bash -c '
CUDA_VISIBLE_DEVICES=8 python /home/kha98/Desktop/rs-imle/train.py \
    --hps fewshot \
    --save_dir  /localscratch/kian/resample_teacher/regression-noise-as-input-25000/ \
    --fid_real_dir /home/kha98/Desktop/rs-imle/teacher/clean/img \
    --model_type unet \
    --force_factor 1 \
    --imle_force_resample 1 \
    --use_comet True \
    --latent_dim 4096 \
    --comet_name regression-noise-as-input-25000 \
    --comet_api_key esx5iX53IbgtEtr4Zj1tkxpYB \
    --use_teacher_resample True \
    --teacher_checkpoint_path /home/kha98/Desktop/rs-imle/teacher/fm_cifar10_weights_step_84000.pt \
    --teacher_resample_steps 20 \
    --teacher_num_samples 25000 \
    --imle_db_size 2000 \
    --imle_batch 50 \
    --teacher_force_resample 20000000000000000000 \
    --subset_len -1 \
    --lr 0.0002 \
    --n_batch 12 \
    --teacher_generate_initial_data True \
    --use_teacher_noise_as_input True \
    --fid_freq 10 \
    --iters_per_save 5000 \
    --restore_path /localscratch/kian/resample_teacher/regression-noise-as-input-25000/train/best_fid-model.th \
    --restore_ema_path /localscratch/kian/resample_teacher/regression-noise-as-input-25000/train/best_fid-model-ema.th \
    --restore_log_path /localscratch/kian/resample_teacher/regression-noise-as-input-25000/train/best_fid-log.jsonl \
    --restore_optimizer_path /localscratch/kian/resample_teacher/regression-noise-as-input-25000/train/best_fid-opt.th \
    --restore_scheduler_path /localscratch/kian/resample_teacher/regression-noise-as-input-25000/train/best_fid-sched.th \
' > /localscratch/kian/resample_teacher/regression-noise-as-input-25000/train-1.log 2>&1 &


nohup bash -c '
CUDA_VISIBLE_DEVICES=0 python /home/kha98/Desktop/rs-imle/train.py \
    --hps fewshot \
    --save_dir  /localscratch/kian/resample_teacher/regression-noise-as-input-25000/ \
    --fid_real_dir /home/kha98/Desktop/rs-imle/teacher/clean/img \
    --model_type unet \
    --force_factor 1 \
    --imle_force_resample 1 \
    --use_comet True \
    --latent_dim 4096 \
    --comet_name regression-noise-as-input-25000 \
    --comet_api_key esx5iX53IbgtEtr4Zj1tkxpYB \
    --use_teacher_resample True \
    --teacher_checkpoint_path /home/kha98/Desktop/rs-imle/teacher/fm_cifar10_weights_step_84000.pt \
    --teacher_resample_steps 20 \
    --teacher_num_samples 25000 \
    --imle_db_size 2000 \
    --imle_batch 50 \
    --teacher_force_resample 20000000000000000000 \
    --subset_len -1 \
    --lr 0.0002 \
    --n_batch 12 \
    --teacher_generate_initial_data True \
    --use_teacher_noise_as_input True \
    --fid_freq 10 \
    --iters_per_save 5000 \
    --comet_experiment_key c4c8e8a37d0d4509b8eb9c274dcf33f6 \
    --restore_path /localscratch/kian/resample_teacher/regression-noise-as-input-25000/train/latest-model.th \
    --restore_ema_path /localscratch/kian/resample_teacher/regression-noise-as-input-25000/train/latest-model-ema.th \
    --restore_log_path /localscratch/kian/resample_teacher/regression-noise-as-input-25000/train/log.jsonl \
    --restore_optimizer_path /localscratch/kian/resample_teacher/regression-noise-as-input-25000/train/latest-opt.th \
    --restore_scheduler_path /localscratch/kian/resample_teacher/regression-noise-as-input-25000/train/latest-sched.th \
' > /localscratch/kian/resample_teacher/regression-noise-as-input-25000/train-2.log 2>&1 &


nohup bash -c '
CUDA_VISIBLE_DEVICES=1 python /home/kha98/Desktop/rs-imle/train.py \
    --hps fewshot \
    --save_dir  /localscratch/kian/resample_teacher/regression-noise-as-input-50000/ \
    --fid_real_dir /home/kha98/Desktop/rs-imle/teacher/clean/img \
    --model_type unet \
    --force_factor 1 \
    --imle_force_resample 1 \
    --use_comet True \
    --latent_dim 4096 \
    --comet_api_key esx5iX53IbgtEtr4Zj1tkxpYB \
    --use_teacher_resample True \
    --teacher_checkpoint_path /home/kha98/Desktop/rs-imle/teacher/fm_cifar10_weights_step_84000.pt \
    --teacher_resample_steps 20 \
    --teacher_num_samples 50000 \
    --imle_db_size 2000 \
    --imle_batch 50 \
    --teacher_force_resample 20000000000000000000 \
    --subset_len -1 \
    --lr 0.0002 \
    --n_batch 12 \
    --teacher_generate_initial_data True \
    --use_teacher_noise_as_input True \
    --fid_freq 10 \
    --iters_per_save 5000 \
    --comet_experiment_key cd62018c663543c9a94c6515b72d3b69 \
    --restore_path /localscratch/kian/resample_teacher/regression-noise-as-input-50000/train/latest-model.th \
    --restore_ema_path /localscratch/kian/resample_teacher/regression-noise-as-input-50000/train/latest-model-ema.th \
    --restore_log_path /localscratch/kian/resample_teacher/regression-noise-as-input-50000/train/log.jsonl \
    --restore_optimizer_path /localscratch/kian/resample_teacher/regression-noise-as-input-50000/train/latest-opt.th \
    --restore_scheduler_path /localscratch/kian/resample_teacher/regression-noise-as-input-50000/train/latest-sched.th \
' > /localscratch/kian/resample_teacher/regression-noise-as-input-50000/train-1.log 2>&1 &



nohup bash -c '
CUDA_VISIBLE_DEVICES=8 python /home/kha98/Desktop/rs-imle/train.py \
    --hps fewshot \
    --save_dir  /localscratch/kian/resample_teacher/regression-noise-as-input-1000-ref-50/ \
    --fid_real_dir /home/kha98/Desktop/rs-imle/teacher/clean/img \
    --model_type unet \
    --force_factor 1 \
    --imle_force_resample 1 \
    --use_comet True \
    --latent_dim 4096 \
    --comet_name regression-noise-as-input-1000-ref-50 \
    --comet_api_key esx5iX53IbgtEtr4Zj1tkxpYB \
    --use_teacher_resample True \
    --teacher_checkpoint_path /home/kha98/Desktop/rs-imle/teacher/fm_cifar10_weights_step_84000.pt \
    --teacher_resample_steps 20 \
    --teacher_num_samples 1000 \
    --imle_db_size 2000 \
    --imle_batch 50 \
    --teacher_force_resample 50 \
    --subset_len -1 \
    --lr 0.0002 \
    --n_batch 12 \
    --teacher_generate_initial_data True \
    --use_teacher_noise_as_input True \
    --fid_freq 30 
' > /localscratch/kian/resample_teacher/regression-noise-as-input-1000-ref-50/train.log 2>&1 &



nohup bash -c '
CUDA_VISIBLE_DEVICES=1 python /home/kha98/Desktop/rs-imle/train.py \
    --hps fewshot \
    --save_dir  /localscratch/kian/resample_teacher/regression-noise-as-input-25000-ref-50/ \
    --fid_real_dir /home/kha98/Desktop/rs-imle/teacher/clean/img \
    --model_type unet \
    --force_factor 1 \
    --imle_force_resample 1 \
    --use_comet True \
    --latent_dim 4096 \
    --comet_name regression-noise-as-input-25000-ref-50 \
    --comet_api_key esx5iX53IbgtEtr4Zj1tkxpYB \
    --use_teacher_resample True \
    --teacher_checkpoint_path /home/kha98/Desktop/rs-imle/teacher/fm_cifar10_weights_step_84000.pt \
    --teacher_resample_steps 20 \
    --teacher_num_samples 25000 \
    --imle_db_size 2000 \
    --imle_batch 50 \
    --teacher_force_resample 50 \
    --subset_len -1 \
    --lr 0.0002 \
    --n_batch 12 \
    --teacher_generate_initial_data True \
    --use_teacher_noise_as_input True \
    --fid_freq 10 
' > /localscratch/kian/resample_teacher/regression-noise-as-input-25000-ref-50/train.log 2>&1 &


nohup bash -c '
CUDA_VISIBLE_DEVICES=3 python /home/kha98/Desktop/rs-imle/train.py \
    --hps fewshot \
    --save_dir /localscratch/kian/resample_teacher/resample-scheduled-lr-2-resample-20/ \
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
    --teacher_resample_steps 10 \
    --teacher_num_samples 4000 \
    --imle_db_size 2000 \
    --imle_batch 50 \
    --teacher_force_resample 20 \
    --subset_len -1 \
    --lr 0.0002 \
    --n_batch 12 \
    --teacher_generate_initial_data True \
    --fid_freq 30 \
    --restore_path /localscratch/kian/resample_teacher/resample-scheduled-lr-2-resample-20/train/best_fid-model.th \
    --restore_ema_path /localscratch/kian/resample_teacher/resample-scheduled-lr-2-resample-20/train/best_fid-model-ema.th \
    --restore_log_path /localscratch/kian/resample_teacher/resample-scheduled-lr-2-resample-20/train/log.jsonl \
    --restore_optimizer_path /localscratch/kian/resample_teacher/resample-scheduled-lr-2-resample-20/train/best_fid-opt.th \
    --restore_scheduler_path /localscratch/kian/resample_teacher/resample-scheduled-lr-2-resample-20/train/best_fid-sched.th \
    --comet_experiment_key '3d60d7f60c444ffa9f958a6d96bdb555'
' > /localscratch/kian/resample_teacher/resample-scheduled-lr-2-resample-20/train.log 2>&1 &


nohup bash -c '
CUDA_VISIBLE_DEVICES=0 python /home/kha98/Desktop/rs-imle/train.py \
    --hps fewshot \
    --save_dir /localscratch/kian/resample_teacher/resample-scheduled-lr-2-resample-40/ \
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
    --teacher_generate_initial_data True \
    --fid_freq 30 \
    --restore_path /localscratch/kian/resample_teacher/resample-scheduled-lr-2-resample-40/train/latest-model.th \
    --restore_ema_path /localscratch/kian/resample_teacher/resample-scheduled-lr-2-resample-40/train/latest-model-ema.th \
    --restore_log_path /localscratch/kian/resample_teacher/resample-scheduled-lr-2-resample-40/train/log.jsonl \
    --restore_optimizer_path /localscratch/kian/resample_teacher/resample-scheduled-lr-2-resample-40/train/latest-opt.th \
    --restore_scheduler_path /localscratch/kian/resample_teacher/resample-scheduled-lr-2-resample-40/train/latest-sched.th \
    --comet_experiment_key '86dd00b2f6a64e84957088cf27181ee8'
' > /localscratch/kian/resample_teacher/resample-scheduled-lr-2-resample-40/train-1.log 2>&1 &


nohup bash -c '
CUDA_VISIBLE_DEVICES=3 python /home/kha98/Desktop/rs-imle/train.py \
    --hps fewshot \
    --save_dir /localscratch/kian/resample_teacher/resample-scheduled-lr-2-resample-20/ \
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
    --teacher_resample_steps 10 \
    --teacher_num_samples 4000 \
    --imle_db_size 2000 \
    --imle_batch 50 \
    --teacher_force_resample 20 \
    --subset_len -1 \
    --lr 0.0002 \
    --n_batch 12 \
    --teacher_generate_initial_data True \
    --fid_freq 30 \
    --restore_path /localscratch/kian/resample_teacher/resample-scheduled-lr-2-resample-20/train/best_fid-model.th \
    --restore_ema_path /localscratch/kian/resample_teacher/resample-scheduled-lr-2-resample-20/train/best_fid-model-ema.th \
    --restore_log_path /localscratch/kian/resample_teacher/resample-scheduled-lr-2-resample-20/train/log.jsonl \
    --restore_optimizer_path /localscratch/kian/resample_teacher/resample-scheduled-lr-2-resample-20/train/best_fid-opt.th \
    --restore_scheduler_path /localscratch/kian/resample_teacher/resample-scheduled-lr-2-resample-20/train/best_fid-sched.th \
    --comet_experiment_key 'a4976dfb96034c3b9d0c55f6f6db3fbb'
' > /localscratch/kian/resample_teacher/resample-scheduled-lr-2-resample-20/train.log 2>&1 &
