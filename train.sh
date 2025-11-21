CUDA_VISIBLE_DEVICES=0 python train.py \
    --hps fewshot \
    --save_dir /home/kha98/Desktop/rs-imle/debug/ \
    --condition_path /home/kha98/Desktop/rs-imle/dataset/ffhq/x_0.pt \
    --data_root /home/kha98/Desktop/rs-imle/dataset/ffhq/intermediate/ \
    --force_factor 20 \
    --imle_staleness 5 \
    --imle_force_resample 5 \
    --lr 0.0002 \
    --comet_api_key 'esx5iX53IbgtEtr4Zj1tkxpYB' \
    --comet_name 'inter-62-lr-2-e-4' \
    --use_comet True \
    --use_snoise False \
    --conditional_force_faiss False \
    --imle_db_size 2000 \
    --imle_batch 50

 # to get 500 samples in total -> self.pool_size = ceil(int(H.force_factor * sz) / H.imle_db_size) * H.imle_db_size



 nohup bash -c "CUDA_VISIBLE_DEVICES=0 python train.py \
    --hps fewshot \
    --save_dir /home/kha98/Desktop/rs-imle/runs/inter-62-lr-2-e-4 \
    --condition_path /home/kha98/Desktop/rs-imle/dataset/ffhq/x_0.pt \
    --data_root /home/kha98/Desktop/rs-imle/dataset/ffhq/intermediate/ \
    --force_factor 20 \
    --imle_staleness 5 \
    --imle_force_resample 5 \
    --lr 0.0002 \
    --comet_api_key 'esx5iX53IbgtEtr4Zj1tkxpYB' \
    --comet_name 'inter-62-lr-2-e-4' \
    --use_comet True \
    --use_snoise False \
    --conditional_force_faiss False \
    --imle_db_size 2000 \
    --imle_batch 50" > /home/kha98/Desktop/rs-imle/runs/inter-62-lr-2-e-4/train.log 2>&1 &



 nohup bash -c "CUDA_VISIBLE_DEVICES=0 python train.py \
    --hps fewshot \
    --save_dir /home/kha98/Desktop/rs-imle/runs/5-steps-lr-2-e-4 \
    --condition_path /home/kha98/Desktop/rs-imle/dataset/ffhq/x0_5_seed_9.pt\
    --data_root /home/kha98/Desktop/rs-imle/dataset/ffhq/5_steps_seed_9/ \
    --force_factor 20 \
    --imle_staleness 5 \
    --imle_force_resample 5 \
    --lr 0.0002 \
    --comet_api_key 'esx5iX53IbgtEtr4Zj1tkxpYB' \
    --comet_name '5-steps-lr-2-e-4' \
    --use_comet True \
    --use_snoise False \
    --conditional_force_faiss False \
    --imle_db_size 2000 \
    --imle_batch 50" > /home/kha98/Desktop/rs-imle/runs/5-steps-lr-2-e-4/train.log 2>&1 &



# step 10

 nohup bash -c "CUDA_VISIBLE_DEVICES=0 python train.py \
    --hps fewshot \
    --save_dir /home/kha98/Desktop/rs-imle/runs/10-prev-5-earlier \
    --condition_path /home/kha98/Desktop/rs-imle/dataset/ffhq/x0_10_seed_9.pt \
    --data_root /home/kha98/Desktop/rs-imle/dataset/ffhq/10_steps_seed_9/ \
    --force_factor 5 \
    --imle_staleness 5 \
    --imle_force_resample 5 \
    --lr 0.0002 \
    --comet_api_key 'esx5iX53IbgtEtr4Zj1tkxpYB' \
    --comet_name '10-steps-5-end-earlier' \
    --use_comet True \
    --use_snoise False \
    --conditional_force_faiss False \
    --imle_db_size 500 \
    --iters_per_ckpt 50000 \
    --restore_path /home/kha98/Desktop/rs-imle/runs/5-steps-factor-10/train/best_fid-model.th \
    --restore_ema_path /home/kha98/Desktop/rs-imle/runs/5-steps-factor-10/train/best_fid-model-ema.th \
    --imle_batch 50" > /home/kha98/Desktop/rs-imle/runs/10-prev-5-earlier/train.log 2>&1 &

 nohup bash -c "CUDA_VISIBLE_DEVICES=0 python train.py \
    --h-mode eval \
    --restore_path /home/kha98/Desktop/rs-imle/one-schedule/VDVAE-clean-debug2/train/best_fid-model.th \ 
    --imle_batch 50" > /home/kha98/Desktop/rs-imle/runs/10-prev-5-earlier/train.log 2>&1 &


# running with sample factor 5 model that was trained with step 5 
   nohup bash -c "CUDA_VISIBLE_DEVICES=0 python train.py \
    --hps fewshot \
    --save_dir /home/kha98/Desktop/rs-imle/runs/20-step-prev-10-5-fact-5 \
    --condition_path /home/kha98/Desktop/rs-imle/dataset/ffhq/x0_20_seed_9.pt \
    --data_root /home/kha98/Desktop/rs-imle/dataset/ffhq/20_steps_seed_9 \
    --force_factor 5 \
    --imle_staleness 5 \
    --imle_force_resample 5 \
    --lr 0.0002 \
    --comet_api_key 'esx5iX53IbgtEtr4Zj1tkxpYB' \
    --comet_name '20-step-prev-10-5-fact-5' \
    --use_comet True \
    --use_snoise False \
    --conditional_force_faiss False \
    --imle_db_size 500 \
    --restore_path /home/kha98/Desktop/rs-imle/runs/10-steps-prev-5-fact-5/train/iter-100000-model.th \
    --restore_ema_path /home/kha98/Desktop/rs-imle/runs/10-steps-prev-5-fact-5/train/iter-100000-model-ema.th \
    --imle_batch 50" > /home/kha98/Desktop/rs-imle/runs/20-step-prev-10-5-fact-5/train.log 2>&1 &

# running without any prev model just step 10 samples

 nohup bash -c "CUDA_VISIBLE_DEVICES=1 python train.py \
    --hps fewshot \
    --save_dir /home/kha98/Desktop/rs-imle/runs/10-steps-no-prev \
    --condition_path /home/kha98/Desktop/rs-imle/dataset/ffhq/x0_10_seed_9.pt \
    --data_root /home/kha98/Desktop/rs-imle/dataset/ffhq/10_steps_seed_9/ \
    --force_factor 5 \
    --imle_staleness 5 \
    --imle_force_resample 5 \
    --lr 0.0002 \
    --comet_api_key 'esx5iX53IbgtEtr4Zj1tkxpYB' \
    --comet_name '10-steps-no-prev' \
    --use_comet True \
    --use_snoise False \
    --conditional_force_faiss False \
    --imle_db_size 500 \
    --imle_batch 50" > /home/kha98/Desktop/rs-imle/runs/10-steps-no-prev/train.log 2>&1 &



 nohup bash -c "CUDA_VISIBLE_DEVICES=0 python train.py --hps fewshot \
    --save_dir /home/kha98/Desktop/rs-imle/runs/VDVAE \
    --data_root /home/kha98/Desktop/rs-imle/dataset/ffhq/clean \
    --lr 0.0002 \
    --imle_force_resample 10 \
    --use_comet True \
    --restore_path /home/kha98/Desktop/rs-imle/runs/VDVAE/train/latest-model.th \
    --restore_ema_path /home/kha98/Desktop/rs-imle/runs/VDVAE/train/latest-model-ema.th \
    --restore_log_path /home/kha98/Desktop/rs-imle/runs/VDVAE/train/log.jsonl \
    --restore_optimizer_path /home/kha98/Desktop/rs-imle/runs/VDVAE/train/latest-opt.th \
    --restore_scheduler_path /home/kha98/Desktop/rs-imle/runs/VDVAE/train/latest-sched.th \
    --comet_experiment_key '331f44168583482883e209feef88835f' \
    --comet_api_key 'esx5iX53IbgtEtr4Zj1tkxpYB'
   " > /home/kha98/Desktop/rs-imle/runs/VDVAE/train.log 2>&1 &



CUDA_VISIBLE_DEVICES=1 python train.py --hps fewshot \
    --save_dir /home/kha98/Desktop/rs-imle/runs/debug \
    --condition_path /home/kha98/Desktop/rs-imle/dataset/ffhq/x0_10_seed_9.pt \
    --data_root /home/kha98/Desktop/rs-imle/dataset/ffhq/clean \
    --lr 0.0002 \
    --imle_force_resample 10 \
    --comet_api_key 'esx5iX53IbgtEtr4Zj1tkxpYB' \
    --comet_name 'debug' \
    --imle_db_size 500 \
    --imle_batch 50 \
    --latent_dim 4096

CUDA_VISIBLE_DEVICES=1 python train.py --hps fewshot \
    --save_dir /home/kha98/Desktop/rs-imle/runs/debug \
    --data_root /home/kha98/Desktop/rs-imle/dataset/ffhq/clean \
    --condition_path /home/kha98/Desktop/rs-imle/dataset/ffhq/x0_10_seed_9.pt \
    --lr 0.0002 \
    --imle_force_resample 10 \
    --comet_api_key 'esx5iX53IbgtEtr4Zj1tkxpYB' \
    --imle_db_size 500 \
    --imle_batch 50 \
    --latent_dim 4096 

CUDA_VISIBLE_DEVICES=0 python train.py \
    --hps fewshot \
    --mode eval_fid \
    --restore_path /home/kha98/Desktop/rs-imle/runs/best_fid-model-ema.th \
    --fid_real_dir /home/kha98/Desktop/rs-imle/dataset/ffhq/clean \
    --data_root /home/kha98/Desktop/rs-imle/dataset/ffhq/clean \
    --save_dir /home/kha98/Desktop/rs-imle/one-schedule/fid_eval

CUDA_VISIBLE_DEVICES=1 python train.py \
    --hps fewshot \
    --mode eval_fid \
    --restore_path /home/kha98/Desktop/rs-imle/runs/best_fid-model.th \
    --fid_real_dir /home/kha98/Desktop/rs-imle/dataset/ffhq/clean \
    --data_root /home/kha98/Desktop/rs-imle/dataset/ffhq/clean \
    --save_dir /home/kha98/Desktop/rs-imle/one-schedule/fid_eval2

python train.py --hps fewshot \
    --save_dir ./new-vanilla-results/ffhq/ \
    --data_root ./datasets/ffhq/ \
    --lr 0.0002 


# ============================================================
# Model Selection Examples with --model_type hyperparameter
# ============================================================

# VDVAE -> clean force factor 20


nohup bash -c "CUDA_VISIBLE_DEVICES=0 python train.py --hps fewshot \
   --save_dir /home/kha98/Desktop/rs-imle/runs/VDVAE-2/VDVAE-clean \
   --data_root /home/kha98/Desktop/rs-imle/dataset/ffhq/clean \
   --lr 0.0002 \
   --imle_force_resample 10 \
   --use_comet True \
   --model_type vdvae \
   --comet_name 'VDVAE-clean-fact-20' \
   --comet_api_key 'esx5iX53IbgtEtr4Zj1tkxpYB' \
   --force_factor 20 \
   --restore_path /home/kha98/Desktop/rs-imle/runs/VDVAE/train/best_fid-model.th \
   --restore_ema_path /home/kha98/Desktop/rs-imle/runs/VDVAE/train/best_fid-model-ema.th \
   --restore_optimizer_path /home/kha98/Desktop/rs-imle/runs/VDVAE/train/best_fid-opt.th \
   --restore_scheduler_path /home/kha98/Desktop/rs-imle/runs/VDVAE/train/best_fid-sched.th" > /home/kha98/Desktop/rs-imle/runs/VDVAE/train.log 2>&1 &

# Example 2: Force VDVAE model (ignores condition_path)
# python train.py --hps fewshot --model_type vdvae --data_root ./datasets/ffhq/

# Example 3: Force UNet model with conditions
# python train.py --hps fewshot --model_type unet \
#     --condition_path /path/to/conditions.pt \
#     --data_root ./datasets/ffhq/

nohup bash -c "CUDA_VISIBLE_DEVICES=0 python train.py --hps fewshot \
   --save_dir /home/kha98/Desktop/rs-imle/one-schedule/VDVAE-clean \
   --data_root /home/kha98/Desktop/rs-imle/dataset/ffhq/clean \
   --lr 0.0002 \
   --imle_force_resample 10 \
   --force_factor 20 \
   --use_comet True \
   --model_type vdvae \
   --comet_name 'VDVAE-clean' \
   --comet_api_key 'esx5iX53IbgtEtr4Zj1tkxpYB'" > /home/kha98/Desktop/rs-imle/one-schedule/VDVAE-clean/train.log 2>&1 &


# Example 4: Force UNet model WITHOUT conditions (unconditional UNet)
# python train.py --hps fewshot --model_type unet --data_root ./datasets/ffhq/
# VDVAE → clean

# UNET → clean
nohup bash -c "CUDA_VISIBLE_DEVICES=1 python train.py --hps fewshot \
   --save_dir /home/kha98/Desktop/rs-imle/one-schedule/UNET-clean \
   --data_root /home/kha98/Desktop/rs-imle/dataset/ffhq/clean \
   --model_type unet \
   --lr 0.0002 \
   --latent_dim 4096 \
   --imle_force_resample 10 \
   --use_comet True \
   --comet_name 'UNET-clean' \
   --comet_api_key 'esx5iX53IbgtEtr4Zj1tkxpYB'" > /home/kha98/Desktop/rs-imle/one-schedule/UNET-clean/train.log 2>&1 &

# VDVAE → 20-steps 
nohup bash -c "CUDA_VISIBLE_DEVICES=1 python train.py --hps fewshot \
   --save_dir /home/kha98/Desktop/rs-imle/one-schedule/VDVAE-20-steps-fact20 \
   --data_root /home/kha98/Desktop/rs-imle/dataset/ffhq/20_steps_seed_9 \
   --fid_real_dir /home/kha98/Desktop/rs-imle/dataset/ffhq/clean \
   --lr 0.0002 \
   --imle_force_resample 10 \
   --use_comet True \
   --model_type vdvae \
   --force_factor 20 \
   --comet_name 'VDVAE-20-steps-fact20' \
   --comet_api_key 'esx5iX53IbgtEtr4Zj1tkxpYB'" > /home/kha98/Desktop/rs-imle/one-schedule/VDVAE-20-steps/train.log 2>&1 &

# Unet -> 20-steps
nohup bash -c "CUDA_VISIBLE_DEVICES=0 python train.py --hps fewshot \
   --save_dir /home/kha98/Desktop/rs-imle/one-schedule/UNET-20-steps-unconditional \
   --data_root /home/kha98/Desktop/rs-imle/dataset/ffhq/20_steps_seed_9 \
   --fid_real_dir /home/kha98/Desktop/rs-imle/dataset/ffhq/clean \
   --model_type unet \
   --lr 0.0002 \
   --latent_dim 4096 \
   --imle_force_resample 10 \
   --use_comet True \
   --comet_name 'UNET-20-steps-unconditional' \
   --comet_api_key 'esx5iX53IbgtEtr4Zj1tkxpYB'" > /home/kha98/Desktop/rs-imle/one-schedule/UNET-20-steps-unconditional/train.log 2>&1 &

# Unet conditional -> 20-steps
nohup bash -c "CUDA_VISIBLE_DEVICES=0 python train.py --hps fewshot \
   --save_dir /home/kha98/Desktop/rs-imle/one-schedule/UNET-20-steps-conditional \
   --data_root /home/kha98/Desktop/rs-imle/dataset/ffhq/20_steps_seed_9 \
   --condition_path /home/kha98/Desktop/rs-imle/dataset/ffhq/x0_20_seed_9.pt \
   --fid_real_dir /home/kha98/Desktop/rs-imle/dataset/ffhq/clean \
   --model_type unet \
   --lr 0.0002 \
   --latent_dim 4096 \
   --imle_force_resample 10 \
   --use_comet True \
   --comet_name 'UNET-20-steps-conditional' \
   --comet_api_key 'esx5iX53IbgtEtr4Zj1tkxpYB'" > /home/kha98/Desktop/rs-imle/one-schedule/UNET-20-steps-conditional/train.log 2>&1 &



# Debug unet
# Unet -> 20-steps
CUDA_VISIBLE_DEVICES=0 python train.py --hps fewshot \
   --save_dir /home/kha98/Desktop/rs-imle/one-schedule/debug \
   --data_root /home/kha98/Desktop/rs-imle/dataset/ffhq/20_steps_seed_9 \
   --fid_real_dir /home/kha98/Desktop/rs-imle/dataset/ffhq/clean \
   --condition_path /home/kha98/Desktop/rs-imle/dataset/ffhq/x0_20_seed_9.pt \
   --model_type unet \
   --lr 0.0002 \
   --latent_dim 4096 \
   --imle_force_resample 10

# Unet conditional -> 20-steps
nohup bash -c "CUDA_VISIBLE_DEVICES=0 python train.py --hps fewshot \
   --save_dir /home/kha98/Desktop/rs-imle/one-schedule/UNET-20-steps-conditional \
   --data_root /home/kha98/Desktop/rs-imle/dataset/ffhq/20_steps_seed_9 \
   --condition_path /home/kha98/Desktop/rs-imle/dataset/ffhq/x0_20_seed_9.pt \
   --fid_real_dir /home/kha98/Desktop/rs-imle/dataset/ffhq/clean \
   --model_type unet \
   --lr 0.0002 \
   --latent_dim 4096 \
   --imle_force_resample 10 \
   --use_comet True \
   --comet_name 'UNET-20-steps-conditional' \
   --comet_api_key 'esx5iX53IbgtEtr4Zj1tkxpYB'" > /home/kha98/Desktop/rs-imle/one-schedule/UNET-20-steps-conditional/train.log 2>&1 &



















   nohup bash -c "CUDA_VISIBLE_DEVICES=0 python train.py --hps fewshot \
   --save_dir /home/kha98/Desktop/rs-imle/one-schedule/VDVAE-clean-debug \
   --data_root /home/kha98/Desktop/rs-imle/dataset/ffhq/clean \
   --lr 0.0002 \
   --imle_force_resample 10 \
   --use_comet True \
   --model_type vdvae \
   --comet_name 'VDVAE-clean-debug1' \
   --comet_api_key 'esx5iX53IbgtEtr4Zj1tkxpYB'" > /home/kha98/Desktop/rs-imle/one-schedule/VDVAE-clean-debug/train.log 2>&1 &


   /latest-model-ema.th
   nohup bash -c "CUDA_VISIBLE_DEVICES=1 python train.py --hps fewshot \
   --save_dir /home/kha98/Desktop/rs-imle/one-schedule/VDVAE-clean-debug2 \
   --data_root /home/kha98/Desktop/rs-imle/dataset/ffhq/clean \
   --lr 0.0002 \
   --imle_force_resample 10 \
   --use_comet True \
   --model_type vdvae \
   --comet_name 'VDVAE-clean-debug2' \
   --restore_path /home/kha98/Desktop/rs-imle/one-schedule/VDVAE-clean-debug2/train/latest-model.th \
   --restore_ema_path /home/kha98/Desktop/rs-imle/one-schedule/VDVAE-clean-debug2/train/latest-model-ema.th \
   --restore_log_path /home/kha98/Desktop/rs-imle/one-schedule/VDVAE-clean-debug2/train/log.jsonl \
   --restore_optimizer_path /home/kha98/Desktop/rs-imle/one-schedule/VDVAE-clean-debug2/train/latest-opt.th \
   --restore_scheduler_path /home/kha98/Desktop/rs-imle/one-schedule/VDVAE-clean-debug2/train/latest-sched.th \
   --comet_experiment_key '83172a9331ab45c4b38ea18a393193da' \
   --comet_api_key 'esx5iX53IbgtEtr4Zj1tkxpYB'" > /home/kha98/Desktop/rs-imle/one-schedule/VDVAE-clean-debug2/train.log 2>&1 &