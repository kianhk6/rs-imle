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



 nohup bash -c "CUDA_VISIBLE_DEVICES=1 python train.py \
    --hps fewshot \
    --save_dir /home/kha98/Desktop/rs-imle/runs/5-steps-factor-5 \
    --condition_path /home/kha98/Desktop/rs-imle/dataset/ffhq/x0_5_seed_9.pt\
    --data_root /home/kha98/Desktop/rs-imle/dataset/ffhq/5_steps_seed_9/ \
    --force_factor 5 \
    --imle_staleness 5 \
    --imle_force_resample 5 \
    --lr 0.0002 \
    --comet_api_key 'esx5iX53IbgtEtr4Zj1tkxpYB' \
    --comet_name '5-steps-lr-2-e-4-factor-5' \
    --use_comet True \
    --use_snoise False \
    --conditional_force_faiss False \
    --imle_db_size 500 \
    --imle_batch 50" > /home/kha98/Desktop/rs-imle/runs/5-steps-lr-2-e-4-factor-5/train.log 2>&1 &