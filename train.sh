CUDA_VISIBLE_DEVICES=0 python train.py \
    --hps fewshot \
    --save_dir ./home/kha98/rs-imle/vanilla-run/learn-sampling-strategy \
    --condition_path /home/kha98/flow-model/flow-model-chirag/results/icfm/x_T.pt \
    --data_root /home/kha98/rs-imle/datasets/ffhq \
    --force_factor 5 \
    --imle_staleness 5 \
    --imle_force_resample 5 \
    --lr 0.0002 \
    --comet_api_key 'esx5iX53IbgtEtr4Zj1tkxpYB' \
    --comet_name 'flow-model-imle' \
    --use_comet False \
    --use_snoise False \
    --conditional_force_faiss False \
    --imle_db_size 500 \
    --imle_batch 50
 # to get 500 samples in total -> self.pool_size = ceil(int(H.force_factor * sz) / H.imle_db_size) * H.imle_db_size