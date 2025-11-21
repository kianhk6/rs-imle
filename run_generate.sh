#!/bin/bash
cd /home/kha98/Desktop/rs-imle
python /home/kha98/Desktop/rs-imle/generate_with_checkpoint.py \
    --checkpoint_path /home/kha98/Desktop/rs-imle/runs/nibi_models/best_fid-model.th \
    --condition_path /home/kha98/Desktop/flow-model-chirag/output_flow/flow-ffhq-debugfm/sample/4000_x0.pt \
    --output_path ./generated_samples_conditional.png \
    --num_images_visualize 10 \
    --num_rows_visualize 5 \
    --dataset ffhq_256 \
    --data_root /home/kha98/Desktop/rs-imle/dataset/ffhq/clean \
    --desc generate_test_conditional



