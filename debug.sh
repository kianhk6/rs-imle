#!/bin/bash

# The SBATCH directives must appear before any executable line in this script.

#SBATCH --time=0-00:15:0  # Time: D-H:M:S
#SBATCH --account=rrg-keli # Account
#SBATCH --mem=40G           # Memory in total
#SBATCH --nodes=1          # Number of nodes requested.
#SBATCH --cpus-per-task=6 # Number of cores per task.
#SBATCH --gres=gpu:h100:1 # 32G V100
##SBATCH --gres=gpu:p100l:4 # 16G P100 -- this line is commented out

# Uncomment this to have Slurm cd to a directory before running the script.
# You can also just run the script from the directory you want to be in.
# Change the folder below to your code directory
#SBATCH -D /home/kianhk6/projects/def-keli/kianhk6/rs-imle

# Uncomment to control the output files. By default stdout and stderr go to
# the same place, but if you use both commands below they'll be split up.
# %N is the hostname (if used, will create output(s) per node).
# %j is jobid.
#SBATCH --output=/home/kianhk6/projects/def-keli/kianhk6/rs-imle/debug
##SBATCH -e slurm.%N.%j.err    # STDERR

# Below sets the email notification, swap to your email to receive notifications
#SBATCH --mail-user=kha98@sfu.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Print some info for context.
pwd
hostname
date
    
echo "Starting job..."

# module load StdEnv/2020 (not needed)

module load python/3.10 
module load cuda
module load scipy-stack
module load openblas/0.3.24

# Configure FlexiBLAS to use OpenBLAS backend (matches local machine setup)
export FLEXIBLAS=OPENBLAS

source /home/kianhk6/projects/def-keli/kianhk6/venvs/rs-imle/bin/activate

# 

# Python will buffer output of your script unless you set this.
# If you're not using python, figure out how to turn off output
# buffering when stdout is a file, or else when watching your output
# script you'll only get updated every several lines printed.
# export PYTHONUNBUFFERED=1
# export COMET_API_KEY=esx5iX53IbgtEtr4Zj1tkxpYB
# export COMET_MODE=offline
# export COMET_OFFLINE_DIRECTORY=/home/kianhk6/projects/def-keli/kianhk6/rs-imle/comet/offline_$(date +%Y%m%d_%H%M%S)
# # Suppress connection warnings in offline mode
# export COMET_LOGGING_CONSOLE=WARNING
# mkdir -p "$COMET_OFFLINE_DIRECTORY"
#pip download -i https://test.pypi.org/simple/ dciknn-cuda==0.1.15

# python train.py --hps fewshot \
#     --save_dir ./new-vanilla-results/flowers-10ff-loadup-4000/ \
#     --data_root ./datasets/flowers/ \
#     --change_coef 0.02 \
#     --force_factor 10 \
#     --imle_staleness 5 \
#     --imle_force_resample 10  \
#     --lr 0.00005 \
#     --iters_per_ckpt 25000 --iters_per_images 5000 --iters_per_save 1000 \
#     --comet_api_key '2SDNAxxWevz4p6SThRTEM2KlD' \
#     --comet_name 'flowers-10ff-loadup-4000' \
#     --num_images_visualize 10 \
#     --num_rows_visualize 5 \
#     --imle_batch 32 \
#     --imle_db_size 512 \
#     --use_comet True \
#     --search_type 'l2' \
#     --use_adaptive False \
#     --n_batch 12 \
#     --lr_decay_iters 1000 \
#     --num_epochs 5000 \
#     --use_snoise False \
#     --fid_freq 100 \
#     --restore_path ./new-vanilla-results/flowers100-rerun-more-checkpoints/train/iter-4000-model.th

#     # --restore_path ./new-vanilla-results/flowers-10ff-earlier-loadup/train/latest-model.th \
#     # --restore_ema_path ./new-vanilla-results/flowers-10ff-earlier-loadup/train/latest-model-ema.th \
#     # --restore_optimizer_path ./new-vanilla-results/flowers-10ff-earlier-loadup/train/latest-opt.th \
#     # --restore_log_path ./new-vanilla-results/flowers-10ff-earlier-loadup/train/log.jsonl \
#     # --restore_scheduler_path ./new-vanilla-results/flowers-10ff-earlier-loadup/train/latest-sched.th \
#     # --comet_experiment_key '29c7547e85554737b1315ecac27b01bb' 

#     # --restore_path ./transfer/100flowers/best_fid-model-ema.th

cp /home/kianhk6/projects/def-keli/kianhk6/rs-imle/clean-fid/inception-2015-12-05.pt /tmp/ 

python /home/kianhk6/projects/def-keli/kianhk6/rs-imle/train.py --hps fewshot \
    --save_dir /home/kianhk6/projects/def-keli/kianhk6/rs-imle/runs/debug \
    --condition_path /home/kianhk6/projects/def-keli/kianhk6/rs-imle/dataset/ffhq/x0_10_seed_9.pt \
    --data_root /home/kianhk6/projects/def-keli/kianhk6/rs-imle/dataset/ffhq/10_steps_seed_9 \
    --lr 0.0002 \
    --force_factor 20 \
    --imle_staleness 5 \
    --imle_force_resample 5 \
    --comet_api_key 'esx5iX53IbgtEtr4Zj1tkxpYB' \
    --comet_name 'debug'  \
    --use_comet True \
    --use_snoise False \
    --imle_db_size 2000 \
    --imle_batch 50 \
    --conditional_force_faiss False \
    --restore_path /home/kianhk6/projects/def-keli/kianhk6/rs-imle/prev-model/5-steps-factor-20/best_fid-model.th \
    --restore_ema_path /home/kianhk6/projects/def-keli/kianhk6/rs-imle/prev-model/5-steps-factor-20/best_fid-model-ema.th 


# Wait for the background process to finish before exiting the script.
wait
date