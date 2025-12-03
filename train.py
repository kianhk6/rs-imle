import os
import time

from comet_ml import Experiment, ExistingExperiment
import imageio
import numpy as np
import torch
import torch.nn as nn
from cleanfid import fid
from torch.utils.data import DataLoader, TensorDataset

from data import set_up_data
from helpers.imle_helpers import backtrack, reconstruct
from helpers.train_helpers import (load_imle, load_opt, save_latents,
                                   save_latents_latest, save_model,
                                   save_snoise, set_up_hyperparams, update_ema)
from helpers.utils import ZippedDataset, get_cpu_stats_over_ranks
from helpers.seed_utils import seed_worker, get_generator
from metrics.ppl import calc_ppl
from metrics.ppl_uniform import calc_ppl_uniform
from sampler import Sampler
from visual.generate_rnd import generate_rnd
from visual.generate_rnd_nn import generate_rnd_nn
from visual.generate_sample_nn import generate_sample_nn
from visual.interpolate import random_interp
from visual.nn_interplate import nn_interp
from visual.spatial_visual import spatial_vissual
from visual.utils import (generate_and_save, generate_for_NN,
                          generate_images_initial,
                          get_sample_for_visualization,
                          delete_content_of_dir)
from helpers.improved_precision_recall import compute_prec_recall
from diffusers.models import AutoencoderKL

# To Do: this one is for the selected latents 
def generate_visualization_grid(batch_noise, teacher_images, teacher_model, teacher_vae, imle, H, latent_size, iteration, experiment=None, save_dir=None, rows=10, cols=10):
    """
    Generate a visualization grid comparing teacher and student outputs.
    
    Grid layout:
    - Row 0: teacher outputs on current batch noise (plus extra random noise if needed)
    - Row 1: student outputs on the SAME noises as row 0
    - Rows 2-9: student outputs on random noise only (teacher not used)
    
    Args:
        batch_noise: Current training batch noise [B, 4, H//8, W//8]
        teacher_images: Teacher outputs for batch_noise [B, H, W, 3] uint8
        teacher_model: Teacher model
        teacher_vae: Teacher VAE
        imle: Student model
        H: Hyperparameters
        latent_size: Latent size (H.image_size // 8)
        iteration: Current iteration number
        experiment: Comet experiment object (optional)
        save_dir: Directory to save image (optional)
        rows: Number of rows in grid (default 10)
        cols: Number of columns in grid (default 10)
    
    Returns:
        grid_image: numpy array of the visualization grid
    """
    import numpy as np
    import imageio
    import copy
    from torchdyn.core import NeuralODE
    
    # How many columns use the current training batch noise
    num_main = min(cols, batch_noise.shape[0])
    
    # 1) Current batch noise → teacher (already computed as teacher_images)
    teacher_main = teacher_images[:num_main]  # [num_main, H, W, 3] uint8
    
    # 2) Current batch noise → student
    #    Student takes noise as latents, unconditional (no conditions, no snoise)
    #    Model outputs in [-1, 1] range, convert to [0, 255]
    noise_main = batch_noise[:num_main]  # [num_main, 4, H//8, W//8]
    px_z_main = imle(noise_main, None).permute(0, 2, 3, 1)  # [B, H, W, C]
    xhat_main = (px_z_main + 1.0) * 127.5
    xhat_main = xhat_main.detach().cpu().numpy()
    student_main = np.minimum(np.maximum(0.0, xhat_main), 255.0).astype(np.uint8)
    
    teacher_row_images = teacher_main
    student_row_images = student_main
    
    # 3) If we want more columns in the first two rows, use fresh random noise
    # so that teacher and student see the same extra noises.
    if cols > num_main:
        rand_n = cols - num_main
        rand_noise = torch.randn(
            rand_n, 4, latent_size, latent_size, device="cuda"
        )
        
        # Teacher: rand_noise → images
        teacher_copy_viz = copy.deepcopy(teacher_model)
        node_viz = NeuralODE(
            teacher_copy_viz, solver="euler", sensitivity="adjoint"
        )
        traj_viz = node_viz.trajectory(
            rand_noise,
            t_span=torch.linspace(
                0,
                1,
                getattr(H, "teacher_resample_steps", 20),
                device="cuda",
            ),
        )
        teacher_latents_viz = traj_viz[-1, :].view(rand_noise.shape)
        teacher_images_viz = teacher_vae.decode(
            teacher_latents_viz / teacher_vae.config.scaling_factor
        ).sample
        teacher_images_viz = (teacher_images_viz + 1) / 2
        teacher_images_viz = (
            torch.clamp(teacher_images_viz, 0, 1) * 255
        )
        teacher_images_viz = (
            teacher_images_viz.byte()
            .permute(0, 2, 3, 1)
            .cpu()
        )
        
        # Student: rand_noise → images (same unconditional mapping as training)
        px_z_rand = imle(rand_noise, None).permute(0, 2, 3, 1)  # [B, H, W, C]
        xhat_rand = (px_z_rand + 1.0) * 127.5
        xhat_rand = xhat_rand.detach().cpu().numpy()
        student_images_rand = np.minimum(np.maximum(0.0, xhat_rand), 255.0).astype(np.uint8)
        
        teacher_row_images = torch.cat(
            [teacher_row_images, teacher_images_viz], dim=0
        )
        student_row_images = np.concatenate(
            [student_row_images, student_images_rand], axis=0
        )
    
    # Ensure exactly `cols` images in the first two rows
    teacher_row_images = teacher_row_images[:cols]
    student_row_images = student_row_images[:cols]
    
    teacher_np = teacher_row_images.numpy()
    student_np = student_row_images
    
    # Build grid rows
    rows_np = []
    
    # Row 0: teacher
    row0 = np.concatenate([teacher_np[i] for i in range(cols)], axis=1)
    rows_np.append(row0)
    
    # Row 1: student on same noises
    row1 = np.concatenate([student_np[i] for i in range(cols)], axis=1)
    rows_np.append(row1)
    
    # Rows 2–9: random noise input to the STUDENT only (no teacher)
    for _ in range(2, rows):
        rand_noise_row = torch.randn(
            cols, 4, latent_size, latent_size, device="cuda"
        )
        px_z_row = imle(rand_noise_row, None).permute(0, 2, 3, 1)  # [B, H, W, C]
        xhat_row = (px_z_row + 1.0) * 127.5
        xhat_row = xhat_row.detach().cpu().numpy()
        student_row_np = np.minimum(np.maximum(0.0, xhat_row), 255.0).astype(np.uint8)
        row = np.concatenate(
            [student_row_np[i] for i in range(cols)], axis=1
        )
        rows_np.append(row)
    
    # Stack all rows vertically → rows x cols grid
    grid_image = np.concatenate(rows_np, axis=0)
    
    # # Save image if save_dir is provided
    # if save_dir is not None:
    #     imageio.imwrite(f"{save_dir}/samples_{iteration}.png", grid_image)
    
    # Log to experiment if provided
    if experiment is not None:
        experiment.log_image(grid_image, name=f"samples_{iteration}")
    
    return grid_image


def training_step_imle(H, n, targets, latents, snoise, imle, ema_imle, optimizer, loss_fn, condition_data=None, batch_conditions=None, batch_condition_indices=None):
    t0 = time.time()
    imle.zero_grad()

    cur_batch_latents = latents
    # print("loss inference: ", batch_conditions)
    # if batch_condition_indices is not None:
    #     if torch.is_tensor(batch_condition_indices):
    #         print("Condition indices passed to training_step_imle:", batch_condition_indices.tolist())
    #     else:
    #         print("Condition indices passed to training_step_imle:", batch_condition_indices)
    # Pass condition data to the model if available
    if batch_conditions is not None:
        if hasattr(H, 'debug_return_condition') and H.debug_return_condition:
            px_z, debug_info = imle(cur_batch_latents, snoise, condition_data=batch_conditions, return_condition=True)
        else:
            px_z = imle(cur_batch_latents, snoise, condition_data=batch_conditions)
    else:
        px_z = imle(cur_batch_latents, snoise)
    
    # Compute loss - pass latents for teacher loss computation if enabled
    # Note: latents parameter is already the selected_latents from sampler for this batch
    if getattr(H, 'use_teacher_loss', False):
        # Pass the latents (selected noise codes z) to loss function for teacher loss computation
        # Ensure latents are on CUDA for teacher model inference
        loss = loss_fn(px_z, targets.permute(0, 3, 1, 2), selected_latents=cur_batch_latents.cuda())
    else:
        loss = loss_fn(px_z, targets.permute(0, 3, 1, 2))
    
    loss.backward()
    optimizer.step()
    if ema_imle is not None:
        update_ema(imle, ema_imle, H.ema_rate)

    stats = get_cpu_stats_over_ranks(dict(loss_nans=0, loss=loss))
    stats.update(skipped_updates=0, iter_time=time.time() - t0, grad_norm=0)
    return stats


def train_loop_imle(H, data_train, data_valid, preprocess_fn, imle, ema_imle, logprint, experiment = None, condition_data = None):
    subset_len = len(data_train)
    if H.subset_len != -1:
        subset_len = H.subset_len
    for data_train in DataLoader(data_train, batch_size=subset_len):
        data_train = TensorDataset(data_train[0])
        break

    if condition_data is not None:
        for condition_data in DataLoader(condition_data, batch_size=subset_len):
            condition_data = TensorDataset(condition_data[0])
            break
    else:
        print("No condition data provided, training without conditions")

    optimizer, scheduler, _, iterate, starting_epoch = load_opt(H, imle, logprint)

    print("Starting epoch: ", starting_epoch)
    print("Starting iteration: ", iterate)

    stats = []
    H.ema_rate = torch.as_tensor(H.ema_rate)

    subset_len = H.subset_len
    if subset_len == -1:
        subset_len = len(data_train)
    
    # Load teacher model and VAE if teacher resampling OR teacher loss is enabled
    teacher_model = None
    teacher_vae = None
    
    use_teacher_for_resample = hasattr(H, 'use_teacher_resample') and H.use_teacher_resample
    use_teacher_for_loss = hasattr(H, 'use_teacher_loss') and H.use_teacher_loss
    
    if use_teacher_for_resample or use_teacher_for_loss:
        purposes = []
        if use_teacher_for_resample:
            purposes.append("dynamic resampling")
        if use_teacher_for_loss:
            purposes.append("teacher loss")
        print(f"Loading teacher model for {' and '.join(purposes)}...")
        import sys
        import importlib.util
        import copy
        from diffusers import AutoencoderKL
        
        # Load teacher UNet by adding the package to sys.path and importing normally
        sys.path.insert(0, '/home/kha98/Desktop/conditional-flow-matching')
        try:
            from torchcfm.models.unet.unet import UNetModelWrapper as TeacherUNet
        finally:
            # Remove from path after loading
            sys.path.remove('/home/kha98/Desktop/conditional-flow-matching')
        
        teacher_model = TeacherUNet(
            dim=(4, H.image_size // 8, H.image_size // 8),
            num_res_blocks=2,
            num_channels=128,
            channel_mult=[1, 2, 2, 2],
            num_heads=4,
            num_head_channels=64,
            attention_resolutions="16",
            dropout=0.1,
        ).to('cuda')
        
        # Load checkpoint
        teacher_checkpoint_path = getattr(H, 'teacher_checkpoint_path', 
            '/home/kha98/Desktop/flow-model-chirag/output_flow/flow-ffhq-debugfm/fm_cifar10_weights_step_84000.pt')
        ckpt = torch.load(teacher_checkpoint_path, map_location='cuda')
        teacher_model.load_state_dict(ckpt['ema_model'])
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        
        # Load VAE for decoding
        teacher_vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to('cuda')
        teacher_vae.eval()
        for param in teacher_vae.parameters():
            param.requires_grad = False
        
        print("Teacher model and VAE loaded successfully!")
        
        # Print teacher loss configuration if enabled
        if use_teacher_for_loss:
            teacher_lambda = getattr(H, 'teacher_loss_lambda', 1.0)
            print(f"✓ Teacher loss enabled with lambda={teacher_lambda}")
            print(f"  Loss = L_imle + {teacher_lambda} * L_teacher")
            print(f"  where L_teacher = (1/|S|) * sum || G_imle(z) - G_flow(z) ||^2")
        
        # Check if we should generate initial data
        if hasattr(H, 'teacher_generate_initial_data') and H.teacher_generate_initial_data:
            print("Will generate INITIAL dataset from teacher at epoch 0, then regenerate during resampling")
        else:
            print("Will use existing data/conditions initially, then regenerate during resampling")
        
        # Validate teacher_force_resample parameter
        if hasattr(H, 'teacher_force_resample') and H.teacher_force_resample is not None:
            if H.teacher_force_resample % H.imle_force_resample != 0:
                raise ValueError(
                    f"teacher_force_resample ({H.teacher_force_resample}) must be a multiple of "
                    f"imle_force_resample ({H.imle_force_resample})"
                )
            print(f"Teacher force resample enabled: will renew dataset every {H.teacher_force_resample} epochs")
    
    print(f"Initializing Sampler with subset_len={subset_len}...", flush=True)
    
    # Check if using teacher mode with placeholder data
    use_teacher_init = (hasattr(H, 'teacher_generate_initial_data') and 
                        H.teacher_generate_initial_data and
                        hasattr(H, 'use_teacher_resample') and 
                        H.use_teacher_resample)
    if condition_data is not None:
        if use_teacher_init:
            # Skip condition configuration for placeholder data (will configure after teacher generation)
            print(f"  Skipping condition configuration (placeholder data)", flush=True)
            sampler = Sampler(H, subset_len, preprocess_fn, 
                            teacher_model=teacher_model, 
                            teacher_vae=teacher_vae)
        else:
            print(f"  With condition_data (size: {len(condition_data)})", flush=True)
            sampler = Sampler(
                H,
                subset_len,
                preprocess_fn,
                condition_config={"condition_tensor": condition_data},
                teacher_model=teacher_model,
                teacher_vae=teacher_vae,
            )
    else:
        print(f"  Without condition_data", flush=True)
        sampler = Sampler(H, subset_len, preprocess_fn, 
                         teacher_model=teacher_model, 
                         teacher_vae=teacher_vae)
    print(f"Sampler initialized successfully!", flush=True)

    last_updated = torch.zeros(subset_len, dtype=torch.int16).cuda()
    times_updated = torch.zeros(subset_len, dtype=torch.int8).cuda()
    change_thresholds = torch.empty(subset_len).cuda()
    change_thresholds[:] = H.change_threshold
    best_fid = 100000
    epoch = starting_epoch - 1

    print(f"Starting training loop...", flush=True)
    for split_ind, split_x_tensor in enumerate(DataLoader(data_train, batch_size=subset_len, pin_memory=True)):
        print(f"Processing split {split_ind}, data shape: {split_x_tensor[0].shape}", flush=True)
        split_x_tensor = split_x_tensor[0].contiguous()
        split_x = TensorDataset(split_x_tensor)
        
        # Skip projection initialization if using placeholder data (will init after teacher generation)
        if use_teacher_init and epoch == starting_epoch - 1:
            print(f"Skipping projection initialization (placeholder data - will init after teacher generation)", flush=True)
        else:
            print(f"Initializing projection...", flush=True)
            sampler.init_projection(split_x_tensor)
            print(f"Projection initialized!", flush=True)
            
        print(f"Getting visualization batch...", flush=True)
        viz_batch_original, _ = get_sample_for_visualization(split_x, preprocess_fn, H.num_images_visualize, H.dataset)
        print(f"Split {split_ind} preparation complete!", flush=True)

        # Extract corresponding split from condition_data
        split_condition_data = None
        if condition_data is not None:
            # Get the same indices that correspond to this split
            start_idx = split_ind * subset_len
            end_idx = min(start_idx + subset_len, len(condition_data))
            split_indices = torch.arange(start_idx, end_idx)
            
            # Extract the condition data for this split
            split_conditions = []
            for idx in split_indices:
                condition_sample = condition_data[idx]
                if isinstance(condition_sample, tuple):
                    split_conditions.append(condition_sample[0])
                else:
                    split_conditions.append(condition_sample)
            split_condition_data = torch.stack(split_conditions).cuda()
            
            print(f'Split {split_ind}: data shape {split_x_tensor.shape}, condition shape {split_condition_data.shape}')

        print('Outer batch - {}'.format(split_ind, len(split_x)))
        
        # Generate initial dataset from teacher if requested (epoch 0)
        if (hasattr(H, 'teacher_generate_initial_data') and H.teacher_generate_initial_data and 
            teacher_model is not None and epoch == starting_epoch - 1):
            print(f"\n{'='*70}", flush=True)
            
            # Determine the correct teacher seed based on whether we're restoring from checkpoint
            restoring_from_checkpoint = (H.restore_path is not None and os.path.isfile(H.restore_path))
            
            if restoring_from_checkpoint and starting_epoch > 0:
                # When restoring, calculate which epoch had the last teacher resample
                # This ensures we regenerate the exact same dataset the model was trained on
                teacher_force_resample = getattr(H, 'teacher_force_resample', 20)
                last_teacher_epoch = (starting_epoch // teacher_force_resample) * teacher_force_resample
                teacher_seed = -1
                print(f"RESTORING: Regenerating teacher dataset from last resample (Epoch {last_teacher_epoch})", flush=True)
                print(f"  Starting epoch: {starting_epoch}, Teacher resample interval: {teacher_force_resample}", flush=True)
                print(f"  Using teacher seed: {teacher_seed}", flush=True)
            else:
                # Fresh training: generate initial dataset at epoch -1 (before epoch 0)
                teacher_seed = -1
                print(f"GENERATING INITIAL DATASET FROM TEACHER (Epoch {epoch})", flush=True)
                print(f"  Using teacher seed: {teacher_seed}", flush=True)
            
            print(f"{'='*70}", flush=True)
            
            # Generate new data and conditions from teacher with deterministic seed
            new_data, new_conditions = sampler.generate_new_data_from_teacher(
                num_teacher_steps=getattr(H, 'teacher_resample_steps', 20),
                seed=teacher_seed
            )
            
            if new_data is not None and new_conditions is not None:
                print(f"Generated initial dataset: {new_data.shape}", flush=True)
                print(f"Updating training data with teacher-generated samples...", flush=True)

                # Update global dataset or just this split depending on what teacher returned.
                # `data_train` is a TensorDataset; its backing tensor is at data_train.tensors[0]
                try:
                    global_dataset = data_train.tensors[0]
                except Exception:
                    global_dataset = None

                n_new = new_data.shape[0]
                n_split = split_x_tensor.shape[0]
                n_global = global_dataset.shape[0] if global_dataset is not None else None

                start_idx = split_ind * subset_len
                end_idx = min(start_idx + subset_len, n_global) if n_global is not None else n_split

                # Case 1: teacher returned a full-dataset update
                if n_global is not None and n_new == n_global:
                    # Replace entire dataset and then refresh this split
                    global_dataset[:] = new_data
                    split_x_tensor[:] = global_dataset[start_idx:end_idx]
                # Case 2: teacher returned exactly the split-size
                elif n_new == n_split:
                    split_x_tensor[:] = new_data
                    if global_dataset is not None:
                        global_dataset[start_idx:end_idx] = new_data
                # Case 3: teacher returned more than the split but not equal to global - try to fill global when possible
                elif n_global is not None and n_new >= n_global:
                    global_dataset[:] = new_data[:n_global]
                    split_x_tensor[:] = global_dataset[start_idx:end_idx]
                else:
                    # Fallback: truncate or pad the new data to fit the current split
                    split_x_tensor[:] = new_data[:n_split]
                    if global_dataset is not None and n_new >= (end_idx - start_idx):
                        global_dataset[start_idx:end_idx] = new_data[:(end_idx - start_idx)]

                # Update the TensorDataset and re-init projection with the (possibly) new data
                split_x = TensorDataset(split_x_tensor)
                print(f"Initializing projections with real teacher-generated data...", flush=True)
                sampler.init_projection(split_x_tensor)
                print(f"Projections initialized successfully!", flush=True)
                
                # Configure conditions with real teacher-generated data
                if condition_data is not None:
                    print(f"Configuring conditions with real teacher-generated data...", flush=True)
                    sampler.configure_conditions(sz=subset_len, condition_tensor=condition_data)
                    print(f"Conditions configured successfully!", flush=True)
                    
                    # Update split_condition_data with the new teacher-generated conditions
                    print(f"Updating split_condition_data with teacher-generated conditions...", flush=True)
                    split_conditions = []
                    for idx in split_indices:
                        condition_sample = condition_data[idx]
                        if isinstance(condition_sample, tuple):
                            split_conditions.append(condition_sample[0])
                        else:
                            split_conditions.append(condition_sample)
                    split_condition_data = torch.stack(split_conditions).cuda()
                    print(f"split_condition_data updated! Shape: {split_condition_data.shape}", flush=True)
                
                print(f"Getting visualization batch after teacher generation...", flush=True)
                viz_batch_original, _ = get_sample_for_visualization(split_x, preprocess_fn, H.num_images_visualize, H.dataset)
                print(f"Visualization batch obtained!", flush=True)

                print(f"Updating split_condition_data with teacher conditions...", flush=True)
                # Update conditions similarly
                if split_condition_data is not None:
                    try:
                        cond_global = condition_data.tensors[0]
                    except Exception:
                        cond_global = None

                    n_cond_new = new_conditions.shape[0]
                    if cond_global is not None and n_cond_new == cond_global.shape[0]:
                        cond_global[:] = new_conditions.cpu()
                        split_condition_data[:] = cond_global[start_idx:end_idx].cuda()
                    elif n_cond_new == n_split:
                        split_condition_data[:] = new_conditions.cuda()
                        if cond_global is not None:
                            cond_global[start_idx:end_idx] = new_conditions.cpu()
                    elif cond_global is not None and n_cond_new >= cond_global.shape[0]:
                        cond_global[:] = new_conditions[:cond_global.shape[0]].cpu()
                        split_condition_data[:] = cond_global[start_idx:end_idx].cuda()
                    else:
                        split_condition_data[:] = new_conditions[:n_split].cuda()

                    if cond_global is not None:
                        print(f"Updated initial conditions: {new_conditions.shape}", flush=True)
                else:
                    print(f"split_condition_data is None, skipping condition update", flush=True)

                print(f"{'='*70}\n", flush=True)
                print(f"Initial dataset generation and setup complete! Starting training epochs...", flush=True)

        print(f"Entering main training loop (epoch={epoch}, num_epochs={H.num_epochs})...", flush=True)
        while (epoch < H.num_epochs):
            
            epoch += 1
            print(f"\n{'*'*70}", flush=True)
            print(f"Starting Epoch {epoch}", flush=True)
            print(f"{'*'*70}\n", flush=True)
            last_updated[:] = last_updated + 1

            print(f"Calculating existing distances for {len(split_x_tensor)} samples...", flush=True)
            if split_condition_data is not None:
                print(f"  With conditions (shape: {split_condition_data.shape})", flush=True)
                sampler.selected_dists[:] = sampler.calc_dists_existing(
                    split_x_tensor,
                    imle,
                    dists=sampler.selected_dists,
                    conditions=(split_condition_data, split_indices.cuda())
                )
            else:
                print(f"  Without conditions", flush=True)
                sampler.selected_dists[:] = sampler.calc_dists_existing(split_x_tensor, imle, dists=sampler.selected_dists)
            print(f"Distance calculation complete!", flush=True)
            print(f"Determining which samples to update...", flush=True)
            dists_in_threshold = sampler.selected_dists < change_thresholds
            updated_enough = last_updated >= H.imle_staleness
            updated_too_much = last_updated >= H.imle_force_resample
            in_threshold = torch.logical_and(dists_in_threshold, updated_enough)

            # everything gets updated when adapted is false
            if(H.use_adaptive):
                all_conditions = torch.logical_or(in_threshold, updated_too_much)
            else:
                all_conditions = updated_too_much
                
            # all_conditions = torch.logical_or(in_threshold, updated_too_much)
            to_update = torch.nonzero(all_conditions, as_tuple=False).squeeze(1)
            print(f"Samples to update: {len(to_update)}/{len(split_x_tensor)}", flush=True)

            if (epoch == starting_epoch):
                if os.path.isfile(str(H.restore_latent_path)):
                    latents = torch.load(H.restore_latent_path)
                    sampler.selected_latents[:] = latents[:]
                    for x in DataLoader(split_x, batch_size=H.num_images_visualize, pin_memory=True):
                        break
                    batch_slice = slice(0, x[0].size()[0])
                    latents = sampler.selected_latents[batch_slice]
                    with torch.no_grad():
                        snoise = [s[batch_slice] for s in sampler.selected_snoise]
                        generate_for_NN(sampler, x[0], latents, snoise, viz_batch_original.shape, imle,
                            f'{H.save_dir}/NN-samples_{epoch}-{split_ind}-imle.png', logprint, experiment=experiment)
                    print('loaded latest latents')

                if os.path.isfile(str(H.restore_latent_path)):
                    threshold = torch.load(H.restore_threshold_path)
                    change_thresholds[:] = threshold[:]
                    print('loaded thresholds', torch.mean(change_thresholds))
                else:
                    to_update = sampler.entire_ds


            change_thresholds[to_update] = sampler.selected_dists[to_update].clone() * (1 - H.change_coef)
            
            print(f"Running IMLE sampling for {len(to_update)} samples...", flush=True)
            # Pass the corresponding condition data split to imle_sample_force
            if split_condition_data is not None:
                new_data, new_conditions = sampler.imle_sample_force(split_x_tensor, imle, to_update, condition_data=split_condition_data, epoch=epoch)
            else:
                new_data, new_conditions = sampler.imle_sample_force(split_x_tensor, imle, to_update, epoch=epoch)
            print(f"IMLE sampling complete!", flush=True)
            
            # Update dataset and conditions if teacher resampling generated new data
            if new_data is not None and new_conditions is not None:
                print(f"Updating dataset with new teacher-generated data: {new_data.shape}")

                try:
                    global_dataset = data_train.tensors[0]
                except Exception:
                    global_dataset = None

                n_new = new_data.shape[0]
                n_split = split_x_tensor.shape[0]
                n_global = global_dataset.shape[0] if global_dataset is not None else None

                start_idx = split_ind * subset_len
                end_idx = min(start_idx + subset_len, n_global) if n_global is not None else n_split

                if n_global is not None and n_new == n_global:
                    global_dataset[:] = new_data
                    split_x_tensor[:] = global_dataset[start_idx:end_idx]
                elif n_new == n_split:
                    split_x_tensor[:] = new_data
                    if global_dataset is not None:
                        global_dataset[start_idx:end_idx] = new_data
                elif n_global is not None and n_new >= n_global:
                    global_dataset[:] = new_data[:n_global]
                    split_x_tensor[:] = global_dataset[start_idx:end_idx]
                else:
                    split_x_tensor[:] = new_data[:n_split]
                    if global_dataset is not None and n_new >= (end_idx - start_idx):
                        global_dataset[start_idx:end_idx] = new_data[:(end_idx - start_idx)]

                split_x = TensorDataset(split_x_tensor)
                sampler.init_projection(split_x_tensor)
                viz_batch_original, _ = get_sample_for_visualization(split_x, preprocess_fn, H.num_images_visualize, H.dataset)

                if split_condition_data is not None:
                    try:
                        cond_global = condition_data.tensors[0]
                    except Exception:
                        cond_global = None

                    n_cond_new = new_conditions.shape[0]
                    if cond_global is not None and n_cond_new == cond_global.shape[0]:
                        cond_global[:] = new_conditions.cpu()
                        split_condition_data[:] = cond_global[start_idx:end_idx].cuda()
                    elif n_cond_new == n_split:
                        split_condition_data[:] = new_conditions.cuda()
                        if cond_global is not None:
                            cond_global[start_idx:end_idx] = new_conditions.cpu()
                    elif cond_global is not None and n_cond_new >= cond_global.shape[0]:
                        cond_global[:] = new_conditions[:cond_global.shape[0]].cpu()
                        split_condition_data[:] = cond_global[start_idx:end_idx].cuda()
                    else:
                        split_condition_data[:] = new_conditions[:n_split].cuda()

                    if cond_global is not None:
                        print(f"Updated conditions: {new_conditions.shape}")
            

            to_update = to_update.cpu()
            last_updated[to_update] = 0
            times_updated[to_update] = times_updated[to_update] + 1

            # save_latents_latest(H, split_ind, sampler.selected_latents)
            # save_latents_latest(H, split_ind, change_thresholds, name='threshold_latest')

            if to_update.shape[0] >= H.num_images_visualize + 8:
                vis_idx = to_update[:H.num_images_visualize]
                latents = sampler.selected_latents[vis_idx]
                cond_vis = None
                if split_condition_data is not None:
                    cond_vis = split_condition_data[vis_idx]
                    with torch.no_grad():
                        generate_for_NN(
                            sampler,
                            split_x_tensor[vis_idx],
                            latents,
                            [s[vis_idx] for s in sampler.selected_snoise],
                            viz_batch_original.shape,
                            imle,
                            f'{H.save_dir}/NN-samples_{epoch}-imle.png',
                            logprint,
                            condition_data=cond_vis,
                            experiment=experiment,
                        )
                else:
                        generate_for_NN(sampler, split_x_tensor[to_update[:H.num_images_visualize]], latents,
                            [s[to_update[:H.num_images_visualize]] for s in sampler.selected_snoise],
                            viz_batch_original.shape, imle,
                            f'{H.save_dir}/NN-samples_{epoch}-imle.png', logprint, experiment=experiment)

            comb_dataset = ZippedDataset(split_x, TensorDataset(sampler.selected_latents))
            data_loader = DataLoader(comb_dataset, batch_size=H.n_batch, pin_memory=True, shuffle=False, num_workers=0, persistent_workers=False)

            start_time = time.time()

            for cur, indices in data_loader:
                x = cur[0]
                latents = cur[1][0]
                _, target = preprocess_fn(x)
                
                # Get condition data for this batch
                batch_conditions = None
                if condition_data is not None:
                    # condition_data is now a TensorDataset, so we access it like data_train
                    batch_conditions = []
                    for idx in indices:
                        condition_sample = condition_data[idx]
                        if isinstance(condition_sample, tuple):
                            batch_conditions.append(condition_sample[0])  # Get the tensor from the tuple
                        else:
                            batch_conditions.append(condition_sample)
                    batch_conditions = torch.stack(batch_conditions).cuda()
                
                cur_snoise = [s[indices] for s in sampler.selected_snoise]

                for i in range(len(H.res)):
                    cur_snoise[i].zero_()

                if condition_data is not None: 
                    stat = training_step_imle(
                        H,
                        target.shape[0],
                        target,
                        latents,
                        cur_snoise,
                        imle,
                        ema_imle,
                        optimizer,
                        sampler.calc_loss,
                        condition_data,
                        batch_conditions,
                        batch_condition_indices=indices,
                    )
                    stats.append(stat)
                else:
                    stat = training_step_imle(H, target.shape[0], target, latents, cur_snoise, imle, ema_imle, optimizer, sampler.calc_loss)
                    stats.append(stat)



                # Only step scheduler during warmup if using iteration-based scheduler
                use_dynamic_scheduler = getattr(H, 'use_teacher_resample', False) and getattr(H, 'teacher_force_resample', None) is not None
                if(iterate <= H.warmup_iters and not use_dynamic_scheduler):
                    # print("Warmup iteration: ", iterate)
                    scheduler.step()

                if iterate % H.iters_per_images == 0:
                    with torch.no_grad():
                        cond_vis2 = None
                        if split_condition_data is not None:
                            cond_vis2 = split_condition_data[0: H.num_images_visualize]
                            generate_images_initial(
                                H, sampler, viz_batch_original,
                                sampler.selected_latents[0: H.num_images_visualize],
                                [s[0: H.num_images_visualize] for s in sampler.selected_snoise],
                                viz_batch_original.shape, imle, ema_imle,
                                f'{iterate}.png', logprint, experiment,
                                condition_data=cond_vis2,
                                epoch=epoch,
                            )
                        else:
                            generate_images_initial(H, sampler, viz_batch_original,
                            sampler.selected_latents[0: H.num_images_visualize],
                            [s[0: H.num_images_visualize] for s in sampler.selected_snoise],
                            viz_batch_original.shape, imle, ema_imle,
                            f'{iterate}.png', logprint, experiment)

                iterate += 1
                if iterate % H.iters_per_save == 0:
                    fp = os.path.join(H.save_dir, 'latest')
                    logprint(f'Saving model@ {iterate} to {fp}')
                    save_model(fp, imle, ema_imle, optimizer, scheduler, H)
                    save_latents_latest(H, split_ind, sampler.selected_latents)
                    save_latents_latest(H, split_ind, change_thresholds, name='threshold_latest')

                if iterate % H.iters_per_ckpt == 0:
                    save_model(os.path.join(H.save_dir, f'iter-{iterate}'), imle, ema_imle, optimizer, scheduler, H)
                    save_latents(H, iterate, split_ind, sampler.selected_latents)
                    save_latents(H, iterate, split_ind, change_thresholds, name='threshold')
                    save_snoise(H, iterate, sampler.selected_snoise)

            print(f'Epoch {epoch} took {time.time() - start_time} seconds')

            # Step scheduler at end of epoch
            use_dynamic_scheduler = getattr(H, 'use_teacher_resample', False) and getattr(H, 'teacher_force_resample', None) is not None
            if use_dynamic_scheduler:
                # For CosineAnnealingWarmRestarts, step every epoch
                scheduler.step()
            elif iterate > H.warmup_iters:
                # For original StepLR, step after warmup
                scheduler.step()

            
            cur_dists = torch.empty([subset_len], dtype=torch.float32).cuda()
            cur_dists_lpips = torch.empty([subset_len], dtype=torch.float32).cuda()
            cur_dists_l2 = torch.empty([subset_len], dtype=torch.float32).cuda()

            if condition_data != None:
                cur_dists[:], cur_dists_lpips[:], cur_dists_l2[:] = sampler.calc_dists_existing(split_x_tensor, imle, 
                                                                                                dists=cur_dists,  
                                                                                                dists_lpips=cur_dists_lpips,
                                                                                                dists_l2=cur_dists_l2, 
                                                                                                logging=True,
                                                                                                conditions=split_condition_data,
                                                                                                expect_condition_indices=False)
            else:
                cur_dists[:], cur_dists_lpips[:], cur_dists_l2[:] = sampler.calc_dists_existing(split_x_tensor, imle, 
                                                                                dists=cur_dists,  
                                                                                dists_lpips=cur_dists_lpips,
                                                                                dists_l2=cur_dists_l2, 
                                                                                logging=True)
                                                                                
            metrics = {
                'mean_loss': torch.mean(cur_dists).item(),
                'std_loss': torch.std(cur_dists).item(),
                'max_loss': torch.max(cur_dists).item(),
                'min_loss': torch.min(cur_dists).item(),
                'mean_loss_lpips': torch.mean(cur_dists_lpips).item(),
                'std_loss_lpips': torch.std(cur_dists_lpips).item(),
                'max_loss_lpips': torch.max(cur_dists_lpips).item(),
                'min_loss_lpips': torch.min(cur_dists_lpips).item(),
                'mean_loss_l2': torch.mean(cur_dists_l2).item(),
                'std_loss_l2': torch.std(cur_dists_l2).item(),
                'max_loss_l2': torch.max(cur_dists_l2).item(),
                'min_loss_l2': torch.min(cur_dists_l2).item(),
                'total_excluded': sampler.total_excluded,
                'total_excluded_percentage': sampler.total_excluded_percentage,
            }

            if (epoch > 0 and epoch % H.fid_freq == 0):
                print("calculating fid")
                print("Learning rate: ", optimizer.param_groups[0]['lr'])
                if split_condition_data != None:
                    generate_and_save(H, imle, sampler, min(1000,subset_len * H.fid_factor), condition_data=split_condition_data)
                else:
                    generate_and_save(H, imle, sampler, min(1000,subset_len * H.fid_factor))

                real_dir = H.fid_real_dir if (hasattr(H, 'fid_real_dir') and H.fid_real_dir) else f'{H.data_root}/img'
                print(real_dir, f'{H.save_dir}/fid/')
                try:
                    cur_fid = fid.compute_fid(real_dir, f'{H.save_dir}/fid/', verbose=False)
                except Exception as e:
                    import traceback
                    logprint(f"[WARN] FID computation failed with unexpected error: {e}")
                    logprint(traceback.format_exc())
                    logprint("Skipping FID evaluation for this iteration.")
                    cur_fid = float('inf')

                if cur_fid < best_fid:
                    best_fid = cur_fid
                    # save models
                    fp = os.path.join(H.save_dir, 'best_fid')
                    logprint(f'Saving model best fid {best_fid} @ {iterate} to {fp}')
                    save_model(fp, imle, ema_imle, optimizer, scheduler, H)
                
                real_dir_prec_rec = H.fid_real_dir if (hasattr(H, 'fid_real_dir') and H.fid_real_dir) else f'{H.data_root}/img'
                precision, recall = compute_prec_recall(real_dir_prec_rec, f'{H.save_dir}/fid/')

                metrics['fid'] = cur_fid
                metrics['best_fid'] = best_fid
                metrics['precision'] = precision
                metrics['recall'] = recall
                
            
            if (to_update.shape[0] != 0):
                metrics['mean_loss_resample'] = torch.mean(cur_dists).item()
                metrics['std_loss_resample'] = torch.std(cur_dists).item()
                metrics['max_loss_resample'] = torch.max(cur_dists).item()
                metrics['min_loss_resample'] = torch.min(cur_dists).item()

            logprint(model=H.desc, type='train_loss', epoch=epoch, step=iterate, **metrics)

            if epoch % 50 == 0:
                with torch.no_grad():

                    cond_vis2 = None
                    if split_condition_data is not None:
                        cond_vis2 = split_condition_data[0: H.num_images_visualize]
                        generate_images_initial(
                            H, sampler, viz_batch_original,
                            sampler.selected_latents[0: H.num_images_visualize],
                            [s[0: H.num_images_visualize] for s in sampler.selected_snoise],
                            viz_batch_original.shape, imle, ema_imle,
                            f'{H.save_dir}/latest.png', logprint, experiment,
                            condition_data=cond_vis2, epoch=epoch,
                        )

                    else:
                        generate_images_initial(H, sampler, viz_batch_original,
                        sampler.selected_latents[0: H.num_images_visualize],
                        [s[0: H.num_images_visualize] for s in sampler.selected_snoise],
                        viz_batch_original.shape, imle, ema_imle,
                        f'{H.save_dir}/latest.png', logprint, experiment, epoch=epoch)                    


            
            if experiment is not None:
                experiment.log_metrics(metrics, epoch=epoch, step=iterate)

def train_loop_resample_every_batch(H, imle, ema_imle, logprint, experiment=None):
    """
    Simplified training loop for resample_every_batch mode.
    Every batch: generate n_batch noise → teacher generates n_batch samples → student learns from same noise.
    No dataset persistence, no epoch-based resampling, no conditions.
    """
    print("="*70)
    print("RESAMPLE_EVERY_BATCH MODE: Simplified training pipeline")
    print("="*70)
    print("Every batch:")
    print("  1. Generate n_batch noise")
    print("  2. Teacher generates n_batch samples from noise")
    print("  3. Student uses same noise to predict")
    print("  4. Calculate loss and update")
    print("="*70)
    
    # Automatically enable use_teacher_noise_as_input for this mode
    # This ensures the model is configured to accept noise as input
    if not hasattr(H, 'use_teacher_noise_as_input') or not H.use_teacher_noise_as_input:
        print("WARNING: resample_every_batch mode works best with use_teacher_noise_as_input=True")
        print("  The model should be configured to accept noise as input (unconditional UNet)")
        print("  Consider setting --use_teacher_noise_as_input True")
    
    # Load teacher model and VAE
    print("Loading teacher model...")
    import sys
    import copy
    from diffusers import AutoencoderKL
    from torchdyn.core import NeuralODE
    
    sys.path.insert(0, '/home/kha98/Desktop/conditional-flow-matching')
    try:
        from torchcfm.models.unet.unet import UNetModelWrapper as TeacherUNet
    finally:
        sys.path.remove('/home/kha98/Desktop/conditional-flow-matching')
    
    teacher_model = TeacherUNet(
        dim=(4, H.image_size // 8, H.image_size // 8),
        num_res_blocks=2,
        num_channels=128,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to('cuda')
    
    teacher_checkpoint_path = getattr(H, 'teacher_checkpoint_path', 
        '/home/kha98/Desktop/flow-model-chirag/output_flow/flow-ffhq-debugfm/fm_cifar10_weights_step_84000.pt')
    ckpt = torch.load(teacher_checkpoint_path, map_location='cuda')
    teacher_model.load_state_dict(ckpt['ema_model'])
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    teacher_vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to('cuda')
    teacher_vae.eval()
    for param in teacher_vae.parameters():
        param.requires_grad = False
    
    print("Teacher model and VAE loaded successfully!")
    
    # Setup optimizer with same scheduler behavior as IMLE
    # Simulates epochs: virtual_dataset_size / n_batch iterations = 1 epoch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR, StepLR, SequentialLR
    from helpers.train_helpers import linear_warmup
    
    optimizer = AdamW(imle.parameters(), weight_decay=H.wd, lr=H.lr, betas=(H.adam_beta1, H.adam_beta2))
    
    # Virtual dataset size for epoch simulation (default 100 like small dataset)
    virtual_dataset_size = getattr(H, 'virtual_dataset_size', 100)
    iters_per_epoch = max(1, virtual_dataset_size // H.n_batch)
    
    # Get num_iters
    num_iters = getattr(H, 'num_iters', None)
    if num_iters is None:
        num_iters = H.num_epochs * iters_per_epoch
    
    # Same scheduler as IMLE: warmup (per-iteration) then StepLR (per-epoch)
    # scheduler1: linear warmup over warmup_iters iterations
    # scheduler2: StepLR that decays every lr_decay_iters epochs
    scheduler1 = LambdaLR(optimizer, lr_lambda=linear_warmup(H.warmup_iters))
    scheduler2 = StepLR(optimizer, step_size=H.lr_decay_iters, gamma=H.lr_decay_rate)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[H.warmup_iters])
    
    logprint(f'IMLE-style scheduler (virtual epochs):')
    logprint(f'  - Virtual dataset size: {virtual_dataset_size}')
    logprint(f'  - Iterations per epoch: {iters_per_epoch} (= {virtual_dataset_size} / {H.n_batch})')
    logprint(f'  - Warmup: 0 to {H.warmup_iters} iters (linear warmup, stepped per iteration)')
    logprint(f'  - StepLR: decay by {H.lr_decay_rate} every {H.lr_decay_iters} epochs (stepped per epoch)')
    logprint(f'  - Total iterations: {num_iters} (~{num_iters // iters_per_epoch} epochs)')
    
    # Try to restore from checkpoint
    iterate = 0
    if H.restore_optimizer_path:
        optimizer.load_state_dict(
            torch.load(H.restore_optimizer_path, map_location='cuda', weights_only=False))
        logprint(f'Restored optimizer from {H.restore_optimizer_path}')
    if H.restore_scheduler_path:
        scheduler.load_state_dict(
            torch.load(H.restore_scheduler_path, map_location='cuda', weights_only=False))
        logprint(f'Restored scheduler from {H.restore_scheduler_path}')
    
    # Track restored best_fid for later use
    restored_best_fid = None
    
    if H.restore_log_path:
        # Simple restore for resample_every_batch (only needs step, no epoch)
        import json
        import re
        loaded = [json.loads(l) for l in open(H.restore_log_path)]
        iterate = max([z['step'] for z in loaded if 'type' in z and z['type'] == 'train_loss'])
        
        # Also restore best_fid from log messages
        for entry in loaded:
            if 'message' in entry and 'New best FID:' in entry['message']:
                # Parse "New best FID: 72.16 @ iteration 64000"
                match = re.search(r'New best FID: ([\d.]+)', entry['message'])
                if match:
                    restored_best_fid = float(match.group(1))
        
        logprint(f'Restored training state: starting at iteration {iterate}')
        if restored_best_fid is not None:
            logprint(f'Restored best FID: {restored_best_fid}')
    
    print(f"Starting iteration: {iterate}")
    print(f"Batch size: {H.n_batch}")
    print(f"Teacher resample steps: {getattr(H, 'teacher_resample_steps', 20)}")
    
    stats = []
    H.ema_rate = torch.as_tensor(H.ema_rate)
    
    # Create a simple loss calculator (like sampler.calc_loss) without full sampler
    from LPNet import LPNet
    class SimpleLossCalculator:
        def __init__(self, H):
            self.H = H
            self.lpips_net = LPNet(pnet_type=H.lpips_net, path=H.lpips_path).cuda()
            self.l2_loss = torch.nn.MSELoss(reduce=False).cuda()
        
        def calc_loss(self, inp, tar, use_mean=True, logging=False):
            """Same calc_loss as in sampler.py"""
            inp_feat, inp_shape = self.lpips_net(inp)
            tar_feat, _ = self.lpips_net(tar)
            
            if use_mean:       
                l2_loss = torch.mean(self.l2_loss(inp, tar), dim=[1, 2, 3])
                res = 0
            
                for i, g_feat in enumerate(inp_feat):
                    lpips_feature_loss = (g_feat - tar_feat[i]) ** 2
                    res += torch.sum(lpips_feature_loss, dim=1) / (inp_shape[i] ** 2)
                
                loss = self.H.lpips_coef * res.mean() + self.H.l2_coef * l2_loss.mean()
                if logging:
                    return loss, res.mean(), l2_loss.mean()
                else:
                    return loss
            else:
                res = 0
                for i, g_feat in enumerate(inp_feat):
                    res += torch.sum((g_feat - tar_feat[i]) ** 2, dim=1) / (inp_shape[i] ** 2)
                l2_loss = torch.mean(self.l2_loss(inp, tar), dim=[1, 2, 3])
                loss = self.H.lpips_coef * res + self.H.l2_coef * l2_loss
                if logging:
                    return loss, res.mean(), l2_loss
                else:
                    return loss
    
    loss_calculator = SimpleLossCalculator(H)
    
    # Training loop - no epochs, just iterations
    # num_iters already calculated above for scheduler
    print(f"Training for {num_iters} iterations...")
    
    latent_size = H.image_size // 8
    
    # Get res from decoder blocks
    from models import parse_layer_string
    blocks = parse_layer_string(H.dec_blocks)
    res = sorted(set([s[0] for s in blocks if s[0] <= H.max_hierarchy]))
    
    # Use restored best_fid if available, otherwise start fresh
    best_fid = restored_best_fid if restored_best_fid is not None else float('inf')
    
    for iteration in range(iterate, num_iters):
        # 1. Generate n_batch fresh noise (conditions)
        batch_noise = torch.randn(H.n_batch, 4, latent_size, latent_size, device='cuda')
        
        # 2. Teacher generates samples from noise
        with torch.no_grad():
            teacher_copy = copy.deepcopy(teacher_model)
            node = NeuralODE(teacher_copy, solver="euler", sensitivity="adjoint")
            
            traj = node.trajectory(
                batch_noise,
                t_span=torch.linspace(0, 1, getattr(H, 'teacher_resample_steps', 20), device='cuda'),
            )
            teacher_latents = traj[-1, :].view(batch_noise.shape)
            
            # Decode latents → images (VAE outputs in [-1, 1] range)
            teacher_images = teacher_vae.decode(
                teacher_latents / teacher_vae.config.scaling_factor
            ).sample  # [B, C, H, W] in [-1, 1]
            
            # Denormalize from [-1, 1] to [0, 1] and then to [0, 255]
            # (Exactly like generate_new_data_from_teacher in sampler.py)
            teacher_images = (teacher_images + 1) / 2
            teacher_images = torch.clamp(teacher_images, 0, 1) * 255
            
            # Convert to uint8 and permute to [B, H, W, C]
            teacher_images = teacher_images.byte().permute(0, 2, 3, 1).cpu()
        
        # 3. Prepare teacher images as targets for training
        #    Teacher outputs are uint8 in [0, 255]; convert to [-1, 1] to match model output range
        #    (Exactly like preprocess_fn: x / 127.5 - 1)
        targets = (teacher_images.float() / 127.5 - 1.0).cuda()  # [B, H, W, C] in [-1, 1]
        
        # 4. Student uses the SAME noise as input (unconditional, no conditions)
        # Treat noise as latents directly; model is unconditional (no condition_data)
        batch_latents = batch_noise  # shape: [B, 4, H//8, W//8]
        
        # No spatial noise (snoise) in this simple mode
        batch_snoise = None
        
        # 5. Training step - use the same calc_loss as regular IMLE training
        stat = training_step_imle(
            H,
            H.n_batch,
            targets,
            batch_latents,
            batch_snoise,
            imle,
            ema_imle,
            optimizer,
            loss_calculator.calc_loss,  # Use the same calc_loss function
            condition_data=None,
            batch_conditions=None,
            batch_condition_indices=None,
        )
        stats.append(stat)
        
        # Step scheduler like IMLE:
        # - During warmup: step every iteration
        # - After warmup: step every iters_per_epoch (once per virtual epoch)
        if iteration <= H.warmup_iters:
            scheduler.step()
        elif (iteration - H.warmup_iters) % iters_per_epoch == 0:
            # End of virtual epoch, step the StepLR scheduler
            scheduler.step()

        # Logging and saving
        if iteration % 500 == 0:
            with torch.no_grad():
                generate_visualization_grid(
                    batch_noise=batch_noise,
                    teacher_images=teacher_images,
                    teacher_model=teacher_model,
                    teacher_vae=teacher_vae,
                    imle=imle,
                    H=H,
                    latent_size=latent_size,
                    iteration=iteration,
                    experiment=experiment,
                    save_dir=H.save_dir,
                    rows=5,
                    cols=5
                )
        
        # Save latest model every 1000 iterations (overwrites previous)
        if iteration % 750 == 0 and iteration > 0:
            fp = os.path.join(H.save_dir, 'latest')
            logprint(f'Saving latest model @ iteration {iteration} to {fp}')
            save_model(fp, imle, ema_imle, optimizer, scheduler, H)
        
        # Log metrics
        if iteration % 100 == 0:
            loss_val = stat['loss'].item() if hasattr(stat['loss'], 'item') else stat['loss']
            logprint(model=H.desc, type='train_loss', step=iteration, loss=loss_val)
            if experiment is not None:
                experiment.log_metric('loss', loss_val, step=iteration)
        
        # FID calculation - use student model to generate images
        fid_freq_iters = getattr(H, 'fid_freq_iters', 1000)  # FID frequency in iterations
        if iteration > 0 and iteration % fid_freq_iters == 0:
            print("Calculating FID with student-generated images (unconditional IMLE-style)...")
            num_fid_samples = min(1000, H.n_batch * H.fid_factor)
            
            # Clear FID directory first
            delete_content_of_dir(f'{H.save_dir}/fid')
            
            # Use EMA model if available for better quality
            model_to_use = ema_imle if ema_imle is not None else imle
            model_to_use.eval()
            
            with torch.no_grad():
                sample_idx = 0
                for i in range(0, num_fid_samples, H.n_batch):
                    batch_size = min(H.n_batch, num_fid_samples - i)
                    if batch_size <= 0:
                        continue
                    
                    batch_fid_noise = torch.randn(batch_size, 4, latent_size, latent_size, device='cuda')

                    # Unconditional IMLE-style: noise is passed directly as latents, no snoise, no conditions
                    px_z = model_to_use(batch_fid_noise, None).permute(0, 2, 3, 1)  # [B, H, W, C]
                    xhat = ((px_z + 1.0) * 127.5).clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
                    
                    for j in range(batch_size):
                        imageio.imwrite(f'{H.save_dir}/fid/{sample_idx}.png', xhat[j])
                        sample_idx += 1
            
            model_to_use.train()

            try:
                cur_fid = fid.compute_fid(H.fid_real_dir, f'{H.save_dir}/fid/', verbose=False)
                print(f"FID: {cur_fid}")
                if experiment is not None:
                    experiment.log_metric('fid', cur_fid, step=iteration)
                
                # Save best FID model
                if cur_fid < best_fid:
                    best_fid = cur_fid
                    fp = os.path.join(H.save_dir, 'best_fid')
                    logprint(f'New best FID: {cur_fid:.2f} @ iteration {iteration}, saving to {fp}')
                    save_model(fp, imle, ema_imle, optimizer, scheduler, H)
                    if experiment is not None:
                        experiment.log_metric('best_fid', best_fid, step=iteration)
            except Exception as e:
                print(f"FID computation failed: {e}")
    
    print("Training complete!")


def main(H=None):
    H_cur, logprint = set_up_hyperparams()
    if not H:
        H = H_cur
    # In resample_every_batch mode, we don't need to load a dataset from disk.
    if getattr(H, "resample_every_batch", False):
        data_train = None
        data_valid_or_test = None
        preprocess_fn = None
        condition_data = None
        print("Resample-every-batch mode: skipping set_up_data() dataset loading")
    else:
        H, data_train, data_valid_or_test, preprocess_fn, condition_data = set_up_data(H)
    
    imle, ema_imle = load_imle(H, logprint)

    if H.use_comet and H.comet_api_key:
        if(H.comet_experiment_key):
            print("Resuming experiment")
            experiment = ExistingExperiment(
                api_key=H.comet_api_key,
                previous_experiment=H.comet_experiment_key
            )
            experiment.log_parameters(H)

        else:
            experiment = Experiment(
                api_key=H.comet_api_key,
                project_name="flow-model-imle",
                workspace="kianhk6",
            )
            experiment.set_name(H.comet_name)
            experiment.log_parameters(H)
    else:
        experiment = None


    os.makedirs(f'{H.save_dir}/fid', exist_ok=True)
    

    if H.mode == 'eval':
        
        os.makedirs(f'{H.save_dir}/eval', exist_ok=True)
        print(H)

        with torch.no_grad():
            # Generating
            sampler = Sampler(H, len(data_train), preprocess_fn)
            n_samp = H.n_batch
            temp_latent_rnds = torch.randn([n_samp, H.latent_dim], dtype=torch.float32).cuda()
            
            # Use EMA model if ema weights were loaded, otherwise use regular model
            model_to_use = ema_imle if H.restore_ema_path else imle
            
            for i in range(0, H.num_images_to_generate // n_samp):
                if (i % 10 == 0):
                    print(i * n_samp)
                temp_latent_rnds.normal_()
                tmp_snoise = [s[:n_samp].normal_() for s in sampler.snoise_tmp]
                
                # Save latents and snoise (with error handling for large files)
                try:
                    torch.save(temp_latent_rnds.cpu(), f'{H.save_dir}/eval/temp_latent_rnds_{i}.pt')
                except Exception as e:
                    print(f"Warning: Could not save latent_rnds_{i}: {e}")
                
                try:
                    # Save snoise as a list of CPU tensors to avoid serialization issues
                    torch.save([s.cpu() for s in tmp_snoise], f'{H.save_dir}/eval/tmp_snoise_{i}.pt')
                except Exception as e:
                    print(f"Warning: Could not save tmp_snoise_{i}: {e}")
                
                # For conditional models, generate random noise as conditions
                if condition_data is not None:
                    # Get a sample from condition_data to determine shape
                    if hasattr(condition_data, 'tensors'):
                        # TensorDataset - access the first tensor
                        sample_cond = condition_data.tensors[0][:n_samp]
                    else:
                        # Regular tensor
                        sample_cond = condition_data[:n_samp]
                    
                    # Use random noise as condition instead of actual condition_data
                    random_cond = torch.randn_like(sample_cond).cuda()
                    try:
                        torch.save(random_cond.cpu(), f'{H.save_dir}/eval/random_conditions_{i}.pt')
                    except Exception as e:
                        print(f"Warning: Could not save random_conditions_{i}: {e}")
                    samp = sampler.sample(temp_latent_rnds, model_to_use, tmp_snoise, condition_data=random_cond)
                else:
                    samp = sampler.sample(temp_latent_rnds, model_to_use, tmp_snoise)
                
                for j in range(n_samp):
                    imageio.imwrite(f'{H.save_dir}/eval/{i * n_samp + j}.png', samp[j])

    elif H.mode == 'eval_fid':
        subset_len = H.subset_len
        if subset_len == -1:
            subset_len = len(data_train)
        sampler = Sampler(H, len(data_train), preprocess_fn)
        generate_and_save(H, imle, sampler, 5000)
        real_dir = H.fid_real_dir if (hasattr(H, 'fid_real_dir') and H.fid_real_dir) else f'{H.data_root}/img'
        print(real_dir, f'{H.save_dir}/fid/')
        cur_fid = fid.compute_fid(real_dir, f'{H.save_dir}/fid/', verbose=False)
        print("FID: ", cur_fid)


    elif H.mode == 'reconstruct':

        subset_len = H.subset_len
        if subset_len == -1:
            subset_len = len(data_train)
        ind = 0
        for split_ind, split_x_tensor in enumerate(DataLoader(data_train, batch_size=H.subset_len, pin_memory=True)):
            if (ind == 14):
                break
            split_x = TensorDataset(split_x_tensor[0])
            ind += 1
            
        for param in imle.parameters():
            param.requires_grad = False
        viz_batch_original, _ = get_sample_for_visualization(split_x, preprocess_fn,
                                                                H.num_images_visualize, H.dataset)
        if os.path.isfile(str(H.restore_latent_path)):
            latents = torch.tensor(torch.load(H.restore_latent_path), requires_grad=True)
        else:
            latents = torch.randn([viz_batch_original.shape[0], H.latent_dim], requires_grad=True)
        sampler = Sampler(H, subset_len, preprocess_fn)
        reconstruct(H, sampler, imle, preprocess_fn, viz_batch_original, latents, 'reconstruct', logprint, training_step_imle)

    elif H.mode == 'backtrack':
        for param in imle.parameters():
            param.requires_grad = False
        for split_x in DataLoader(data_train, batch_size=H.subset_len):
            split_x = split_x[0]
            pass
        print(f'split shape is {split_x.shape}')
        sampler = Sampler(H, H.subset_len, preprocess_fn)
        backtrack(H, sampler, imle, preprocess_fn, split_x, logprint, training_step_imle)


    elif H.mode == 'train':
        print(H)
        # Check if resample_every_batch mode is enabled
        if hasattr(H, 'resample_every_batch') and H.resample_every_batch:
            train_loop_resample_every_batch(H, imle, ema_imle, logprint, experiment)
        else:
            train_loop_imle(H, data_train, data_valid_or_test, preprocess_fn, imle, ema_imle, logprint, experiment, condition_data)

    elif H.mode == 'ppl':
        sampler = Sampler(H, H.subset_len, preprocess_fn)
        calc_ppl(H, imle, sampler)

    elif H.mode == 'ppl_uniform':
        sampler = Sampler(H, H.subset_len, preprocess_fn)
        calc_ppl_uniform(H, imle, sampler)
    
    elif H.mode == 'interpolate':
        subset_len = H.subset_len
        if subset_len == -1:
            subset_len = len(data_train)
        with torch.no_grad():
            for split_x in DataLoader(data_train, batch_size=subset_len):
                split_x = split_x[0]
            viz_batch_original, _ = get_sample_for_visualization(split_x, preprocess_fn,
                                                                    H.num_images_visualize, H.dataset)
            sampler = Sampler(H, subset_len, preprocess_fn)
            for i in range(H.num_images_to_generate):
                random_interp(H, sampler, (0, 256, 256, 3), imle, f'{H.save_dir}/interp-{i}.png', logprint)

    elif H.mode == 'spatial_visual':
        with torch.no_grad():
            for split_x in DataLoader(data_train, batch_size=H.subset_len):
                split_x = split_x[0]
            viz_batch_original, _ = get_sample_for_visualization(split_x, preprocess_fn,
                                                                    H.num_images_visualize, H.dataset)
            sampler = Sampler(H, H.subset_len, preprocess_fn)
            for i in range(H.num_images_to_generate):
                print(H.num_images_to_generate, i)
                spatial_vissual(H, sampler, (0, 256, 256, 3), imle, f'{H.save_dir}/interp-{i}.png', logprint)

    elif H.mode == 'generate_rnd':
        with torch.no_grad():
            for split_x in DataLoader(data_train, batch_size=H.subset_len):
                split_x = split_x[0]
            viz_batch_original, _ = get_sample_for_visualization(split_x, preprocess_fn,
                                                                    H.num_images_visualize, H.dataset)
            sampler = Sampler(H, H.subset_len, preprocess_fn)
            generate_rnd(H, sampler, (0, 256, 256, 3), imle, f'{H.save_dir}/rnd.png', logprint)

    elif H.mode == 'generate_rnd_nn':
        with torch.no_grad():
            for split_x in DataLoader(data_train, batch_size=len(data_train)):
                split_x = split_x[0]
            viz_batch_original, _ = get_sample_for_visualization(split_x, preprocess_fn,
                                                                    H.num_images_visualize, H.dataset)
            sampler = Sampler(H, H.subset_len, preprocess_fn)
            generate_rnd_nn(H, split_x,  sampler, (0, 256, 256, 3), imle, f'{H.save_dir}', logprint, preprocess_fn)

    elif H.mode == 'nn_interp':
        with torch.no_grad():
            for split_x in DataLoader(data_train, batch_size=len(data_train)):
                split_x = split_x[0]
            viz_batch_original, _ = get_sample_for_visualization(split_x, preprocess_fn,
                                                                    H.num_images_visualize, H.dataset)
            sampler = Sampler(H, H.subset_len, preprocess_fn)
            nn_interp(H, split_x,  sampler, (0, 256, 256, 3), imle, f'{H.save_dir}', logprint, preprocess_fn)

    elif H.mode == 'generate_sample_nn':
        with torch.no_grad():
            for split_x in DataLoader(data_train, batch_size=len(data_train)):
                split_x = split_x[0]
            viz_batch_original, _ = get_sample_for_visualization(split_x, preprocess_fn,
                                                                    H.num_images_visualize, H.dataset)
            sampler = Sampler(H, H.subset_len, preprocess_fn)
            generate_sample_nn(H, split_x,  sampler, (0, 256, 256, 3), imle, f'{H.save_dir}/rnd2.png', logprint, preprocess_fn)

    elif H.mode == 'backtrack_interpolate':
        subset_len = H.subset_len
        if subset_len == -1:
            subset_len = len(data_train)
        with torch.no_grad():
            for split_x in DataLoader(data_train, batch_size=subset_len):
                split_x = split_x[0]
            viz_batch_original, _ = get_sample_for_visualization(split_x, preprocess_fn,
                                                                    H.num_images_visualize, H.dataset)
            sampler = Sampler(H, subset_len, preprocess_fn)
            latents = torch.tensor(torch.load(f'{H.restore_latent_path}'), requires_grad=True, dtype=torch.float32, device='cuda')
            for i in range(latents.shape[0] - 1):
                lat0 = latents[i:i+1]
                lat1 = latents[i+1:i+2]
                sn1 = None
                sn2 = None
                random_interp(H, sampler, (0, 256, 256, 3), imle, f'{H.save_dir}/back-interp-{i}.png', logprint, lat0, lat1, sn1, sn2)

    elif H.mode == 'prec_rec':
        
        os.makedirs(f'{H.save_dir}/prec_rec', exist_ok=True)

        subset_len = H.subset_len
        if subset_len == -1:
            subset_len = len(data_train)
        sampler = Sampler(H, len(data_train), preprocess_fn)
        # generate_and_save(H, imle, sampler, 5000)

        print("Generating images")
        generate_and_save(H, imle, sampler, 1000, subdir='prec_rec')
        real_dir_final = H.fid_real_dir if (hasattr(H, 'fid_real_dir') and H.fid_real_dir) else f'{H.data_root}/img'
        print(real_dir_final, f'{H.save_dir}/prec_rec/')
        precision, recall = compute_prec_recall(real_dir_final, f'{H.save_dir}/prec_rec/')
        print("Precision: ", precision)
        print("Recall: ", recall)


if __name__ == "__main__":
    main()
