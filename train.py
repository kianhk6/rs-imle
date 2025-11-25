import os
import time

from comet_ml import Experiment, ExistingExperiment
import imageio
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
                          get_sample_for_visualization)
from helpers.improved_precision_recall import compute_prec_recall
from diffusers.models import AutoencoderKL

# To Do: this one is for the selected latents 
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
    
    # Load teacher model and VAE if teacher resampling is enabled
    teacher_model = None
    teacher_vae = None
    
    if hasattr(H, 'use_teacher_resample') and H.use_teacher_resample:
        print("Loading teacher model for dynamic resampling...")
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
                teacher_seed = H.seed + last_teacher_epoch if hasattr(H, 'seed') else None
                print(f"RESTORING: Regenerating teacher dataset from last resample (Epoch {last_teacher_epoch})", flush=True)
                print(f"  Starting epoch: {starting_epoch}, Teacher resample interval: {teacher_force_resample}", flush=True)
                print(f"  Using teacher seed: {teacher_seed}", flush=True)
            else:
                # Fresh training: generate initial dataset at epoch -1 (before epoch 0)
                teacher_seed = H.seed + epoch if hasattr(H, 'seed') else None
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

            save_latents_latest(H, split_ind, sampler.selected_latents)
            save_latents_latest(H, split_ind, change_thresholds, name='threshold_latest')

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
            # # Compute metrics with the generator in eval mode to avoid dropout noise
            # was_training = imle.training
            # imle.eval()
            # try:
            #     cur_dists[:], cur_dists_lpips[:], cur_dists_l2[:] = sampler.calc_dists_existing(
            #         split_x_tensor,
            #         imle,
            #         dists=cur_dists,
            #         dists_lpips=cur_dists_lpips,
            #         dists_l2=cur_dists_l2,
            #         logging=True,
            #         conditions=split_condition_data,
            #         expect_condition_indices=False,
            #     )
            # finally:
            #     if was_training:
            #         imle.train()
            # torch.save(cur_dists, f'{H.save_dir}/latent/dists-{epoch}.npy')
                    
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
                            condition_data=cond_vis2,
                        )

                    else:
                        generate_images_initial(H, sampler, viz_batch_original,
                        sampler.selected_latents[0: H.num_images_visualize],
                        [s[0: H.num_images_visualize] for s in sampler.selected_snoise],
                        viz_batch_original.shape, imle, ema_imle,
                        f'{H.save_dir}/latest.png', logprint, experiment)                    


            
            if experiment is not None:
                experiment.log_metrics(metrics, epoch=epoch, step=iterate)

def main(H=None):
    H_cur, logprint = set_up_hyperparams()
    if not H:
        H = H_cur
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
