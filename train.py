import os
import time

from comet_ml import Experiment, ExistingExperiment
import imageio
import torch
import wandb
import torch.nn as nn
from cleanfid import fid
from torch.utils.data import DataLoader, TensorDataset

from data import set_up_data
from helpers.imle_helpers import backtrack, reconstruct
from helpers.train_helpers import (load_imle, load_opt, save_latents,
                                   save_latents_latest, save_model,
                                   save_snoise, set_up_hyperparams, update_ema)
from helpers.utils import ZippedDataset, get_cpu_stats_over_ranks
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
    
    if condition_data is not None:
        sampler = Sampler(
            H,
            subset_len,
            preprocess_fn,
            condition_config={"condition_tensor": condition_data},
        )
    else:
        sampler = Sampler(H, subset_len, preprocess_fn)

    last_updated = torch.zeros(subset_len, dtype=torch.int16).cuda()
    times_updated = torch.zeros(subset_len, dtype=torch.int8).cuda()
    change_thresholds = torch.empty(subset_len).cuda()
    change_thresholds[:] = H.change_threshold
    best_fid = 100000
    epoch = starting_epoch - 1

    for split_ind, split_x_tensor in enumerate(DataLoader(data_train, batch_size=subset_len, pin_memory=True)):
        split_x_tensor = split_x_tensor[0].contiguous()
        split_x = TensorDataset(split_x_tensor)
        sampler.init_projection(split_x_tensor)
        viz_batch_original, _ = get_sample_for_visualization(split_x, preprocess_fn, H.num_images_visualize, H.dataset)

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

        while (epoch < H.num_epochs):
            
            epoch += 1
            last_updated[:] = last_updated + 1

            if split_condition_data is not None:
                sampler.selected_dists[:] = sampler.calc_dists_existing(
                    split_x_tensor,
                    imle,
                    dists=sampler.selected_dists,
                    conditions=(split_condition_data, split_indices.cuda())
                )
            else:
                sampler.selected_dists[:] = sampler.calc_dists_existing(split_x_tensor, imle, dists=sampler.selected_dists)
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
                            f'{H.save_dir}/NN-samples_{epoch}-{split_ind}-imle.png', logprint)
                    print('loaded latest latents')

                if os.path.isfile(str(H.restore_latent_path)):
                    threshold = torch.load(H.restore_threshold_path)
                    change_thresholds[:] = threshold[:]
                    print('loaded thresholds', torch.mean(change_thresholds))
                else:
                    to_update = sampler.entire_ds


            change_thresholds[to_update] = sampler.selected_dists[to_update].clone() * (1 - H.change_coef)
            
            # Pass the corresponding condition data split to imle_sample_force
            if split_condition_data is not None:
                sampler.imle_sample_force(split_x_tensor, imle, to_update, condition_data=split_condition_data)
            else:
                sampler.imle_sample_force(split_x_tensor, imle, to_update)
            

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
                    )

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

                if(iterate <= H.warmup_iters):
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
                            f'{H.save_dir}/samples-{iterate}.png', logprint, experiment,
                            condition_data=cond_vis2,
                        )

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

            if(iterate > H.warmup_iters):
                scheduler.step()

            
            cur_dists = torch.empty([subset_len], dtype=torch.float32).cuda()
            cur_dists_lpips = torch.empty([subset_len], dtype=torch.float32).cuda()
            cur_dists_l2 = torch.empty([subset_len], dtype=torch.float32).cuda()


            cur_dists[:], cur_dists_lpips[:], cur_dists_l2[:] = sampler.calc_dists_existing(split_x_tensor, imle, 
                                                                                            dists=cur_dists,  
                                                                                            dists_lpips=cur_dists_lpips,
                                                                                            dists_l2=cur_dists_l2, 
                                                                                            logging=True,
                                                                                            conditions=split_condition_data,
                                                                                            expect_condition_indices=False)

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
                generate_and_save(H, imle, sampler, min(5000,subset_len * H.fid_factor), condition_data=split_condition_data)
                print(f'{H.data_root}/img', f'{H.save_dir}/fid/')
                cur_fid = fid.compute_fid(f'{H.data_root}/img', f'{H.save_dir}/fid/', verbose=False)
                if cur_fid < best_fid:
                    best_fid = cur_fid
                    # save models
                    fp = os.path.join(H.save_dir, 'best_fid')
                    logprint(f'Saving model best fid {best_fid} @ {iterate} to {fp}')
                    save_model(fp, imle, ema_imle, optimizer, scheduler, H)
                
                precision, recall = compute_prec_recall(f'{H.data_root}/img', f'{H.save_dir}/fid/')

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

            if H.use_wandb:
                wandb.log(metrics, step=iterate)
            
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

    if H.use_wandb:
        wandb.init(
            name=H.wandb_name,
            project=H.wandb_project,
            config=H,
            mode=H.wandb_mode,
        )

    os.makedirs(f'{H.save_dir}/fid', exist_ok=True)
    

    if H.mode == 'eval':
        
        os.makedirs(f'{H.save_dir}/eval', exist_ok=True)
        print(H)

        with torch.no_grad():
            # Generating
            sampler = Sampler(H, len(data_train), preprocess_fn)
            n_samp = H.n_batch
            temp_latent_rnds = torch.randn([n_samp, H.latent_dim], dtype=torch.float32).cuda()
            for i in range(0, H.num_images_to_generate // n_samp):
                if (i % 10 == 0):
                    print(i * n_samp)
                temp_latent_rnds.normal_()
                tmp_snoise = [s[:n_samp].normal_() for s in sampler.snoise_tmp]
                torch.save(temp_latent_rnds, f'{H.save_dir}/eval/temp_latent_rnds_{i}.pt')
                torch.save(tmp_snoise, f'{H.save_dir}/eval/tmp_snoise_{i}.pt')
                samp = sampler.sample(temp_latent_rnds, imle, tmp_snoise)
                for j in range(n_samp):
                    imageio.imwrite(f'{H.save_dir}/eval/{i * n_samp + j}.png', samp[j])

    elif H.mode == 'eval_fid':
        subset_len = H.subset_len
        if subset_len == -1:
            subset_len = len(data_train)
        sampler = Sampler(H, len(data_train), preprocess_fn)
        # generate_and_save(H, imle, sampler, 5000)

        generate_and_save(H, imle, sampler, 5000, condition_data=condition_data)
        print(f'{H.data_root}/img', f'{H.save_dir}/fid/')
        cur_fid = fid.compute_fid(f'{H.data_root}/img', f'{H.save_dir}/fid/', verbose=False)
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
        generate_and_save(H, imle, sampler, 1000, subdir='prec_rec', condition_data=condition_data)
        print(f'{H.data_root}/img', f'{H.save_dir}/prec_rec/')
        precision, recall = compute_prec_recall(f'{H.data_root}/img', f'{H.save_dir}/prec_rec/')
        print("Precision: ", precision)
        print("Recall: ", recall)


if __name__ == "__main__":
    main()
