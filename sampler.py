from curses import update_lines_cols
from math import comb, ceil
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from LPNet import LPNet
# from dciknn_cuda import DCI, MDCI
from torch.optim import AdamW
from helpers.utils import ZippedDataset
from models import parse_layer_string



class Sampler:
    def __init__(self, H, sz, preprocess_fn, condition_config=None, teacher_model=None, teacher_vae=None):
        # In regression mode (use_teacher_noise_as_input), we don't need IMLE sampling pool
        # so pool_size = sz directly. Otherwise, round up to imle_db_size for batching.
        self.use_teacher_noise_as_input = getattr(H, 'use_teacher_noise_as_input', False)
        if self.use_teacher_noise_as_input:
            self.pool_size = sz  # Regression mode: 1:1 mapping, no pool needed
            print(f"✓ Regression mode: pool_size set to {sz} (no IMLE sampling pool)")
        else:
            self.pool_size = ceil(int(H.force_factor * sz) / H.imle_db_size) * H.imle_db_size
        
        self.preprocess_fn = preprocess_fn
        self.l2_loss = torch.nn.MSELoss(reduce=False).cuda()
        self.H = H
        self.latent_lr = H.latent_lr
        self.entire_ds = torch.arange(sz)
        self.selected_latents = torch.empty([sz, H.latent_dim], dtype=torch.float32)
        self.selected_latents_tmp = torch.empty([sz, H.latent_dim], dtype=torch.float32)
        
        # Teacher model for dynamic data generation during resampling
        self.teacher_model = teacher_model
        self.teacher_vae = teacher_vae
        self.use_teacher_resample = (teacher_model is not None) and hasattr(H, 'use_teacher_resample') and H.use_teacher_resample
        
        # Initialize tracking for scheduled resampling
        self.last_teacher_resample_epoch = 0
        
        # Parse list-based scheduling if provided
        self.schedule_intervals = None
        self.schedule_transitions = None
        if hasattr(H, 'every_n_epochs_resample_data') and H.every_n_epochs_resample_data:
            self.schedule_intervals = [int(x.strip()) for x in H.every_n_epochs_resample_data.split(',')]
        if hasattr(H, 'change_schedule_of_data_resampling_every_n_epoch') and H.change_schedule_of_data_resampling_every_n_epoch:
            self.schedule_transitions = [int(x.strip()) for x in H.change_schedule_of_data_resampling_every_n_epoch.split(',')]
        
        # Validate schedule configuration
        if self.schedule_intervals is not None and self.schedule_transitions is not None:
            # Check: intervals must be N+1, transitions must be N
            if len(self.schedule_intervals) != len(self.schedule_transitions) + 1:
                raise ValueError(
                    f"every_n_epochs_resample_data must have exactly ONE MORE element than change_schedule_of_data_resampling_every_n_epoch.\n"
                    f"Got {len(self.schedule_intervals)} intervals and {len(self.schedule_transitions)} transitions.\n"
                    f"Expected: {len(self.schedule_transitions) + 1} intervals for {len(self.schedule_transitions)} transitions."
                )
            
            # Check: each interval must be <= the range it covers
            print(f"\n{'='*70}")
            print(f"TEACHER RESAMPLE SCHEDULE VALIDATION")
            print(f"{'='*70}")
            print(f"every_n_epochs_resample_data: {self.schedule_intervals}")
            print(f"change_schedule_of_data_resampling_every_n_epoch: {self.schedule_transitions}")
            print(f"\nSchedule breakdown:")
            
            # Phase 0: from 0 to first transition
            range_length = self.schedule_transitions[0]
            interval = self.schedule_intervals[0]
            print(f"  Phase 1: Epoch 0 → {self.schedule_transitions[0]}")
            print(f"    Resample every {interval} epochs")
            if interval > range_length:
                print(f"    ⚠️  WARNING: Interval ({interval}) > Range ({range_length})")
                print(f"    This means only 1 resample will occur in this phase!")
            else:
                num_resamples = range_length // interval
                print(f"    ✓ Expected resamples: ~{num_resamples}")
            
            # Middle phases
            for i in range(len(self.schedule_transitions) - 1):
                start = self.schedule_transitions[i]
                end = self.schedule_transitions[i + 1]
                range_length = end - start
                interval = self.schedule_intervals[i + 1]
                print(f"\n  Phase {i+2}: Epoch {start} → {end}")
                print(f"    Resample every {interval} epochs")
                if interval > range_length:
                    print(f"    ⚠️  WARNING: Interval ({interval}) > Range ({range_length})")
                    print(f"    This phase may have very few resamples!")
                else:
                    num_resamples = range_length // interval
                    print(f"    ✓ Expected resamples: ~{num_resamples}")
            
            # Final phase
            last_transition = self.schedule_transitions[-1]
            last_interval = self.schedule_intervals[-1]
            num_epochs = getattr(H, 'num_epochs', 15000)
            range_length = num_epochs - last_transition
            print(f"\n  Phase {len(self.schedule_transitions)+1}: Epoch {last_transition} → {num_epochs} (end)")
            print(f"    Resample every {last_interval} epochs")
            if last_interval > range_length:
                print(f"    ⚠️  WARNING: Interval ({last_interval}) > Range ({range_length})")
            else:
                num_resamples = range_length // last_interval
                print(f"    ✓ Expected resamples: ~{num_resamples}")
            
            print(f"{'='*70}\n")

        blocks = parse_layer_string(H.dec_blocks)
        self.block_res = [s[0] for s in blocks]
        self.res = sorted(set([s[0] for s in blocks if s[0] <= H.max_hierarchy]))
        self.neutral_snoise = [torch.zeros([self.H.imle_db_size, 1, s, s], dtype=torch.float32) for s in self.res]

        if(H.use_snoise == True):
            self.snoise_tmp = [torch.randn([self.H.imle_db_size, 1, s, s], dtype=torch.float32) for s in self.res]
            self.selected_snoise = [torch.randn([sz, 1, s, s,], dtype=torch.float32) for s in self.res]
            self.snoise_pool = [torch.randn([self.pool_size, 1, s, s], dtype=torch.float32) for s in self.res]
        else:
            self.snoise_tmp = [torch.zeros([self.H.imle_db_size, 1, s, s], dtype=torch.float32) for s in self.res]
            self.selected_snoise = [torch.zeros([sz, 1, s, s,], dtype=torch.float32) for s in self.res]
            self.snoise_pool = [torch.zeros([self.pool_size, 1, s, s], dtype=torch.float32) for s in self.res]

        self.selected_dists = torch.empty([sz], dtype=torch.float32).cuda()
        self.selected_dists[:] = np.inf
        self.selected_dists_tmp = torch.empty([sz], dtype=torch.float32).cuda()

        self.selected_dists_lpips = torch.empty([sz], dtype=torch.float32).cuda()
        self.selected_dists_lpips[:] = np.inf

        self.selected_dists_l2 = torch.empty([sz], dtype=torch.float32).cuda()
        self.selected_dists_l2[:] = np.inf 


        self.temp_latent_rnds = torch.empty([self.H.imle_db_size, self.H.latent_dim], dtype=torch.float32)
        self.temp_samples = torch.empty([self.H.imle_db_size, H.image_channels, self.H.image_size, self.H.image_size],
                                        dtype=torch.float32)

        self.pool_latents = torch.randn([self.pool_size, H.latent_dim], dtype=torch.float32)
        self.sample_pool_usage = torch.ones([sz], dtype=torch.bool)

        self.projections = []
        self.lpips_net = LPNet(pnet_type=H.lpips_net, path=H.lpips_path).cuda()
        self.l2_projection = None
        self.knn_ignore = H.knn_ignore
        self.ignore_radius = H.ignore_radius
        fake = torch.zeros(1, 3, H.image_size, H.image_size).cuda()
        out, shapes = self.lpips_net(fake)
        sum_dims = 0

        if(H.search_type == 'lpips'):
            dims = [int(H.proj_dim * 1. / len(out)) for _ in range(len(out))]
            if H.proj_proportion:
                sm = sum([dim.shape[1] for dim in out])
                dims = [int(out[feat_ind].shape[1] * (H.proj_dim / sm)) for feat_ind in range(len(out) - 1)]
                dims.append(H.proj_dim - sum(dims))
            for ind, feat in enumerate(out):
                self.projections.append(F.normalize(torch.randn(feat.shape[1], dims[ind]), p=2, dim=1).cuda())
            sum_dims = sum(dims)

        elif(H.search_type == 'l2'):
            interpolated = F.interpolate(fake,scale_factor = H.l2_search_downsample)
            interpolated = interpolated.reshape(interpolated.shape[0],-1)
            self.l2_projection = F.normalize(torch.randn(interpolated.shape[1], H.proj_dim), p=2, dim=1).cuda()
            sum_dims = H.proj_dim

        else:

            projection_dim = H.proj_dim // 2
            dims = [int(projection_dim * 1. / len(out)) for _ in range(len(out))]
            if H.proj_proportion:
                sm = sum([dim.shape[1] for dim in out])
                dims = [int(out[feat_ind].shape[1] * (projection_dim / sm)) for feat_ind in range(len(out) - 1)]
                dims.append(projection_dim - sum(dims))
            for ind, feat in enumerate(out):
                self.projections.append(F.normalize(torch.randn(feat.shape[1], dims[ind]), p=2, dim=1).cuda())

            interpolated = F.interpolate(fake,scale_factor = H.l2_search_downsample)
            interpolated = interpolated.reshape(interpolated.shape[0],-1)
            self.l2_projection = F.normalize(torch.randn(interpolated.shape[1], H.proj_dim // 2), p=2, dim=1).cuda()
            sum_dims = H.proj_dim

        # self.dci_dim = sum_dims
        # print('dci_dim', self.dci_dim)

        self.temp_samples_proj = torch.empty([self.H.imle_db_size, sum_dims], dtype=torch.float32).cuda()
        self.dataset_proj = torch.empty([sz, sum_dims], dtype=torch.float32)
        self.pool_samples_proj = torch.empty([self.pool_size, sum_dims], dtype=torch.float32)
        self.pool_condition_data = None
        self.pool_condition_indices = None

        self.condition_config = None

        if condition_config is not None:
            self.configure_conditions(sz=sz, **condition_config, use_teacher_noise_as_input=self.use_teacher_noise_as_input)

    def configure_conditions(
        self,
        condition_tensor=None,
        force_factor=None,
        base_indices=None,
        device=None,
        sz=None,
        use_teacher_noise_as_input = False,
    ):
        if condition_tensor is None:
            self.pool_condition_data = None
            self.pool_condition_indices = None
            self.condition_config = None
            return

        if isinstance(condition_tensor, list):
            condition_tensor = torch.stack(condition_tensor)

        if force_factor is None:
            force_factor = self.H.force_factor

        sample_factor = int(force_factor)
        if sample_factor <= 0:
            raise ValueError("force_factor must be a positive integer when configuring conditions")

        if isinstance(condition_tensor, TensorDataset):
            if len(condition_tensor.tensors) == 0:
                raise ValueError("TensorDataset provided for conditions contains no tensors")
            condition_tensor = condition_tensor.tensors[0]

        if hasattr(condition_tensor, "detach"):
            condition_tensor = condition_tensor.detach()

        if device is not None:
            device = torch.device(device)
            condition_tensor = condition_tensor.to(device=device)
        else:
            device = condition_tensor.device if hasattr(condition_tensor, "device") else torch.device("cpu")

        data_size = condition_tensor.shape[0]
        expected_pool = sample_factor * data_size
        
        # In regression mode (use_teacher_noise_as_input), skip pool size validation
        # since we don't use IMLE sampling and force_factor=1
        if not use_teacher_noise_as_input:
            if expected_pool != self.pool_size:
                raise ValueError(
                    f"Provided condition tensor of length {data_size} with force_factor {sample_factor} "
                    f"does not match pool_size {self.pool_size}"
                )
        else:
            # Regression mode: verify force_factor=1
            if sample_factor != 1:
                print(f"WARNING: use_teacher_noise_as_input=True but force_factor={sample_factor}")
                print(f"         Regression mode should use force_factor=1 for direct mapping")

        # For regression mode with force_factor=1, this just keeps the tensor as-is
        # For IMLE mode with force_factor>1, this creates multiple candidates per sample
        interleaved_conditions = torch.repeat_interleave(condition_tensor, sample_factor, dim=0)
        if not interleaved_conditions.is_contiguous():
            interleaved_conditions = interleaved_conditions.contiguous()

        self.pool_condition_data = interleaved_conditions

        if base_indices is None:
            base_indices = torch.arange(data_size, device=device, dtype=torch.long)
        else:
            base_indices = base_indices.to(device=device, dtype=torch.long)

        self.pool_condition_indices = torch.repeat_interleave(base_indices, sample_factor)

        self.condition_config = {
            "base_shape": tuple(condition_tensor.shape),
            "force_factor": sample_factor,
            "device": str(device),
            "dtype": str(condition_tensor.dtype),
        }

        self.total_excluded = 0
        self.total_excluded_percentage = 0
        self.dataset_size = sz
        self.db_iter = 0

        # Condition debug tracking (stores condition tensors and indices per pool sample)
        self.condition_debug = True


    def get_projected(self, inp, permute=True):
        if permute:
            out, _ = self.lpips_net(inp.permute(0, 3, 1, 2).cuda())
        else:
            out, _ = self.lpips_net(inp.cuda())
        gen_feat = []
        for i in range(len(out)):
            gen_feat.append(torch.mm(out[i], self.projections[i]))
            # TODO divide?
        lpips_feat = torch.cat(gen_feat, dim=1)
        lpips_feat = F.normalize(lpips_feat, p=2, dim=1)
        return lpips_feat.cuda()
    
    def get_l2_feature(self, inp, permute=True):
        if(permute):
            inp = inp.permute(0, 3, 1, 2)
        interpolated = F.interpolate(inp,scale_factor = self.H.l2_search_downsample)
        interpolated = interpolated.reshape(interpolated.shape[0],-1)
        interpolated = torch.mm(interpolated, self.l2_projection)
        interpolated = F.normalize(interpolated, p=2, dim=1)
        return interpolated.cuda()
    
    def get_combined_feature(self, inp, permute=True):
        lpips_feat = self.get_projected(inp, permute)
        l2_feat = self.get_l2_feature(inp, permute)
        return torch.cat([lpips_feat, l2_feat], dim=1)
        # return torch.cat([lpips_feat, l2_feat], dim=1)
        # if(permute):
        #     inp = inp.permute(0, 3, 1, 2)

        # out, _ = self.lpips_net(inp.cuda())
        # gen_feat = []
        # for i in range(len(out)):
        #     gen_feat.append(torch.mm(out[i], self.projections[i]))
        #     # TODO divide?
        # gen_feat = torch.cat(gen_feat, dim=1)
        # interpolated = F.interpolate(inp,scale_factor = self.H.l2_search_downsample)
        # interpolated = interpolated.reshape(interpolated.shape[0],-1)
        # interpolated = torch.mm(interpolated, self.l2_projection)
        # return gen_feat + interpolated.cuda()

    def init_projection(self, dataset):
        for proj_mat in self.projections:
            proj_mat[:] = F.normalize(torch.randn(proj_mat.shape), p=2, dim=1)

        for ind, x in enumerate(DataLoader(TensorDataset(dataset), batch_size=self.H.n_batch)):
            batch_slice = slice(ind * self.H.n_batch, ind * self.H.n_batch + x[0].shape[0])
            if(self.H.search_type == 'lpips'):
                self.dataset_proj[batch_slice] = self.get_projected(self.preprocess_fn(x)[1])
            elif(self.H.search_type == 'l2'):
                self.dataset_proj[batch_slice] = self.get_l2_feature(self.preprocess_fn(x)[1])
            else:
                self.dataset_proj[batch_slice] = self.get_combined_feature(self.preprocess_fn(x)[1])

    def sample(self, latents, gen, snoise=None, condition_data=None):
        with torch.no_grad():
            nm = latents.shape[0]
            if snoise is None:
                for i in range(len(self.res)):
                    if(self.H.use_snoise == True):
                        self.snoise_tmp[i].normal_()
                snoise = [s[:nm] for s in self.snoise_tmp]
            if condition_data is not None:
                px_z = gen(latents, snoise, condition_data=condition_data).permute(0, 2, 3, 1)
            else:
                px_z = gen(latents, snoise).permute(0, 2, 3, 1)
            xhat = (px_z + 1.0) * 127.5
            xhat = xhat.detach().cpu().numpy()
            xhat = np.minimum(np.maximum(0.0, xhat), 255.0).astype(np.uint8)
            return xhat

    def sample_from_out(self, px_z):
        with torch.no_grad():
            px_z = px_z.permute(0, 2, 3, 1)
            xhat = (px_z + 1.0) * 127.5
            xhat = xhat.detach().cpu().numpy()
            xhat = np.minimum(np.maximum(0.0, xhat), 255.0).astype(np.uint8)
            return xhat
    
    def calc_loss_projected(self, inp, tar):
        inp_feat = self.get_projected(inp,False)
        tar_feat = self.get_projected(tar,False)
        res = torch.linalg.norm(inp_feat - tar_feat, dim=1)
        return res
    
    def calc_loss_l2(self, inp, tar):
        inp_feat = self.get_l2_feature(inp,False)
        tar_feat = self.get_l2_feature(tar,False)
        res = torch.linalg.norm(inp_feat - tar_feat, dim=1)
        return res

    def calc_loss(self, inp, tar, use_mean=True, logging=False):
        # inp_feat, inp_shape = self.lpips_net(inp)
        # tar_feat, _ = self.lpips_net(tar)
        # res = 0
        # for i, g_feat in enumerate(inp_feat):
        #     res += torch.sum((g_feat - tar_feat[i]) ** 2, dim=1) / (inp_shape[i] ** 2)
        # if use_mean:
        #     l2_loss = self.l2_loss(inp, tar)
        #     loss = self.H.lpips_coef * res.mean() + self.H.l2_coef * l2_loss.mean()
        #     if logging:
        #         return loss, res.mean(), l2_loss.mean()
        #     else:
        #         return loss

        # else:
        #     l2_loss = torch.mean(self.l2_loss(inp, tar), dim=[1, 2, 3])
        #     loss = self.H.lpips_coef * res + self.H.l2_coef * l2_loss
        #     if logging:
        #         return loss, res.mean(), l2_loss
        #     else:
        #         return loss

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


    def calc_dists_existing(self, dataset_tensor, gen, dists=None, dists_lpips = None, dists_l2 = None, latents=None, to_update=None, snoise=None, logging=False, conditions=None, expect_condition_indices: bool = True):
        if dists is None:
            dists = self.selected_dists
        if dists_lpips is None:
            dists_lpips = self.selected_dists_lpips
        if dists_l2 is None:
            dists_l2 = self.selected_dists_l2
        if latents is None:
            latents = self.selected_latents
        if snoise is None:
            snoise = self.selected_snoise

        # Normalize conditions input
        conditions_tensor = None
        condition_indices = None
        if conditions is not None:
            # If debugging with indices, expect a (tensor, indices) tuple; otherwise accept a tensor directly
            if expect_condition_indices and isinstance(conditions, (tuple, list)) and len(conditions) == 2:
                conditions_tensor, condition_indices = conditions
            elif isinstance(conditions, (tuple, list)) and len(conditions) >= 1:
                conditions_tensor = conditions[0]
                condition_indices = None
            else:
                conditions_tensor = conditions
                condition_indices = None

        if to_update is not None:
            latents = latents[to_update]
            dists = dists[to_update]
            dataset_tensor = dataset_tensor[to_update]
            snoise = [s[to_update] for s in snoise]
            if conditions_tensor is not None:
                conditions_tensor = conditions_tensor[to_update]
                if expect_condition_indices and condition_indices is not None:
                    condition_indices = condition_indices[to_update]
                else:
                    condition_indices = None
        
        for ind, x in enumerate(DataLoader(TensorDataset(dataset_tensor), batch_size=self.H.n_batch)):
            _, target = self.preprocess_fn(x)
            batch_slice = slice(ind * self.H.n_batch, ind * self.H.n_batch + target.shape[0])
            cur_latents = latents[batch_slice]
            cur_snoise = [s[batch_slice] for s in snoise]
            with torch.no_grad():
                if conditions_tensor is not None:
                    batch_conditions = conditions_tensor[batch_slice]
                    if expect_condition_indices and condition_indices is not None:
                        batch_condition_indices = condition_indices[batch_slice]
                        out, _ = gen(
                            cur_latents,
                            cur_snoise,
                            condition_data=batch_conditions,
                            condition_indices=batch_condition_indices,
                            return_condition=True,
                        )
                    else:
                        out = gen(cur_latents, cur_snoise, condition_data=batch_conditions)
                else:
                    out = gen(cur_latents, cur_snoise)
                
                # Calculate mean and variance of the output
                out_flat = out.view(out.shape[0], -1)
                out_mean_per_dim = out_flat.mean(dim=0)
                out_mean = out_mean_per_dim.mean().item()
                out_variance_per_dim = out_flat.var(dim=0)
                out_variance = out_variance_per_dim.mean().item()
                if(logging):
                    dist, dist_lpips, dist_l2 = self.calc_loss(target.permute(0, 3, 1, 2), out, use_mean=False, logging=True)
                    dists[batch_slice] = torch.squeeze(dist)
                    dists_lpips[batch_slice] = torch.squeeze(dist_lpips)
                    dists_l2[batch_slice] = torch.squeeze(dist_l2)
                else:
                    dist = self.calc_loss(target.permute(0, 3, 1, 2), out, use_mean=False)
                    dists[batch_slice] = torch.squeeze(dist)
        
        if(logging):
            return dists, dists_lpips, dists_l2
        else:
            return dists
    
    def calc_dists_existing_nn(self, dataset_tensor, gen, dists=None, latents=None, to_update=None, snoise=None):
        if dists is None:
            dists = self.selected_dists
        if latents is None:
            latents = self.selected_latents
        if snoise is None:
            snoise = self.selected_snoise

        if to_update is not None:
            latents = latents[to_update]
            dists = dists[to_update]
            dataset_tensor = dataset_tensor[to_update]
            snoise = [s[to_update] for s in snoise]

        for ind, x in enumerate(DataLoader(TensorDataset(dataset_tensor), batch_size=self.H.n_batch)):
            _, target = self.preprocess_fn(x)
            batch_slice = slice(ind * self.H.n_batch, ind * self.H.n_batch + target.shape[0])
            cur_latents = latents[batch_slice]
            cur_snoise = [s[batch_slice] for s in snoise]
            with torch.no_grad():
                out = gen(cur_latents, cur_snoise)
                if(self.H.search_type == 'lpips'):
                    dist = self.calc_loss_projected(target.permute(0, 3, 1, 2), out)
                else:
                    dist = self.calc_loss_l2(target.permute(0, 3, 1, 2), out)
                dists[batch_slice] = torch.squeeze(dist)
        return dists


    def resample_pool(self, gen, ds, conditions=True):
        # self.init_projection(ds)
        self.pool_latents.normal_()
        for i in range(len(self.res)):
            if(self.H.use_snoise == True):
                self.snoise_pool[i].normal_()
        
        if self.condition_config:
            interleaved_conditions = self.pool_condition_data
            interleaved_cond_idx = self.pool_condition_indices
            if interleaved_conditions.shape[0] != self.pool_size:
                raise ValueError(
                    f"Expected interleaved conditions with first dimension {self.pool_size}, "
                    f"got {interleaved_conditions.shape[0]}"
                )

            if interleaved_cond_idx.shape[0] != self.pool_size:
                raise ValueError(
                    f"Expected interleaved condition indices with length {self.pool_size}, "
                    f"got {interleaved_cond_idx.shape[0]}"
                )

        num_batches = self.pool_size // self.H.imle_batch
        
        # Accumulators for overall statistics across all batches
        all_variances = []
        all_means = []
        
        for j in range(self.pool_size // self.H.imle_batch):
            batch_slice = slice(j * self.H.imle_batch, (j + 1) * self.H.imle_batch)
            cur_latents = self.pool_latents[batch_slice]
            
            cur_snosie = [s[batch_slice] for s in self.snoise_pool]

            with torch.no_grad():
                # Generate samples with or without condition data
                if self.condition_config:
                    # Take the contiguous slice from the prebuilt interleaved list
                    batch_conditions = interleaved_conditions[batch_slice]
                    if self.pool_condition_indices is not None:
                        batch_cond_idx = self.pool_condition_indices[batch_slice]
                        generated_samples = gen(
                            cur_latents,
                            condition_data=batch_conditions,
                            condition_indices=batch_cond_idx,
                        )
                    else:
                        generated_samples = gen(cur_latents, condition_data=batch_conditions)
                else:
                    generated_samples = gen(cur_latents, cur_snosie)

                if(self.H.search_type == 'lpips'):
                    self.pool_samples_proj[batch_slice] = self.get_projected(generated_samples, False)
                elif(self.H.search_type == 'l2'):
                    self.pool_samples_proj[batch_slice] = self.get_l2_feature(generated_samples, False)
                else:
                    self.pool_samples_proj[batch_slice] = self.get_combined_feature(generated_samples, False)
                
                # # Compute variance statistics for generated_samples
                # # Shape: [Batch, 3, 256, 256] -> compute variance ACROSS different images (batch dimension)
                # # Variance across batch at each pixel/channel location
                # variance_across_images = torch.var(generated_samples, dim=0)  # Shape: [3, 256, 256]
                # # Mean of the variance (average variance across all pixel locations)
                # mean_variance = torch.mean(variance_across_images).item()
                # # Overall mean across all dimensions
                # overall_mean = torch.mean(generated_samples).item()
                
                # # Accumulate for overall statistics
                # all_variances.append(mean_variance)
                # all_means.append(overall_mean)
                
                # print(f"Batch {j}/{num_batches} - Shape: {generated_samples.shape}, Mean variance across images: {mean_variance:.6f}, Overall mean: {overall_mean:.6f}")
        
        # Print overall statistics across all batches
        if all_variances:
            avg_variance_across_batches = sum(all_variances) / len(all_variances)
            avg_mean_across_batches = sum(all_means) / len(all_means)
            print(f"\n=== OVERALL STATISTICS (Variance across different images) ===")
            print(f"Average variance across images: {avg_variance_across_batches:.6f}")
            print(f"Average of overall means: {avg_mean_across_batches:.6f}")
            print(f"Total batches processed: {len(all_variances)}")
            print(f"Total images: {len(all_variances) * self.H.imle_batch}\n")

        # #############                 print(generated_samples.shape) get variance here
            

    def generate_new_data_from_teacher(self, num_teacher_steps=20, seed=None):
        """
        Generate new data from teacher model with new conditions.
        Always generates exactly the number specified by teacher_num_samples parameter.
        
        Args:
            num_teacher_steps: Number of steps for the ODE solver
            seed: Optional seed for reproducible generation. If None, uses current RNG state.
        
        Returns:
            new_data: Generated images [num_samples, 3, H, W]
            new_conditions: New noise conditions [num_samples, 4, H//8, W//8]
        """
        if not self.use_teacher_resample:
            return None, None
        
        # Get number of samples to generate (default 100)
        num_samples = getattr(self.H, 'teacher_num_samples', 100)
        
        print(f"Generating {num_samples} new samples from teacher model...", flush=True)
        if seed is not None:
            print(f"  Using seed: {seed}", flush=True)
        
        import copy
        from torchdyn.core import NeuralODE
        
        # Generate new conditions (noise in latent space) with optional seeding
        latent_size = self.H.image_size // 8
        
        if seed is not None:
            # Create a generator with the specified seed for reproducibility
            generator = torch.Generator(device='cuda')
            generator.manual_seed(seed)
            new_conditions = torch.randn(
                num_samples, 4, latent_size, latent_size, 
                device='cuda', 
                generator=generator
            )
        else:
            # Use current RNG state (not reproducible across runs)
            new_conditions = torch.randn(num_samples, 4, latent_size, latent_size, device='cuda')
        
        # Generate data from teacher using flow matching
        teacher_copy = copy.deepcopy(self.teacher_model)
        node = NeuralODE(teacher_copy, solver="euler", sensitivity="adjoint")
        
        new_data_list = []
        batch_size = self.H.imle_batch
        
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                batch_conditions = new_conditions[i:end_idx]
                
                # Flow matching: noise → latents
                traj = node.trajectory(
                    batch_conditions,
                    t_span=torch.linspace(0, 1, num_teacher_steps, device='cuda'),
                )
                teacher_latents = traj[-1, :].view(batch_conditions.shape)
                
                # Decode latents → images
                teacher_images = self.teacher_vae.decode(
                    teacher_latents / self.teacher_vae.config.scaling_factor
                ).sample
                
                # Denormalize from [-1, 1] to [0, 1] and then to [0, 255]
                teacher_images = (teacher_images + 1) / 2
                teacher_images = torch.clamp(teacher_images, 0, 1) * 255
                
                # Convert to uint8 and permute to [B, H, W, C]
                teacher_images = teacher_images.byte().permute(0, 2, 3, 1).cpu()
                
                new_data_list.append(teacher_images)
                
                print(f"  Generated {end_idx}/{num_samples} samples...", flush=True)
        
        new_data = torch.cat(new_data_list, dim=0)
        
        print(f"Generated {num_samples} new samples from teacher. Shape: {new_data.shape}", flush=True)
        return new_data, new_conditions.cpu()

    def get_scheduled_teacher_resample_interval(self, current_epoch):
        """
        Compute the teacher resample interval based on a list-based schedule.
        
        With N+1 intervals and N transitions:
        - intervals[0] is used from epoch 0 to transitions[0]
        - intervals[1] is used from transitions[0] to transitions[1]
        - intervals[i] is used from transitions[i-1] to transitions[i]
        - intervals[N] is used from transitions[N-1] onwards
        
        Example:
          intervals = [800, 200, 100, 50]  (N+1 = 4 elements)
          transitions = [800, 2000, 6000]  (N = 3 elements)
          
          epoch 0-800: use 800 (resample every 800, so only at epoch 800)
          epoch 800-2000: use 200 (resample every 200)
          epoch 2000-6000: use 100 (resample every 100)
          epoch 6000+: use 50 (resample every 50)
        
        Returns the number of epochs until the next resample should occur.
        """
        if not hasattr(self.H, 'use_teacher_resample_schedule') or not self.H.use_teacher_resample_schedule:
            # If scheduling is disabled, return the fixed value
            return getattr(self.H, 'teacher_force_resample', None)
        
        # Use list-based scheduling if available
        if self.schedule_intervals is not None and self.schedule_transitions is not None:
            # Find which phase we're in based on current epoch
            for i, transition_epoch in enumerate(self.schedule_transitions):
                if current_epoch < transition_epoch:
                    return self.schedule_intervals[i]
            # If we're past all transitions, use the last interval
            return self.schedule_intervals[-1]
        
        # Fallback: if no lists provided, return None (shouldn't happen if validation works)
        return None
    
    def should_teacher_resample(self, current_epoch):
        """
        Determine if teacher resampling should occur at the current epoch.
        """
        if not self.use_teacher_resample:
            return False
        
        # Check if scheduling is enabled
        if hasattr(self.H, 'use_teacher_resample_schedule') and self.H.use_teacher_resample_schedule:
            # Use scheduled resampling
            interval = self.get_scheduled_teacher_resample_interval(current_epoch)
            epochs_since_last = current_epoch - self.last_teacher_resample_epoch
            
            if epochs_since_last >= interval:
                return True
            return False
        else:
            # Use fixed interval from teacher_force_resample
            teacher_force_resample = getattr(self.H, 'teacher_force_resample', None)

            # If initial dataset was generated at epoch 0, avoid triggering a
            # duplicate resample at epoch 0. This respects the flag
            # `H.teacher_generate_initial_data` which indicates an initial
            # generation already occurred.
            if current_epoch == 0 and getattr(self.H, 'teacher_generate_initial_data', False):
                return False

            if teacher_force_resample is not None and current_epoch is not None:
                # Force complete renewal every teacher_force_resample epochs
                if current_epoch % teacher_force_resample == 0:
                    return True
            else:
                # Default behavior: resample on every resampling step
                return True

            return False

    def imle_sample_force(self, dataset, gen, to_update=None, condition_data=None, epoch=None):
        if to_update is None:
            to_update = self.entire_ds
        if to_update.shape[0] == 0:
            return None, None  # Return None for new data and conditions
        
        to_update = to_update.cpu()

        t1 = time.time()
        new_data, new_conditions = None, None
        
        if torch.any(self.sample_pool_usage[to_update]):
            print("hello1")
            # Check if we should generate new data from teacher using the new scheduling logic
            if epoch is not None and self.should_teacher_resample(epoch):
                print("hello2")

                # Use global seed + epoch for deterministic teacher resampling
                teacher_seed = self.H.seed + epoch if hasattr(self.H, 'seed') else None
                new_data, new_conditions = self.generate_new_data_from_teacher(
                    num_teacher_steps=getattr(self.H, 'teacher_resample_steps', 20),
                    seed=teacher_seed
                )
                
                # Update the last resample epoch
                self.last_teacher_resample_epoch = epoch
                
                # Get the interval for logging
                if hasattr(self.H, 'use_teacher_resample_schedule') and self.H.use_teacher_resample_schedule:
                    interval = self.get_scheduled_teacher_resample_interval(epoch)
                    
                    # Determine which phase we're in
                    if self.schedule_intervals is not None and self.schedule_transitions is not None:
                        # Find current phase index
                        phase_idx = len(self.schedule_intervals) - 1  # default to last phase
                        for i, transition_epoch in enumerate(self.schedule_transitions):
                            if epoch < transition_epoch:
                                phase_idx = i
                                break
                        
                        # Check if this is the first resample (at transitions[0])
                        if epoch == self.schedule_transitions[0] and self.last_teacher_resample_epoch == 0:
                            phase_name = "initial"
                        else:
                            phase_name = f"phase{phase_idx + 1}"
                    else:
                        phase_name = "unknown"
                        phase_idx = 0
                    
                    print(f'[Epoch {epoch}] Forcing complete dataset renewal (scheduled: {phase_name}, interval={interval})')
                    if self.schedule_transitions:
                        print(f'  Schedule: intervals={self.schedule_intervals}, transitions={self.schedule_transitions}')
                else:
                    teacher_force_resample = getattr(self.H, 'teacher_force_resample', None)
                    print(f'[Epoch {epoch}] Forcing complete dataset renewal (teacher_force_resample={teacher_force_resample})')
                
                print(f'Teacher-based resampling generated new data: {new_data.shape if new_data is not None else None}')
            
            print(f'Starting resample_pool (generating {self.pool_size} samples from student model)...', flush=True)
            self.resample_pool(gen, dataset)
            self.sample_pool_usage[:] = False
            print(f'Resampling took {time.time() - t1} seconds', flush=True)

        self.selected_dists_tmp[:] = np.inf
        self.sample_pool_usage[to_update] = True

        # If conditions are provided, take a different path: iterate one sample at a time
        if condition_data is not None:
            print(f'Starting rejection sampling for {to_update.shape[0]} samples...', flush=True)
            total_rejected = 0
            with torch.no_grad():
                for i in range(to_update.shape[0]):
                    if i > 0 and i % 1000 == 0:
                        print(f'  Processed {i}/{to_update.shape[0]} samples...', flush=True)
                    idx = to_update[i]
                    if torch.is_tensor(idx):
                        idx_i = idx.item()
                    else:
                        idx_i = int(idx)

                    # Access ground-truth sample and corresponding condition
                    ff = int(self.H.force_factor)
                    pool_slice = slice(idx_i * ff, (idx_i + 1) * ff)

                    # Use precomputed projection
                    gt_feat = self.dataset_proj[idx_i]

                    pool_feats = self.pool_samples_proj[pool_slice]  
                    # print("labels of latents being checked: " , self.pool_condition_indices[pool_slice])
                    dists_label = torch.linalg.norm(pool_feats - gt_feat, dim=1)  

                    # Rejection sampling: mask distances below eps_radius if enabled
                    if getattr(self.H, 'use_rsimle', False):
                        reject_mask = dists_label < self.H.eps_radius
                        if torch.any(reject_mask):
                            total_rejected += int(reject_mask.sum().item())
                            dists_label = dists_label.masked_fill(reject_mask, float('inf'))

                    # Choose the minimum-distance candidate (after rejection)
                    j_local = torch.argmin(dists_label).item()

                    # Update selected latents (and snoise) from the chosen pool element
                    chosen_pool_idx = (idx_i * ff) + j_local
                    self.selected_latents_tmp[idx_i] = self.pool_latents[chosen_pool_idx].clone() + self.H.imle_perturb_coef * torch.randn(self.H.latent_dim)
                    if self.H.use_snoise:
                        for r in range(len(self.res)):
                            self.selected_snoise[r][idx_i] = self.snoise_pool[r][chosen_pool_idx].clone()
                            
            self.total_excluded = total_rejected
            self.total_excluded_percentage = (total_rejected * 1.0 / max(1, self.pool_size)) * 100
            
            # Copy updated latents from tmp to selected
            self.selected_latents[to_update] = self.selected_latents_tmp[to_update]
            
            print(f'Rejection sampling complete! Total rejected: {total_rejected}', flush=True)
            print(f'Force resampling took {time.time() - t1} seconds total', flush=True)
            
            # Stop collecting latent statistics after first resampling
            if hasattr(gen.module, 'stop_collecting_stats'):
                gen.module.stop_collecting_stats()
            
            return new_data, new_conditions

        
        # else: 
            
            # total_rejected = 0

            # if(self.H.use_rsimle):
            #     with torch.no_grad():
            #         for i in range(self.pool_size // self.H.imle_db_size):
            #             pool_slice = slice(i * self.H.imle_db_size, (i + 1) * self.H.imle_db_size)
            #             if not gen.module.dci_db:
            #                 device_count = torch.cuda.device_count()
            #                 gen.module.dci_db = MDCI(self.dci_dim, num_comp_indices=self.H.num_comp_indices,
            #                                             num_simp_indices=self.H.num_simp_indices, 
            #                                             devices=[i for i in range(device_count)])
            #             gen.module.dci_db.add(self.pool_samples_proj[pool_slice])
            #             pool_latents = self.pool_latents[pool_slice]
            #             snoise_pool = [b[pool_slice] for b in self.snoise_pool]

            #             rejected_flag = torch.zeros(self.H.imle_db_size, dtype=torch.bool)

            #             for ind, y in enumerate(DataLoader(TensorDataset(dataset[to_update]), batch_size=self.H.imle_batch)):
            #                 _, target = self.preprocess_fn(y)
            #                 batch_slice = slice(ind * self.H.imle_batch, ind * self.H.imle_batch + target.shape[0])
            #                 indices = to_update[batch_slice]
            #                 x = self.dataset_proj[indices]
            #                 nearest_indices, dci_dists = gen.module.dci_db.query(x.float(), num_neighbours=self.H.knn_ignore)
            #                 nearest_indices = nearest_indices.long()
            #                 check = dci_dists < self.H.eps_radius 
            #                 easy_samples_list = torch.unique(nearest_indices[check])
            #                 self.pool_samples_proj[pool_slice][easy_samples_list] = torch.tensor(float('inf'))
            #                 rejected_flag[easy_samples_list] = 1

            #             gen.module.dci_db.clear()
                        
            #             total_rejected += rejected_flag.sum().item()
            
            #     self.total_excluded = total_rejected
            #     self.total_excluded_percentage = (total_rejected * 1.0 / self.pool_size) * 100

            #     with torch.no_grad():
            #         for i in range(self.pool_size // self.H.imle_db_size):
            #             pool_slice = slice(i * self.H.imle_db_size, (i + 1) * self.H.imle_db_size)
            #             if not gen.module.dci_db:
            #                 device_count = torch.cuda.device_count()
            #                 gen.module.dci_db = MDCI(self.dci_dim, num_comp_indices=self.H.num_comp_indices,
            #                                             num_simp_indices=self.H.num_simp_indices, devices=[i for i in range(device_count)])
            #             gen.module.dci_db.add(self.pool_samples_proj[pool_slice])
            #             pool_latents = self.pool_latents[pool_slice]
            #             snoise_pool = [b[pool_slice] for b in self.snoise_pool]

            #             t0 = time.time()
            #             for ind, y in enumerate(DataLoader(TensorDataset(dataset[to_update]), batch_size=self.H.imle_batch)):
            #                 _, target = self.preprocess_fn(y)
            #                 batch_slice = slice(ind * self.H.imle_batch, ind * self.H.imle_batch + target.shape[0])
            #                 indices = to_update[batch_slice]
            #                 x = self.dataset_proj[indices]
            #                 nearest_indices, dci_dists = gen.module.dci_db.query(x.float(), num_neighbours=1)
            #                 nearest_indices = nearest_indices.long()[:, 0]
            #                 nearest_indices = nearest_indices.cpu()
            #                 dci_dists = dci_dists[:, 0]

            #                 need_update = dci_dists < self.selected_dists_tmp[indices]
            #                 need_update = need_update.cpu()
            #                 global_need_update = indices[need_update]

            #                 self.selected_dists_tmp[global_need_update] = dci_dists[need_update].clone()
            #                 self.selected_latents_tmp[global_need_update] = pool_latents[nearest_indices[need_update]].clone() + self.H.imle_perturb_coef * torch.randn((need_update.sum(), self.H.latent_dim))
            #                 for j in range(len(self.res)):
            #                     self.selected_snoise[j][global_need_update] = snoise_pool[j][nearest_indices[need_update]].clone()

            #             gen.module.dci_db.clear()

            #             if i % 100 == 0:
            #                 print("NN calculated for {} out of {} - {}".format((i + 1) * self.H.imle_db_size, self.pool_size, time.time() - t0))
                
                # self.selected_latents[to_update] = self.selected_latents_tmp[to_update]

        print(f'Force resampling took {time.time() - t1}')
        
        # Stop collecting latent statistics after first resampling
        if hasattr(gen.module, 'stop_collecting_stats'):
            gen.module.stop_collecting_stats()
        
        return new_data, new_conditions
    