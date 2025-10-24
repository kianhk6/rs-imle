from curses import update_lines_cols
from math import comb, ceil
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from LPNet import LPNet
from dciknn_cuda import DCI, MDCI
from torch.optim import AdamW
from helpers.utils import ZippedDataset
from models import parse_layer_string



class Sampler:
    def __init__(self, H, sz, preprocess_fn, condition_config=None):
        self.pool_size = ceil(int(H.force_factor * sz) / H.imle_db_size) * H.imle_db_size
        self.preprocess_fn = preprocess_fn
        self.l2_loss = torch.nn.MSELoss(reduce=False).cuda()
        self.H = H
        self.latent_lr = H.latent_lr
        self.entire_ds = torch.arange(sz)
        self.selected_latents = torch.empty([sz, H.latent_dim], dtype=torch.float32)
        self.selected_latents_tmp = torch.empty([sz, H.latent_dim], dtype=torch.float32)

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

        # print("hellooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo")
        # where you sample: sample something 32 * 32, when passing into network reshape it 
        self.temp_latent_rnds = torch.empty([self.H.imle_db_size, self.H.latent_dim], dtype=torch.float32)
        self.temp_samples = torch.empty([self.H.imle_db_size, H.image_channels, self.H.image_size, self.H.image_size],
                                        dtype=torch.float32)

        # print("hellooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo")

        self.pool_latents = torch.randn([self.pool_size, H.latent_dim], dtype=torch.float32)
        self.sample_pool_usage = torch.ones([sz], dtype=torch.bool)

        self.projections = []
        self.lpips_net = LPNet(pnet_type=H.lpips_net, path=H.lpips_path).cuda()
        self.l2_projection = None

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

        self.dci_dim = sum_dims
        print('dci_dim', self.dci_dim)

        self.temp_samples_proj = torch.empty([self.H.imle_db_size, sum_dims], dtype=torch.float32).cuda()
        self.dataset_proj = torch.empty([sz, sum_dims], dtype=torch.float32)
        self.pool_samples_proj = torch.empty([self.pool_size, sum_dims], dtype=torch.float32)
        self.pool_condition_data = None
        self.pool_condition_indices = None

        self.condition_config = None
        if condition_config is not None:
            self.configure_conditions(sz=sz, **condition_config)

    def configure_conditions(
        self,
        condition_tensor=None,
        force_factor=None,
        base_indices=None,
        device=None,
        sz=None
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
        if expected_pool != self.pool_size:
            raise ValueError(
                f"Provided condition tensor of length {data_size} with force_factor {sample_factor} "
                f"does not match pool_size {self.pool_size}"
            )

        interleaved_conditions = torch.repeat_interleave(condition_tensor, sample_factor, dim=0)
        # print("sample_factor", sample_factor)
        # print("shape", interleaved_conditions.shape)
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
        
        if conditions is True:
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
        
        
        for j in range(self.pool_size // self.H.imle_batch):
            batch_slice = slice(j * self.H.imle_batch, (j + 1) * self.H.imle_batch)
            cur_latents = self.pool_latents[batch_slice]
            
            cur_snosie = [s[batch_slice] for s in self.snoise_pool]

            with torch.no_grad():
                # Generate samples with or without condition data
                if conditions:
                    # Take the contiguous slice from the prebuilt interleaved list
                    batch_conditions = interleaved_conditions[batch_slice]
                    print(batch_slice)
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
            

    def imle_sample_force(self, dataset, gen, to_update=None, condition_data=None):
        if to_update is None:
            to_update = self.entire_ds
        if to_update.shape[0] == 0:
            return
        
        to_update = to_update.cpu()

        t1 = time.time()
        if torch.any(self.sample_pool_usage[to_update]):
            self.resample_pool(gen, dataset)
            self.sample_pool_usage[:] = False
            print(f'resampling took {time.time() - t1}')

        self.selected_dists_tmp[:] = np.inf
        self.sample_pool_usage[to_update] = True

        # If conditions are provided, take a different path: iterate one sample at a time
        if condition_data is not None:
            total_rejected = 0
            with torch.no_grad():
                for i in range(to_update.shape[0]):
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
            
            print(f'Force resampling took {time.time() - t1}')
            return

        
        else: 
            total_rejected = 0

            if(self.H.use_rsimle):
                with torch.no_grad():
                    for i in range(self.pool_size // self.H.imle_db_size):
                        pool_slice = slice(i * self.H.imle_db_size, (i + 1) * self.H.imle_db_size)
                        if not gen.module.dci_db:
                            device_count = torch.cuda.device_count()
                            gen.module.dci_db = MDCI(self.dci_dim, num_comp_indices=self.H.num_comp_indices,
                                                        num_simp_indices=self.H.num_simp_indices, 
                                                        devices=[i for i in range(device_count)])
                        gen.module.dci_db.add(self.pool_samples_proj[pool_slice])
                        pool_latents = self.pool_latents[pool_slice]
                        snoise_pool = [b[pool_slice] for b in self.snoise_pool]

                        rejected_flag = torch.zeros(self.H.imle_db_size, dtype=torch.bool)

                        for ind, y in enumerate(DataLoader(TensorDataset(dataset[to_update]), batch_size=self.H.imle_batch)):
                            _, target = self.preprocess_fn(y)
                            batch_slice = slice(ind * self.H.imle_batch, ind * self.H.imle_batch + target.shape[0])
                            indices = to_update[batch_slice]
                            x = self.dataset_proj[indices]
                            nearest_indices, dci_dists = gen.module.dci_db.query(x.float(), num_neighbours=self.H.knn_ignore)
                            nearest_indices = nearest_indices.long()
                            check = dci_dists < self.H.eps_radius 
                            easy_samples_list = torch.unique(nearest_indices[check])
                            self.pool_samples_proj[pool_slice][easy_samples_list] = torch.tensor(float('inf'))
                            rejected_flag[easy_samples_list] = 1

                        gen.module.dci_db.clear()
                        
                        total_rejected += rejected_flag.sum().item()
            
            self.total_excluded = total_rejected
            self.total_excluded_percentage = (total_rejected * 1.0 / self.pool_size) * 100

            with torch.no_grad():
                for i in range(self.pool_size // self.H.imle_db_size):
                    pool_slice = slice(i * self.H.imle_db_size, (i + 1) * self.H.imle_db_size)
                    if not gen.module.dci_db:
                        device_count = torch.cuda.device_count()
                        gen.module.dci_db = MDCI(self.dci_dim, num_comp_indices=self.H.num_comp_indices,
                                                    num_simp_indices=self.H.num_simp_indices, devices=[i for i in range(device_count)])
                    gen.module.dci_db.add(self.pool_samples_proj[pool_slice])
                    pool_latents = self.pool_latents[pool_slice]
                    snoise_pool = [b[pool_slice] for b in self.snoise_pool]

                    t0 = time.time()
                    for ind, y in enumerate(DataLoader(TensorDataset(dataset[to_update]), batch_size=self.H.imle_batch)):
                        _, target = self.preprocess_fn(y)
                        batch_slice = slice(ind * self.H.imle_batch, ind * self.H.imle_batch + target.shape[0])
                        indices = to_update[batch_slice]
                        x = self.dataset_proj[indices]
                        nearest_indices, dci_dists = gen.module.dci_db.query(x.float(), num_neighbours=1)
                        nearest_indices = nearest_indices.long()[:, 0]
                        nearest_indices = nearest_indices.cpu()
                        dci_dists = dci_dists[:, 0]

                        need_update = dci_dists < self.selected_dists_tmp[indices]
                        need_update = need_update.cpu()
                        global_need_update = indices[need_update]

                        self.selected_dists_tmp[global_need_update] = dci_dists[need_update].clone()
                        self.selected_latents_tmp[global_need_update] = pool_latents[nearest_indices[need_update]].clone() + self.H.imle_perturb_coef * torch.randn((need_update.sum(), self.H.latent_dim))
                        for j in range(len(self.res)):
                            self.selected_snoise[j][global_need_update] = snoise_pool[j][nearest_indices[need_update]].clone()

                    gen.module.dci_db.clear()

                    if i % 100 == 0:
                        print("NN calculated for {} out of {} - {}".format((i + 1) * self.H.imle_db_size, self.pool_size, time.time() - t0))
            
            self.selected_latents[to_update] = self.selected_latents_tmp[to_update]

        print(f'Force resampling took {time.time() - t1}')
    
    def imle_sample_force_conditional(self, dataset, gen, to_update=None):
        """
        For each real sample index in `to_update`, generate `force_factor` candidates
        and select the nearest only within that sample's own candidate group.

        This avoids global mixing and enforces per-data matching.
        """
        if to_update is None:
            to_update = self.entire_ds
        if to_update.shape[0] == 0:
            return

        # Ensure CPU tensor for indexing and also create a CUDA copy when updating CUDA tensors
        to_update = to_update.cpu()

        ff = max(1, int(self.H.force_factor))
        # Keep candidate batch within the same memory budget as imle_db_size
        chunk_size = max(1, self.H.imle_db_size // ff)

        # Reset temporary distances on the subset we are updating
        self.selected_dists_tmp[to_update.cuda()] = float('inf')

        t1 = time.time()
        micro = max(1, int(getattr(self.H, 'cond_micro_batch', 16)))
        with torch.no_grad():
            for start in range(0, to_update.shape[0], chunk_size):
                end = min(start + chunk_size, to_update.shape[0])
                indices = to_update[start:end]  # CPU LongTensor
                if indices.numel() == 0:
                    continue

                cur_chunk = indices.shape[0]
                real_proj_chunk = self.dataset_proj[indices].cuda()

                # For each sample, stream its group in micro-batches
                for i in range(cur_chunk):
                    best_dist = float('inf')
                    best_latent = None
                    best_snoise = None
                    accepted = False

                    rp = real_proj_chunk[i:i+1]  # [1, d]

                    for k in range(0, ff, micro):
                        m = min(micro, ff - k)
                        z = torch.randn(m, self.H.latent_dim, dtype=torch.float32)
                        if self.H.use_snoise:
                            s_micro = [torch.randn(m, 1, s, s, dtype=torch.float32) for s in self.res]
                        else:
                            s_micro = [torch.zeros(m, 1, s, s, dtype=torch.float32) for s in self.res]

                        out = gen(z, s_micro)

                        if (self.H.search_type == 'lpips'):
                            p = self.get_projected(out, False)
                        elif (self.H.search_type == 'l2'):
                            p = self.get_l2_feature(out, False)
                        else:
                            p = self.get_combined_feature(out, False)

                        dists = torch.linalg.norm(p - rp, dim=1)

                        if getattr(self.H, 'use_rsimle', False):
                            reject_mask = dists < self.H.eps_radius
                            if torch.all(reject_mask):
                                del out, p, dists
                                continue
                            dists = dists.masked_fill(reject_mask, float('inf'))

                        j = torch.argmin(dists).item()
                        d_best = dists[j].item()
                        if d_best != float('inf') and d_best < best_dist:
                            best_dist = d_best
                            best_latent = z[j].clone()
                            best_snoise = [s[j].clone() for s in s_micro]
                            accepted = True

                        del out, p, dists

                    if not accepted:
                        raise RuntimeError(
                            f"All {ff} candidates rejected for sample index {indices[i].item()} at eps_radius={self.H.eps_radius}"
                        )

                    idx = indices[i]
                    self.selected_dists_tmp[idx.cuda()] = torch.tensor(best_dist, device='cuda')
                    self.selected_latents_tmp[idx] = best_latent
                    for j in range(len(self.res)):
                        self.selected_snoise[j][idx] = best_snoise[j]

        # Commit updates
        self.selected_latents[to_update] = self.selected_latents_tmp[to_update]

        # Track exclusion metrics for logging consistency
        # self.total_excluded = total_rejected
        # if processed_candidates > 0:
        #     self.total_excluded_percentage = (total_rejected * 1.0 / processed_candidates) * 100
        # else:
        #     self.total_excluded_percentage = 0

        print(f'Conditional force resampling took {time.time() - t1}')

