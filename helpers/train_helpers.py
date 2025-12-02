from pathlib import Path

import torch
import numpy as np
import socket
import argparse
import os
import json
import subprocess
from hps import Hyperparams, parse_args_and_update_hparams, add_imle_arguments
from helpers.utils import (logger, maybe_download)
from helpers.seed_utils import seed_everything
from data import mkdir_p
from contextlib import contextmanager
import torch.distributed as dist
# from apex.optimizers import FusedAdam as AdamW
from torch.optim import AdamW
from models import IMLE
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR, StepLR, SequentialLR, CosineAnnealingWarmRestarts


def update_ema(imle, ema_imle, ema_rate):
    for p1, p2 in zip(imle.parameters(), ema_imle.parameters()):
        p2.data.mul_(ema_rate)
        p2.data.add_(p1.data * (1 - ema_rate))


def save_model(path, imle, ema_imle, optimizer, scheduler, H):
    torch.save(imle.state_dict(), f'{path}-model.th')
    torch.save(ema_imle.state_dict(), f'{path}-model-ema.th')
    torch.save(optimizer.state_dict(), f'{path}-opt.th')
    torch.save(scheduler.state_dict(), f'{path}-sched.th')
    from_log = os.path.join(H.save_dir, 'log.jsonl')
    to_log = f'{os.path.dirname(path)}/{os.path.basename(path)}-log.jsonl'
    subprocess.check_output(['cp', from_log, to_log])


def accumulate_stats(stats, frequency):
    z = {}
    for k in stats[-1]:
        if k in ['distortion_nans', 'rate_nans', 'skipped_updates', 'gcskip', 'loss_nans']:
            z[k] = np.sum([a[k] for a in stats[-frequency:]])
        elif k == 'grad_norm':
            vals = [a[k] for a in stats[-frequency:]]
            finites = np.array(vals)[np.isfinite(vals)]
            if len(finites) == 0:
                z[k] = 0.0
            else:
                z[k] = np.max(finites)
        elif k == 'loss':
            vals = [a[k] for a in stats[-frequency:]]
            finites = np.array(vals)[np.isfinite(vals)]
            z['loss'] = np.mean(vals)
            z['loss_filtered'] = np.mean(finites)
        elif k == 'iter_time':
            z[k] = stats[-1][k] if len(stats) < frequency else np.mean([a[k] for a in stats[-frequency:]])
        else:
            z[k] = np.mean([a[k] for a in stats[-frequency:]])
    return z


def linear_warmup(warmup_iters):
    def f(iteration):
        if warmup_iters <= 0:
            return 1.0  # No warmup, return full LR immediately
        return 1.0 if iteration > warmup_iters else iteration / warmup_iters
    return f


class PiecewiseConstantWithRestart:
    """
    Piecewise constant scheduler with gentle tail decay and restart at resample boundaries.
    
    Args:
        optimizer: PyTorch optimizer
        resample_period: Number of epochs between resamples (e.g., 800)
        base_lr: Base learning rate to restart to
        lr_min: Minimum learning rate floor
        tail_epochs: Number of epochs before resample to start decay (e.g., 200)
        tail_decay: Decay factor per epoch during tail (e.g., 0.99)
    """
    def __init__(self, optimizer, resample_period=800, base_lr=2e-4, lr_min=1e-5, 
                 tail_epochs=200, tail_decay=0.99):
        self.optimizer = optimizer
        self.resample_period = resample_period
        self.base_lr = base_lr
        self.lr_min = lr_min
        self.tail_epochs = tail_epochs
        self.tail_decay = tail_decay
        self.epoch = 0
        
    def step(self):
        """Step the scheduler (call once per epoch)"""
        pos = self.epoch % self.resample_period
        
        if pos == 0 and self.epoch > 0:
            # Just resampled - reset to base LR
            lr = self.base_lr
        elif pos < self.resample_period - self.tail_epochs:
            # Constant phase
            lr = self.base_lr
        else:
            # Tail decay phase
            decay_steps = pos - (self.resample_period - self.tail_epochs)
            lr = max(self.base_lr * (self.tail_decay ** decay_steps), self.lr_min)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.epoch += 1
        
    def state_dict(self):
        """Return state dict for checkpointing"""
        return {
            'epoch': self.epoch,
            'resample_period': self.resample_period,
            'base_lr': self.base_lr,
            'lr_min': self.lr_min,
            'tail_epochs': self.tail_epochs,
            'tail_decay': self.tail_decay,
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint"""
        self.epoch = state_dict['epoch']
        self.resample_period = state_dict['resample_period']
        self.base_lr = state_dict['base_lr']
        self.lr_min = state_dict['lr_min']
        self.tail_epochs = state_dict['tail_epochs']
        self.tail_decay = state_dict['tail_decay']



def distributed_maybe_download(path, local_rank, mpi_size):
    if not path.startswith('gs://'):
        return path
    filename = path[5:].replace('/', '-')
    with first_rank_first(local_rank, mpi_size):
        fp = maybe_download(path, filename)
    return fp


@contextmanager
def first_rank_first(local_rank, mpi_size):
    if mpi_size > 1 and local_rank > 0:
        dist.barrier()

    try:
        yield
    finally:
        if mpi_size > 1 and local_rank == 0:
            dist.barrier()


def setup_save_dirs(H):
    H.save_dir = os.path.join(H.save_dir, H.desc)
    mkdir_p(H.save_dir)
    mkdir_p(f'H.save_dir/fid')
    H.logdir = os.path.join(H.save_dir, 'log')


def set_up_hyperparams(s=None):
    H = Hyperparams()
    parser = argparse.ArgumentParser()
    parser = add_imle_arguments(parser)
    parse_args_and_update_hparams(H, parser, s=s)
    setup_save_dirs(H)
    logprint = logger(H.logdir)
    for i, k in enumerate(sorted(H)):
        logprint(type='hparam', key=k, value=H[k])
    
    # # # # # Comprehensive seeding for reproducibility
    deterministic_mode = getattr(H, 'deterministic_mode', False)
    seed_everything(H.seed, deterministic_mode=deterministic_mode)
    
    logprint('training model', H.desc, 'on', H.dataset)
    return H, logprint


def restore_params(model, path, local_rank, mpi_size, map_ddp=True, map_cpu=False, strict=True):
    state_dict = torch.load(distributed_maybe_download(path, local_rank, mpi_size), map_location='cpu' if map_cpu else None)
    if map_ddp:
        new_state_dict = {}
        l = len('module.')  
        for k in state_dict:
            if k.startswith('module.'):
                new_state_dict[k[l:]] = state_dict[k]
            else:
                new_state_dict[k] = state_dict[k]
        state_dict = new_state_dict
    model.load_state_dict(state_dict, strict=strict)


def restore_log(path, local_rank, mpi_size):
    loaded = [json.loads(l) for l in open(distributed_maybe_download(path, local_rank, mpi_size))]
    try:
        cur_eval_loss = min([z['elbo'] for z in loaded if 'type' in z and z['type'] == 'eval_loss'])
    except ValueError:
        cur_eval_loss = float('inf')
    starting_epoch = max([z['epoch'] for z in loaded if 'type' in z and z['type'] == 'train_loss'])
    iterate = max([z['step'] for z in loaded if 'type' in z and z['type'] == 'train_loss'])
    return cur_eval_loss, iterate, starting_epoch


def load_imle(H, logprint):
    imle = IMLE(H)
    if H.restore_path:
        logprint(f'Restoring imle from {H.restore_path}')
        restore_params(imle, H.restore_path, map_cpu=True, local_rank=H.local_rank, mpi_size=H.mpi_size, strict=H.load_strict)

    ema_imle = IMLE(H)
    if H.restore_ema_path:
        logprint(f'Restoring ema imle from {H.restore_ema_path}')
        restore_params(ema_imle, H.restore_ema_path, map_cpu=True, local_rank=H.local_rank, mpi_size=H.mpi_size, strict=H.load_strict)
    else:
        ema_imle.load_state_dict(imle.state_dict())
    ema_imle.requires_grad_(False)

    ema_imle = ema_imle.cuda()

    imle = imle.cuda()
    imle = torch.nn.DataParallel(imle)

    if len(list(imle.named_parameters())) != len(list(imle.parameters())):
        raise ValueError('Some params are not named. Please name all params.')
    total_params = 0
    for name, p in imle.named_parameters():
        total_params += np.prod(p.shape)
    logprint(total_params=total_params, readable=f'{total_params:,}')
    return imle, ema_imle


def load_opt(H, imle, logprint):
    optimizer = AdamW(imle.parameters(), weight_decay=H.wd, lr=H.lr, betas=(H.adam_beta1, H.adam_beta2))
    
    # Check if we should use restarting scheduler (controlled by hyperparameter)
    use_restarting_scheduler = getattr(H, 'use_restarting_scheduler', False)
    
    if use_restarting_scheduler:
        # Require teacher_force_resample to be set for restarting scheduler
        if getattr(H, 'teacher_force_resample', None) is None:
            raise ValueError('use_restarting_scheduler=True requires teacher_force_resample to be set')
        resample_period = H.teacher_force_resample
        eta_min = getattr(H, 'lr_min', 1e-5)
        scheduler_type = getattr(H, 'dynamic_scheduler_type', 'cosine')  # 'cosine' or 'piecewise'
        
        if scheduler_type == 'piecewise':
            # Option 2: Piecewise constant with tail decay and restart
            tail_epochs = getattr(H, 'lr_tail_epochs', 200)
            tail_decay = getattr(H, 'lr_tail_decay', 0.99)
            
            logprint(f'Using PiecewiseConstantWithRestart scheduler:')
            logprint(f'  - Resample period: {resample_period} epochs')
            logprint(f'  - Base LR: {H.lr}, Min LR: {eta_min}')
            logprint(f'  - Constant phase: epochs 0-{resample_period - tail_epochs}')
            logprint(f'  - Tail decay: last {tail_epochs} epochs with decay={tail_decay}')
            logprint(f'  - LR restarts to {H.lr} at each resample boundary')
            
            scheduler = PiecewiseConstantWithRestart(
                optimizer, 
                resample_period=resample_period,
                base_lr=H.lr,
                lr_min=eta_min,
                tail_epochs=tail_epochs,
                tail_decay=tail_decay
            )
        else:
            # Option 1: Cosine Annealing with Warm Restarts (default)
            logprint(f'Using CosineAnnealingWarmRestarts scheduler:')
            logprint(f'  - T_0={resample_period} epochs, eta_min={eta_min}')
            logprint(f'  - LR will smoothly decay from {H.lr} to {eta_min} over {resample_period} epochs')
            logprint(f'  - LR restarts to {H.lr} every {resample_period} epochs (aligned with teacher resampling)')
            
            # Note: CosineAnnealingWarmRestarts expects step() to be called per epoch
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=resample_period, T_mult=1, eta_min=eta_min)
    else:
        # Original scheduler for fixed dataset runs
        logprint('Using original StepLR scheduler (iteration-based)')
        scheduler1 = LambdaLR(optimizer, lr_lambda=linear_warmup(H.warmup_iters))
        scheduler2 = StepLR(optimizer, step_size=H.lr_decay_iters, gamma=H.lr_decay_rate)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[H.warmup_iters])

    if H.restore_optimizer_path:
        optimizer.load_state_dict(
            torch.load(distributed_maybe_download(H.restore_optimizer_path, H.local_rank, H.mpi_size), map_location='cpu', weights_only=False))
    if H.restore_scheduler_path:
        scheduler.load_state_dict(
            torch.load(distributed_maybe_download(H.restore_scheduler_path, H.local_rank, H.mpi_size), map_location='cpu', weights_only=False))
    if H.restore_log_path:
        cur_eval_loss, iterate, starting_epoch = restore_log(H.restore_log_path, H.local_rank, H.mpi_size)
    else:
        cur_eval_loss, iterate, starting_epoch = float('inf'), 0, 0
    logprint('starting at epoch', starting_epoch, 'iterate', iterate, 'eval loss', cur_eval_loss)
    return optimizer, scheduler, cur_eval_loss, iterate, starting_epoch


def save_latents(H, outer, split_ind, latents, name='latents'):
    Path("{}/latent/".format(H.save_dir)).mkdir(parents=True, exist_ok=True)
    # for ind, z in enumerate(latents):
    torch.save(latents, '{}/latent/{}-{}-{}.npy'.format(H.save_dir, outer, split_ind, name))


def save_snoise(H, outer, snoise):
    Path("{}/latent/".format(H.save_dir)).mkdir(parents=True, exist_ok=True)
    for sn in snoise:
        torch.save(sn, '{}/latent/snoise-{}-{}.npy'.format(H.save_dir, outer, sn.shape[2]))


def save_latents_latest(H, split_ind, latents, name='latest'):
    Path("{}/latent/".format(H.save_dir)).mkdir(parents=True, exist_ok=True)
    # for ind, z in enumerate(latents):
    torch.save(latents, '{}/latent/{}-{}.npy'.format(H.save_dir, split_ind, name))
