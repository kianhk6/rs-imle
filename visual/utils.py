import torch
from torch.utils.data import DataLoader
import numpy as np
import imageio
import os
import shutil


def delete_content_of_dir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def get_sample_for_visualization(data, preprocess_fn, num, dataset):
    for x in DataLoader(data, batch_size=num):
        break
    orig_image = (x[0] * 255.0).to(torch.uint8).permute(0, 2, 3, 1) if dataset == 'ffhq_1024' else x[0]
    preprocessed = preprocess_fn(x)[0]
    return orig_image, preprocessed



def generate_for_NN(sampler, orig, initial, snoise, shape, ema_imle, fname, logprint, condition_data=None):
    mb = shape[0]
    initial = initial[:mb].cuda()
    nns = sampler.sample(initial, ema_imle, snoise, condition_data=condition_data)
    batches = [orig[:mb], nns]
    n_rows = len(batches)
    im = np.concatenate(batches, axis=0).reshape((n_rows, mb, *shape[1:])).transpose([0, 2, 1, 3, 4]).reshape(
        [n_rows * shape[1], mb * shape[2], 3])

    logprint(f'printing samples to {fname}')
    imageio.imwrite(fname, im)


def generate_images_initial(H, sampler, orig, initial, snoise, shape, imle, ema_imle, fname, logprint, experiment=None, condition_data=None):
    mb = shape[0]
    initial = initial[:mb]
    batches = [orig[:mb]]
    # Row 1: with matching condition slice if provided
    first_cond = condition_data[:mb] if condition_data is not None else None
    batches.append(sampler.sample(initial, imle, snoise, condition_data=first_cond))

    temp_latent_rnds = torch.randn([mb, H.latent_dim], dtype=torch.float32).cuda()
    for t in range(H.num_rows_visualize + 4):
        temp_latent_rnds.normal_()
        if(H.use_snoise == True):
            tmp_snoise = [s[:mb].normal_() for s in sampler.snoise_tmp]
        else:
            tmp_snoise = [s[:mb] for s in sampler.neutral_snoise]
        # Subsequent rows: step through next condition windows of size mb
        if condition_data is not None:
            start = (t + 1) * mb
            end = start + mb
            if condition_data.shape[0] >= end:
                cur_cond = condition_data[start:end]
            else:
                idx = torch.arange(start, end) % max(1, condition_data.shape[0])
                cur_cond = condition_data[idx]
        else:
            cur_cond = None
        batches.append(sampler.sample(temp_latent_rnds, imle, tmp_snoise, condition_data=cur_cond))
        
    n_rows = len(batches)
    im = np.concatenate(batches, axis=0).reshape((n_rows, mb, *shape[1:])).transpose([0, 2, 1, 3, 4]).reshape(
        [n_rows * shape[1], mb * shape[2], 3])

    logprint(f'printing samples to {fname}')
    imageio.imwrite(fname, im)
    if(experiment):
        experiment.log_image(fname, overwrite=True)

def generate_and_save(H, imle, sampler, n_samp, subdir='fid'):

    delete_content_of_dir(f'{H.save_dir}/{subdir}')
    
    with torch.no_grad():
        temp_latent_rnds = torch.randn([H.imle_batch, H.latent_dim], dtype=torch.float32).cuda()
        for i in range(0, (n_samp // H.imle_batch)+1):
            
            batch_size = min(H.imle_batch, n_samp-i*H.imle_batch)

            temp_latent_rnds.normal_()
            tmp_snoise = [s[:H.imle_batch].normal_() for s in sampler.snoise_tmp]
            samp = sampler.sample(temp_latent_rnds, imle, tmp_snoise)

            for j in range(batch_size):
                imageio.imwrite(f'{H.save_dir}/{subdir}/{i * H.imle_batch + j}.png', samp[j])
