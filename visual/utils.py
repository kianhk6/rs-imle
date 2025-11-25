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



def generate_for_NN(sampler, orig, initial, snoise, shape, ema_imle, fname, logprint, condition_data=None, save_to_file=False, experiment=None):
    mb = shape[0]
    initial = initial[:mb].cuda()
    if condition_data is not None:
        nns = sampler.sample(initial, ema_imle, snoise, condition_data=condition_data)
    else:
        nns = sampler.sample(initial, ema_imle, snoise)

    batches = [orig[:mb], nns]
    n_rows = len(batches)
    im = np.concatenate(batches, axis=0).reshape((n_rows, mb, *shape[1:])).transpose([0, 2, 1, 3, 4]).reshape(
        [n_rows * shape[1], mb * shape[2], 3])

    if save_to_file:
        logprint(f'printing samples to {fname}')
        imageio.imwrite(fname, im)
        # if experiment:
        #     experiment.log_image(fname, overwrite=False)
    else:
        logprint(f'generated samples (not saved to file, only for logging)')
        # if experiment:
        #     # Log numpy array directly to comet without saving to file
        #     experiment.log_image(im, name=fname, overwrite=False)
    
    return im


def generate_images_initial(H, sampler, orig, initial, snoise, shape, imle, ema_imle, fname, logprint, experiment=None, condition_data=None, save_to_file=False):
    mb = shape[0]
    initial = initial[:mb]
    if condition_data is not None:
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
            # Subsequent rows: use random noise as condition instead of actual condition_data
            random_cond = torch.randn_like(condition_data[:mb])

            batches.append(sampler.sample(temp_latent_rnds, imle, tmp_snoise, condition_data=random_cond))
            
        n_rows = len(batches)
        im = np.concatenate(batches, axis=0).reshape((n_rows, mb, *shape[1:])).transpose([0, 2, 1, 3, 4]).reshape(
            [n_rows * shape[1], mb * shape[2], 3])

        if save_to_file:
            logprint(f'printing samples to {fname}')
            imageio.imwrite(fname, im)
            if(experiment):
                experiment.log_image(fname, overwrite=False)
        else:
            logprint(f'generated samples (not saved to file, only for logging)')
            if(experiment):
                # Log numpy array directly to comet without saving to file
                experiment.log_image(im, name=fname, overwrite=False)
    else:
        batches = [orig[:mb], sampler.sample(initial, imle, snoise)]
        temp_latent_rnds = torch.randn([mb, H.latent_dim], dtype=torch.float32).cuda()
        for t in range(H.num_rows_visualize + 4):
            temp_latent_rnds.normal_()
            if(H.use_snoise == True):
                tmp_snoise = [s[:mb].normal_() for s in sampler.snoise_tmp]
            else:
                tmp_snoise = [s[:mb] for s in sampler.neutral_snoise]
            batches.append(sampler.sample(temp_latent_rnds, imle, tmp_snoise))
        n_rows = len(batches)
        im = np.concatenate(batches, axis=0).reshape((n_rows, mb, *shape[1:])).transpose([0, 2, 1, 3, 4]).reshape([n_rows * shape[1], mb * shape[2], 3])
        
        if save_to_file:
            logprint(f'printing samples to {fname}')
            imageio.imwrite(fname, im)
            if(experiment):
                experiment.log_image(fname, overwrite=False)
        else:
            logprint(f'generated samples (not saved to file, only for logging)')
            if(experiment):
                # Log numpy array directly to comet without saving to file
                experiment.log_image(im, name=fname, overwrite=False)



        

def generate_and_save(H, imle, sampler, n_samp, subdir='fid', condition_data=None):
    if condition_data is not None:
        delete_content_of_dir(f'{H.save_dir}/{subdir}')
        
        with torch.no_grad():
            temp_latent_rnds = torch.randn([H.imle_batch, H.latent_dim], dtype=torch.float32).cuda()
            for i in range(0, (n_samp // H.imle_batch)+1):
                
                batch_size = min(H.imle_batch, n_samp-i*H.imle_batch)
                
                # Skip if batch_size is 0 or negative
                if batch_size <= 0:
                    continue
                
                temp_latent_rnds.normal_()
                tmp_snoise = [s[:H.imle_batch].normal_() for s in sampler.snoise_tmp]
                
                # Generate random noise as conditions (not using actual condition_data)
                batch_condition_data = None
                if condition_data is not None:
                    # Get a sample to determine the condition shape
                    sample_cond = condition_data[0]
                    if isinstance(sample_cond, tuple):
                        sample_cond = sample_cond[0]
                    # Generate random noise with the same shape
                    batch_condition_data = torch.randn(batch_size, *sample_cond.shape, device='cuda')
                
                samp = sampler.sample(temp_latent_rnds[:batch_size], imle, [s[:batch_size] for s in tmp_snoise], condition_data=batch_condition_data)

                for j in range(batch_size):
                    imageio.imwrite(f'{H.save_dir}/{subdir}/{i * H.imle_batch + j}.png', samp[j])
    else:
        delete_content_of_dir(f'{H.save_dir}/{subdir}')    
        with torch.no_grad():
            temp_latent_rnds = torch.randn([H.imle_batch, H.latent_dim], dtype=torch.float32).cuda()
            for i in range(0, (n_samp // H.imle_batch)+1):
                
                batch_size = min(H.imle_batch, n_samp-i*H.imle_batch)
                
                # Skip if batch_size is 0 or negative
                if batch_size <= 0:
                    continue

                temp_latent_rnds.normal_()
                tmp_snoise = [s[:H.imle_batch].normal_() for s in sampler.snoise_tmp]
                samp = sampler.sample(temp_latent_rnds, imle, tmp_snoise)

                for j in range(batch_size):
                    imageio.imwrite(f'{H.save_dir}/{subdir}/{i * H.imle_batch + j}.png', samp[j])
