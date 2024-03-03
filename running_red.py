import os
from argparse import ArgumentParser

import deepinv as dinv

import torch
import numpy as np
import matplotlib.pyplot as plt

from data import get_restoration_data
from utils.utils import get_model, get_physics, EquivariantDenoiser, to_image


def run_red(problem_type='sr', dataset_name='set3c', img_size=None, results_folder='results/', device='cpu',
            model_name='dncnn', rand_rotations=False, rand_translations=False,
            n_channels=3, sr=2, mean_rotations=False, sigma_den=0.01, save_image=True):

    torch.manual_seed(0)  # Reproducibility

    test_dataloader = get_restoration_data(dataset_name=dataset_name, img_size=img_size, n_channels=n_channels)

    for num_slice, batch in enumerate(test_dataloader):

        torch.cuda.empty_cache()

        with torch.no_grad():

            denoiser_backbone = get_model(model_name, device=device, channels=n_channels)
            denoiser = EquivariantDenoiser(denoiser_backbone, rand_rot=rand_rotations, mean_rot=mean_rotations,
                                           rand_translations=rand_translations, clamp=True)
            denoiser = denoiser.eval().to(device)

            max_iter = 10000 if not 'diffunet' in model_name else 500
            max_iter = 500 if 'swinir' in model_name else max_iter

            x_true, _ = batch
            x_true = x_true.to(device)
            lambd = 10.0 if 'diffunet' in model_name else 5.0

            step_size = 1/(1+lambd)

            params_algo = {"stepsize": step_size, "g_param": sigma_den, "lambda": lambd}

            physics = get_physics(x_true, n_channels, problem_type=problem_type, device=device, sr=sr, img_size=img_size)

            prior = dinv.optim.RED(denoiser)

            data_fid = dinv.optim.L2()

            # instantiate the algorithm class to solve the IP problem.
            model = dinv.optim.optim_builder(
                iteration="GD",
                prior=prior,
                data_fidelity=data_fid,
                early_stop=False,
                max_iter=max_iter,
                verbose=False,
                params_algo=params_algo
            ).to(device)

            y = physics(x_true)

            out, metrics = model(y, physics, compute_metrics=True, x_gt=x_true)

            psnr_mean = dinv.utils.cal_psnr(x_true, out)

            if save_image:
                plt.imsave(results_folder + 'out_' + str(num_slice) + '.png',
                           to_image(out)[0].detach().cpu().numpy(), cmap='viridis')

                plt.imsave(results_folder + 'in_' + str(num_slice) + '.png',
                           to_image(y)[0].detach().cpu().numpy(), cmap='viridis')

            with open(results_folder+'psnr_'+str(num_slice)+'.npy', 'wb') as f:
                np.save(f, metrics['psnr'])

            with open(results_folder+'crit_'+str(num_slice)+'.npy', 'wb') as f:
                np.save(f, metrics['residual'])

            with open(results_folder+'psnr_mean_'+str(num_slice)+'.npy', 'wb') as f:
                np.save(f, psnr_mean)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='dncnn')
    parser.add_argument('--dataset_name', type=str, default='set3c')
    parser.add_argument('--results_folder', type=str, default='results_ula/')
    parser.add_argument('--rand_rotations', type=int, default=0)
    parser.add_argument('--mean_rotations', type=int, default=0)
    parser.add_argument('--rand_translations', type=int, default=0)
    parser.add_argument('--sigma_den', type=float, default=0.02)
    parser.add_argument('--sr', type=int, default=2)
    parser.add_argument('--problem', type=str, default='gaussian_blur')
    parser.add_argument('--pth_dataset', type=str, default='')
    args = parser.parse_args()

    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"


    # Convert to bool
    rand_rotations = True if args.rand_rotations == 1 else False
    mean_rotations = True if args.mean_rotations == 1 else False
    rand_translations = True if args.rand_translations == 1 else False

    results_folder_base = args.results_folder

    if rand_rotations:
        results_folder = results_folder_base+args.problem+'_'+args.dataset_name+'/'+args.model_name+'_'+str(args.sigma_den)+'_randrot/'
    elif mean_rotations:
        results_folder = results_folder_base+args.problem+'_'+args.dataset_name+'/'+args.model_name+'_'+str(args.sigma_den)+'_meanrot/'
    elif rand_translations:
        results_folder = results_folder_base+args.problem+'_'+args.dataset_name+'/'+args.model_name+'_'+str(args.sigma_den)+'_rand_translations/'
    else:
        results_folder = results_folder_base+args.problem+'_'+args.dataset_name+'/'+args.model_name+'_'+str(args.sigma_den)+'/'

    if mean_rotations or rand_rotations:
        assert mean_rotations != rand_rotations, 'Choose either rand_rotations or mean_rotations'
    if (mean_rotations or rand_rotations) and rand_translations:
        raise NotImplementedError('rand translations only can be used')

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    if args.problem == 'sr' or args.problem == 'gaussian_blur' or args.problem == 'motion_blur':
        run_red(dataset_name=args.dataset_name, problem_type=args.problem, results_folder=results_folder, device=device,
                rand_rotations=rand_rotations, mean_rotations=mean_rotations, rand_translations=rand_translations,
                model_name=args.model_name, sigma_den=args.sigma_den, sr=args.sr)
    else:
        raise NotImplementedError
