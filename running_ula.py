import os
from argparse import ArgumentParser

import deepinv as dinv

import torch
import numpy as np
import matplotlib.pyplot as plt

from data import get_restoration_data
from utils.utils import get_model, get_physics, EquivariantDenoiser, to_image


def run_ula(problem_type='sr', dataset_name='set3c', img_size=None, results_folder='results/', device='cpu',
            model_name='dncnn',rand_rotations=False, rand_translations=False,
            n_channels=3, sr=2, mean_rotations=False, sigma_den=0.01, save_image=True):

    torch.manual_seed(0)  # Reproducibility

    test_dataloader = get_restoration_data(dataset_name=dataset_name, img_size=img_size, n_channels=n_channels)

    for num_slice, batch in enumerate(test_dataloader):

        torch.cuda.empty_cache()

        with torch.no_grad():

            denoiser_backbone = get_model(model_name, device=device, channels=n_channels)
            denoiser_backbone = denoiser_backbone.eval().to(device)
            denoiser = EquivariantDenoiser(denoiser_backbone, rand_rot=rand_rotations, mean_rot=mean_rotations,
                                           rand_translations=rand_translations, clamp=True)
            denoiser = denoiser.eval().to(device)

            sigma = 0.01
            burnin = .001
            thinning = 20 if 'diffunet' in model_name else 10
            MC = int(10000/thinning)
            lambd = 2

            x_true, _ = batch
            x_true = x_true.to(device)

            physics = get_physics(x_true, n_channels, problem_type=problem_type, device=device, sr=sr, img_size=img_size)

            norm = physics.compute_norm(torch.randn_like(x_true))

            alpha = lambd*torch.tensor(norm, device=device)
            step_size = float(1. / (norm / (sigma ** 2) + alpha / (sigma_den ** 2)))

            prior = dinv.optim.ScorePrior(denoiser)

            data_fid = dinv.optim.L2(sigma)

            model = dinv.sampling.ULA(prior, data_fidelity=data_fid, step_size=step_size,
                                      sigma=sigma_den, alpha=alpha, verbose=True,
                                      max_iter=int(MC * thinning / (.95 - burnin)),
                                      thinning=thinning, save_chain=True, burnin_ratio=burnin, clip=(-1., 2),
                                      thresh_conv=1e-4)

            y = physics(x_true)

            mean, var = model(y, physics, x_init=y)

            chain = model.get_chain()
            psnr = np.zeros(len(chain))
            for i, xi in enumerate(chain):
                psnr[i] = dinv.utils.cal_psnr(x_true, xi)
            psnr_mean = dinv.utils.cal_psnr(x_true, mean)

            if save_image:
                plt.imsave(results_folder + 'sample_' + str(num_slice) + '_last.png',
                           to_image(chain[-1])[0].detach().cpu().numpy(), cmap='viridis')
                plt.imsave(results_folder + 'mean_' + str(num_slice) + '.png',
                           to_image(mean)[0].detach().cpu().numpy(), cmap='viridis')
                plt.imsave(results_folder+'var_'+str(num_slice) + '.png',
                           to_image(var, clamp=False, rescale=True)[0].detach().cpu().numpy(), cmap='viridis')

            with open(results_folder+'var_'+str(num_slice)+'.npy', 'wb') as f:
                np.save(f, var.detach().cpu().numpy())

            with open(results_folder+'psnr_'+str(num_slice)+'.npy', 'wb') as f:
                np.save(f, psnr)

            with open(results_folder+'psnr_mean_'+str(num_slice)+'.npy', 'wb') as f:
                np.save(f, psnr_mean)

        del model, chain, psnr, mean, var, y, prior, norm, physics, x_true, denoiser, denoiser_backbone


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='dncnn')
    parser.add_argument('--dataset_name', type=str, default='set3c')
    parser.add_argument('--results_folder', type=str, default='results_ula/')
    parser.add_argument('--rand_rotations', type=int, default=0)
    parser.add_argument('--mean_rotations', type=int, default=0)
    parser.add_argument('--rand_translations', type=int, default=0)
    parser.add_argument('--sigma_den', type=float, default=0.02)
    parser.add_argument('--problem', type=str, default='sr')
    parser.add_argument('--pth_dataset', type=str, default='')
    args = parser.parse_args()

    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

    # Convert to bool
    rand_rotations = True if args.rand_rotations == 1 else False
    mean_rotations = True if args.mean_rotations == 1 else False
    rand_translations = True if args.rand_translations == 1 else False

    if mean_rotations or rand_rotations:
        assert mean_rotations != rand_rotations, 'Choose either rand_rotations or mean_rotations'
    if (mean_rotations or rand_rotations) and rand_translations:
        raise NotImplementedError('rand translations only can be used')

    results_folder_base = args.results_folder

    if rand_rotations:
        results_folder = results_folder_base+args.problem+'_'+args.dataset_name+'/'+args.model_name+'_'+str(args.sigma_den)+'_randrot/'
    elif mean_rotations:
        results_folder = results_folder_base+args.problem+'_'+args.dataset_name+'/'+args.model_name+'_'+str(args.sigma_den)+'_meanrot/'
    elif rand_translations:
        results_folder = results_folder_base+args.problem+'_'+args.dataset_name+'/'+args.model_name+'_'+str(args.sigma_den)+'_rand_translations/'
    else:
        results_folder = results_folder_base+args.problem+'_'+args.dataset_name+'/'+args.model_name+'_'+str(args.sigma_den)+'/'

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    if args.problem == 'sr' or args.problem == 'gaussian_blur' or args.problem == 'motion_blur':
        run_ula(dataset_name=args.dataset_name, problem_type=args.problem, results_folder=results_folder, device=device,
                rand_rotations=rand_rotations, mean_rotations=mean_rotations, rand_translations=rand_translations,
                model_name=args.model_name, sigma_den=args.sigma_den)
    else:
        raise NotImplementedError
