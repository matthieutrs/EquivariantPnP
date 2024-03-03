import os
from argparse import ArgumentParser

import numpy as np
import torch

import matplotlib.pyplot as plt

import deepinv as dinv
from deepinv.optim.data_fidelity import L2
from deepinv.utils import cal_psnr
from deepinv.loss import JacobianSpectralNorm

from data import get_mri_data, create_vertical_lines_mask, get_restoration_data

from utils.utils import get_model, denoise_complex, to_image, get_physics, EquivariantDenoiser


def run_experiment_mri(img_size=128, results_folder='results/', device='cpu', model_name='dncnn',
                       num_iter=10001, num_test_samples=10, rand_rotations=False, acc=4, mean_rotations=False,
                       sigma=0.01, save_image=True):

    torch.manual_seed(0)  # Reproducibility

    test_dataloader = get_mri_data(img_size=img_size)

    data_fidelity = L2()
    denoiser = get_model(model_name, device=device, channels=1)

    psnr_array = np.zeros((num_test_samples, num_iter))
    crit_array = np.zeros((num_test_samples, num_iter))

    for num_slice, batch in enumerate(test_dataloader):

        x_true = batch.to(device)
        mask = create_vertical_lines_mask(image_shape=(img_size, img_size), acceleration_factor=acc, seed=0)
        physics = dinv.physics.MRI(mask=mask, device=device)

        y = physics(x_true)

        backproj = physics.A_adjoint(y)

        plt.imsave(results_folder+'backproj_'+str(num_slice)+'.png', to_image(backproj)[0].cpu().numpy(), cmap='viridis')
        plt.imsave(results_folder+'target_'+str(num_slice)+'.png', to_image(x_true)[0].cpu().numpy(), cmap='viridis')

        x = backproj.clone()
        gamma = 1.0

        for it in range(num_iter):
            x_prev = x.clone()

            # PnP iteration
            x = x - gamma*data_fidelity.grad(x, y, physics)
            x = denoise_complex(denoiser, x, sigma, rand_rot=rand_rotations, mean_rot=mean_rotations)

            # Compute metrics
            psnr = cal_psnr(to_image(x), to_image(x_true))
            psnr_array[num_slice, it] = psnr
            crit = torch.linalg.norm(x.flatten()-x_prev.flatten())
            crit_array[num_slice, it] = crit

        if save_image:
            plt.imsave(results_folder+'slice_'+str(num_slice) + '.png', to_image(x)[0].cpu().numpy(), cmap='viridis')

        with open(results_folder+'metrics.npy', 'wb') as f:
            np.save(f, psnr_array)

        with open(results_folder+'criterion.npy', 'wb') as f:
            np.save(f, crit_array)


def run_experiment_restoration(problem_type='sr', dataset_name='set3c', img_size=None, results_folder='results/',
                               device='cpu', model_name='dncnn', num_iter=10001, num_test_samples=10,
                               rand_rotations=False, rand_translations=False,
                               n_channels=3, sr=2, mean_rotations=False, compute_lip=False, sigma=0.01,
                               slice_idx=None):

    torch.manual_seed(0)  # Reproducibility

    test_dataloader = get_restoration_data(dataset_name=dataset_name, img_size=img_size, n_channels=n_channels)

    data_fidelity = L2()
    denoiser_backbone = get_model(model_name, device=device, channels=n_channels)
    denoiser = EquivariantDenoiser(denoiser_backbone, rand_rot=rand_rotations, mean_rot=mean_rotations,
                                   rand_translations=rand_translations, clamp=True)

    num_test_samples_max = min(num_test_samples, len(test_dataloader))

    psnr_array = np.zeros((num_test_samples_max, num_iter))
    crit_array = np.zeros((num_test_samples_max, num_iter))
    lip_den_array = np.zeros((num_test_samples_max, num_iter))
    lip_step_array = np.zeros((num_test_samples_max, num_iter))

    for num_slice, batch in enumerate(test_dataloader):

        x_true, _ = batch
        x_true = x_true.to(device)

        physics = get_physics(x_true, n_channels, problem_type=problem_type, device=device, sr=sr, img_size=img_size)

        y = physics(x_true)
        backproj = physics.A_adjoint(y)

        plt.imsave(results_folder+'backproj_'+str(num_slice)+'.png', to_image(backproj)[0].cpu().numpy(), cmap='viridis')
        plt.imsave(results_folder+'target_'+str(num_slice)+'.png', to_image(x_true)[0].cpu().numpy(), cmap='viridis')

        x = backproj.clone()

        gamma = 1.0/physics.compute_norm(x_true)

        if compute_lip:
            compute_lip_fn = JacobianSpectralNorm(max_iter=100, tol=1e-3, eval_mode=True, verbose=False)

        for it in range(num_iter):
            x_prev = x.clone()

            if compute_lip:
                x_prev.detach_().requires_grad_()
                retain_grad = True
            else:
                retain_grad = False

            # PnP iteration
            u = x_prev - gamma*data_fidelity.grad(x_prev, y, physics)
            x = denoiser(u, sigma, retain_grad=retain_grad)
            x[x > 1.] = 1.
            x[x < 0.] = 0.

            # Compute Lipschitz constant
            if compute_lip:
                lip_den = compute_lip_fn(x, x_prev)
                lip_step = compute_lip_fn(x, u)
                lip_step_array[num_slice, it] = lip_step.item()
                lip_den_array[num_slice, it] = lip_den.item()

            # Compute metrics
            psnr = cal_psnr(to_image(x), to_image(x_true))
            crit = torch.linalg.norm(x.flatten()-x_prev.flatten())

            if slice_idx is None:
                psnr_array[num_slice, it] = psnr
                crit_array[num_slice, it] = crit
            else:
                psnr_array[0, it] = psnr
                crit_array[0, it] = crit

            if it % 1000 == 0 or (it == 100 or it == 500 or it == 50):
                plt.imsave(results_folder+'slice_'+str(num_slice)+'_it_' + str(it) + '.png', to_image(x)[0].detach().cpu().numpy(), cmap='viridis')

        with open(results_folder+'lip.npy', 'wb') as f:
            np.save(f, lip_den_array)

        with open(results_folder+'lip_net.npy', 'wb') as f:
            np.save(f, lip_step_array)

        with open(results_folder+'metrics.npy', 'wb') as f:
            np.save(f, psnr_array)

        with open(results_folder+'criterion.npy', 'wb') as f:
            np.save(f, crit_array)



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='dncnn')
    parser.add_argument('--dataset_name', type=str, default='set3c')
    parser.add_argument('--results_folder', type=str, default='results/')
    parser.add_argument('--rand_rotations', type=int, default=0)
    parser.add_argument('--mean_rotations', type=int, default=0)
    parser.add_argument('--rand_translations', type=int, default=0)
    parser.add_argument('--sigma_den', type=float, default=0.01)
    parser.add_argument('--noise_level', type=float, default=0.01)  # BEWARE BEFORE CHANGING THIS
    parser.add_argument('--slice_idx', type=int, default=0)
    parser.add_argument('--acc', type=int, default=4)
    parser.add_argument('--problem', type=str, default='mri')
    parser.add_argument('--compute_lip', type=int, default=0)
    args = parser.parse_args()

    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

    # Convert to bool
    rand_rotations = True if args.rand_rotations == 1 else False
    mean_rotations = True if args.mean_rotations == 1 else False
    rand_translations = True if args.rand_translations == 1 else False
    compute_lip = True if args.compute_lip == 1 else False

    if mean_rotations or rand_rotations:
        assert mean_rotations != rand_rotations, 'Choose either rand_rotations or mean_rotations'
    if (mean_rotations or rand_rotations) and rand_translations:
        raise NotImplementedError('rand translations only can be used')

    results_folder_base = args.results_folder

    if rand_rotations:
        results_folder = results_folder_base+args.problem+'_'+args.dataset_name+'_'+str(args.noise_level)+'/'+args.model_name+'_'+str(args.acc)+'_'+str(args.sigma_den)+'_randrot/'
    elif mean_rotations:
        results_folder = results_folder_base+args.problem+'_'+args.dataset_name+'_'+str(args.noise_level)+'/'+args.model_name+'_'+str(args.acc)+'_'+str(args.sigma_den)+'_meanrot/'
    elif rand_translations:
        results_folder = results_folder_base+args.problem+'_'+args.dataset_name+'_'+str(args.noise_level)+'/'+args.model_name+'_'+str(args.acc)+'_'+str(args.sigma_den)+'_rand_translations/'
    else:
        results_folder = results_folder_base+args.problem+'_'+args.dataset_name+'_'+str(args.noise_level)+'/'+args.model_name+'_'+str(args.acc)+'_'+str(args.sigma_den)+'/'


    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    num_iter = 10001
    slice_idx = args.slice_idx if args.model_name == 'swinir' else None

    if args.problem == 'mri':
        run_experiment_mri(results_folder=results_folder, device=device, rand_rotations=rand_rotations,
                           sigma=args.sigma_den, mean_rotations=mean_rotations, acc=args.acc, img_size=320,
                           model_name=args.model_name)
    elif args.problem == 'sr' or args.problem == 'gaussian_blur' or args.problem == 'motion_blur':
        run_experiment_restoration(dataset_name=args.dataset_name, problem_type=args.problem, results_folder=results_folder,
                                   device=device, rand_rotations=rand_rotations, mean_rotations=mean_rotations,
                                   rand_translations=rand_translations, model_name=args.model_name,
                                   compute_lip=compute_lip, sigma=args.sigma_den, num_iter=num_iter, slice_idx=slice_idx)
    else:
        raise NotImplementedError
