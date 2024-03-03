import numpy as np
import torch

import deepinv as dinv
from data import get_mask_sr, get_blur_kernel


class EquivariantDenoiser(torch.nn.Module):
    def __init__(self, denoiser, rand_rot=False, mean_rot=False, rand_translations=False, clamp=True, retain_grad=False):
        super().__init__()
        self.denoiser = denoiser
        self.rand_rot = rand_rot
        self.mean_rot = mean_rot
        self.rand_translations = rand_translations
        self.clamp = clamp

    def forward(self, x, sigma, retain_grad=False):
        return denoise_rotate(self.denoiser, x, sigma, rand_rot=self.rand_rot, mean_rot=self.mean_rot,
                              rand_translations=self.rand_translations, clamp=self.clamp, retain_grad=retain_grad)


class ComplexDenoiser(torch.nn.Module):
    def __init__(self, denoiser, clamp=True):
        super().__init__()
        self.denoiser = denoiser
        self.clamp = clamp

    def forward(self, x, sigma):
        denoised_real = self.denoiser(x[:, 0:1, ...], sigma)
        denoised_cplx = self.denoiser(x[:, 1:2, ...], sigma)
        return torch.cat((denoised_real, denoised_cplx), axis=1)


def to_complex(x):
    x_ = torch.moveaxis(x, 1, -1).contiguous()
    return torch.view_as_complex(x_)


def to_image(x, clamp=True, rescale=False):
    if x.shape[1] == 2:
        x_complex = to_complex(x).contiguous()
        out = torch.abs(x_complex)
    elif x.shape[1] == 1:
        out = x[:, 0, ...]  # keep batch dim
        out = torch.nan_to_num(out)
        if clamp:
            out = torch.clamp(out, 0, 1)
    else:
        out = torch.moveaxis(x, 1, -1).contiguous()
        out = torch.nan_to_num(out)
        if clamp:
            out = torch.clamp(out, 0, 1)
        if rescale:
            out = out - out.min()
            out = out/out.max()
    return out


def denoise_complex(denoiser, image, sigma_real, sigma_cplx=None, rand_rot=False, mean_rot=False):
    r'''
    Case of complex images

    :param denoiser:
    :param image:
    :param sigma_real:
    :param sigma_cplx:
    :param rot_idx:
    :return:
    '''
    if rand_rot:
        k = np.random.choice([0, 1, 2, 3])
        denoised = denoise_complex_rotate(denoiser, image, sigma_real, sigma_cplx=sigma_cplx, rot_idx=k)
    elif mean_rot:
        denoised = 0*image
        for k in range(4):
            denoised = denoised + denoise_complex_rotate(denoiser, image, sigma_real, sigma_cplx=sigma_cplx, rot_idx=k)
        denoised = denoised/4.
    else:  # normal denoising
        denoised = denoise_complex_rotate(denoiser, image, sigma_real, sigma_cplx=sigma_cplx, rot_idx=0)
    return denoised


def denoise_complex_rotate(denoiser, image, sigma_real, sigma_cplx=None, rot_idx=0):
    r'''
    Case of complex images

    :param denoiser:
    :param image:
    :param sigma_real:
    :param sigma_cplx:
    :param rot_idx:
    :return:
    '''
    image = torch.rot90(image, k=rot_idx, dims=[-2, -1])
    with torch.no_grad():
        if sigma_cplx is None:
            sigma_cplx = sigma_real
        denoised_real = denoiser(image[:, 0:1, ...], sigma_real)
        denoised_cplx = denoiser(image[:, 1:2, ...], sigma_cplx)
    denoised = torch.cat((denoised_real, denoised_cplx), axis=1)
    denoised = torch.rot90(denoised, k=-rot_idx, dims=[-2, -1])
    return denoised


def get_padding(im):
    # Pads an image to its closest 2**p size
    s_1 = int(2 ** (np.ceil(np.log10(im.shape[-2]) / np.log10(2))))
    s_2 = int(2 ** (np.ceil(np.log10(im.shape[-1]) / np.log10(2))))

    wpad_1 = (s_1 - im.shape[-2]) // 2
    wpad_2 = (s_1 - im.shape[-2] + 1) // 2
    hpad_1 = (s_2 - im.shape[-1]) // 2
    hpad_2 = (s_2 - im.shape[-1] + 1) // 2

    p = (hpad_1, hpad_2, wpad_1, wpad_2)

    im_pad = torch.nn.functional.pad(im, p, 'circular')

    return im_pad, p

def denoise_rotate_fn(denoiser, image, sigma, rot_idx=0, retain_grad=False):
    r'''
    Case of natural images

    :param denoiser:
    :param image:
    :param sigma:
    :param rand_rot:
    :param mean_rot:
    :return:
    '''
    image = torch.rot90(image, k=rot_idx, dims=[-2, -1])
    if not retain_grad:
        with torch.no_grad():
            if denoiser.__class__.__name__ == 'DiffUNet':
                image_pad, p = get_padding(image)
                denoised_pad = denoiser(image_pad, sigma)
                if p[0] == 0 and p[2] == 0:
                    denoised = denoised_pad
                else:
                    denoised = denoised_pad[..., p[2]:-p[3], p[0]:-p[1]]
            else:
                denoised = denoiser(image, sigma)
    else:
        if denoiser.__class__.__name__ == 'DiffUNet':
            image_pad, p = get_padding(image)
            denoised_pad = denoiser(image_pad, sigma)
            if p[0] == 0 and p[2] == 0:
                denoised = denoised_pad
            else:
                denoised = denoised_pad[..., p[2]:-p[3], p[0]:-p[1]]
        else:
            denoised = denoiser(image, sigma)
    denoised = torch.rot90(denoised, k=-rot_idx, dims=[-2, -1])
    return denoised


def denoise_rotate_translate_fn(denoiser, image, sigma, rot_idx=0, translate_shift=(0, 0)):
    r'''
    Case of natural images

    :param denoiser:
    :param image:
    :param sigma:
    :param rand_rot:
    :param mean_rot:
    :return:
    '''
    image = torch.roll(image, shifts=translate_shift, dims=(-2, -1))
    image = torch.rot90(image, k=rot_idx, dims=[-2, -1])
    with torch.no_grad():
        denoised = denoiser(image, sigma)
    denoised = torch.rot90(denoised, k=-rot_idx, dims=[-2, -1])
    denoised = torch.roll(denoised, shifts=(-translate_shift[0], -translate_shift[1]), dims=(-2, -1))
    return denoised


def denoise_rotate(denoiser, image, sigma, rand_rot=False, mean_rot=False, rand_translations=False, clamp=True, retain_grad=False):
    r'''
    Case of natural images

    :param denoiser:
    :param image:
    :param sigma:
    :param rand_rot:
    :param mean_rot:
    :return:
    '''
    if rand_rot:
        k = np.random.choice([0, 1, 2, 3])
        flip = 0
        # flip = np.random.choice([0, 1, 2])
        # x_shift = np.random.choice(list(range(-64, 64)))
        # y_shift = np.random.choice(list(range(-64, 64)))
        # x_shift = np.random.choice(list(range(-1, 1)))
        # y_shift = np.random.choice(list(range(-1, 1)))

        if flip == 1:
            image = torch.flip(image, dims=[-2])
        elif flip == 2:
            image = torch.flip(image, dims=[-1])
        denoised = denoise_rotate_fn(denoiser, image, sigma, rot_idx=k, retain_grad=retain_grad)
        if flip == 1:
            denoised = torch.flip(denoised, dims=[-2])
        elif flip == 2:
            denoised = torch.flip(denoised, dims=[-1])
    elif mean_rot:
        denoised = 0*image
        for k in range(4):
            denoised = denoised + denoise_rotate_fn(denoiser, image, sigma, rot_idx=k, retain_grad=retain_grad)
        denoised = denoised/4.
    elif rand_translations:
        k = np.random.choice([0, 1, 2, 3])
        flip = np.random.choice([0, 1, 2])

        if flip == 1:
            image = torch.flip(image, dims=[-2])
        elif flip == 2:
            image = torch.flip(image, dims=[-1])

        x_shift = np.random.choice(list(range(-64, 64)))
        y_shift = np.random.choice(list(range(-64, 64)))
        denoised = denoise_rotate_translate_fn(denoiser, image, sigma, rot_idx=k, translate_shift=(x_shift, y_shift))

        if flip == 2:
            denoised = torch.flip(denoised, dims=[-1])
        elif flip == 1:
            denoised = torch.flip(denoised, dims=[-2])

    else:  # normal denoising
        denoised = denoise_rotate_fn(denoiser, image, sigma, rot_idx=0, retain_grad=retain_grad)

    if clamp:
        denoised = torch.clamp(denoised, 0, 1)
    return denoised


def denoise_translate_fn(denoiser, image, sigma, trans_idx=0):
    axis = np.random.choice([-2, -1])
    image = torch.roll(image, trans_idx, axis)
    with torch.no_grad():
        denoised = denoiser(image, sigma)
    denoised = torch.roll(denoised, -trans_idx, axis)
    return denoised


def denoise_translate(denoiser, image, sigma, rand_rot=False, mean_rot=False):
    if rand_rot:
        # k = np.random.choice([-1, 0, 1])
        k = np.random.choice(list(range(-64, 64)))
        denoised = denoise_translate_fn(denoiser, image, sigma, trans_idx=k)
    elif mean_rot:
        denoised = 0*image
        for k in range(4):
            denoised = denoised + denoise_translate_fn(denoiser, image, sigma, trans_idx=k)
        denoised = denoised/4.
    else:  # normal denoising
        denoised = denoise_translate_fn(denoiser, image, sigma, trans_idx=0)
    return denoised


def denoise_translate_rotate(denoiser, image, sigma, rand_rot=False, mean_rot=False):
    if rand_rot:
        # k = np.random.choice([-1, 0, 1])
        k = np.random.choice(list(range(-64, 64)))
        denoised = denoise_translate_fn(denoiser, image, sigma, trans_idx=k)
    elif mean_rot:
        denoised = 0*image
        for k in range(4):
            denoised = denoised + denoise_translate_fn(denoiser, image, sigma, trans_idx=k)
        denoised = denoised/4.
    else:  # normal denoising
        denoised = denoise_translate_fn(denoiser, image, sigma, trans_idx=0)
    return denoised


class alpha_DnCNN(torch.nn.Module):
    def __init__(self, dncnn):
        super().__init__()
        self.denoiser = dncnn
        self.sigma_train = 0.008

    def forward(self, x, sigma):
        alpha = sigma/self.sigma_train
        denoised = self.denoiser(x/alpha, sigma)
        return torch.clamp(alpha*denoised, 0, 1)


def get_model(model_name, device='cpu', channels=1):
    col_str = 'color' if channels == 3 else 'gray'
    if model_name == 'dncnn':
        model = dinv.models.DnCNN(in_channels=channels, out_channels=channels, device=device, pretrained='/gpfsstore/rech/tbo/ubd63se/checkpoints/dncnn_sigma2_'+col_str+'.pth')
    elif model_name == 'alpha_dncnn':
        dncnn = dinv.models.DnCNN(in_channels=channels, out_channels=channels, device=device, pretrained='/gpfsstore/rech/tbo/ubd63se/checkpoints/dncnn_sigma2_'+col_str+'.pth')
        model = alpha_DnCNN(dncnn)
    elif model_name == 'lip_dncnn':
        model = dinv.models.DnCNN(in_channels=channels, out_channels=channels, device=device, pretrained='/gpfsstore/rech/tbo/ubd63se/checkpoints/dncnn_sigma2_lipschitz_'+col_str+'.pth')
    elif model_name == 'drunet':
        model = dinv.models.DRUNet(in_channels=channels, out_channels=channels, device=device, pretrained='/gpfsstore/rech/tbo/ubd63se/checkpoints/drunet_'+col_str+'.pth')
    elif model_name == 'gs_drunet':
        model = dinv.models.GSDRUNet(in_channels=channels, out_channels=channels, device=device, pretrained='/gpfsstore/rech/tbo/ubd63se/checkpoints/GSDRUNet.ckpt')
    elif model_name == 'drunet_deepinv':
        model = dinv.models.DRUNet(in_channels=channels, out_channels=channels, device=device, pretrained='/gpfsstore/rech/tbo/ubd63se/checkpoints/drunet_deepinv_'+col_str+'.pth')
    elif model_name == 'diffunet':
        model = dinv.models.DiffUNet(in_channels=channels, out_channels=channels, pretrained='/gpfsstore/rech/tbo/ubd63se/checkpoints/diffusion_ffhq_10m.pt')
    elif model_name == 'diffunet_large':
        model = dinv.models.DiffUNet(in_channels=channels, large_model=True, out_channels=channels, pretrained='/gpfsstore/rech/tbo/ubd63se/checkpoints/diffusion_openai.pt')
    elif model_name == 'scunet':
        model = dinv.models.SCUNet(in_nc=channels, device=device,
                                   pretrained='/gpfsstore/rech/tbo/ubd63se/checkpoints/scunet_' + col_str + '_real_psnr.pth')
    elif model_name == 'swinir':
        col_str = '005_color' if channels == 3 else '004_gray'
        model = dinv.models.SwinIR(in_chans=channels, device=device,
                                   pretrained='/gpfsstore/rech/tbo/ubd63se/checkpoints/'+col_str+'DN_DFWB_s128w8_SwinIR-M_noise15.pth')
    elif model_name == 'wavelets':
        model = dinv.models.WaveletDict(non_linearity="hard", level=1)
    else:
        raise NotImplementedError
    return model.to(device)


def get_physics(x_true, n_channels, problem_type='sr', device='cpu', sr=2, img_size=None, id_blur=1, noise_level=0.01):

    if problem_type == 'sr':
        if sr == 2:
            noise_level = 0.01
        else:
            noise_level = 0.05
        physics = dinv.physics.Downsampling((n_channels, x_true.shape[-2], x_true.shape[-1]),
                                            filter='gaussian',
                                            factor=sr,
                                            device=device,
                                            noise_model=dinv.physics.GaussianNoise(sigma=noise_level))
    elif problem_type == 'denoising':
        physics = dinv.physics.DecomposablePhysics()
        physics.noise_model = dinv.physics.GaussianNoise(sigma_max=0.2)
    elif problem_type == 'gaussian_blur':
        physics = dinv.physics.BlurFFT(
            img_size=(n_channels, x_true.shape[-2], x_true.shape[-1]),
            filter=dinv.physics.blur.gaussian_blur(),
            device=device,
            noise_model=dinv.physics.GaussianNoise(sigma=noise_level),
        )
    elif problem_type == 'motion_blur':
        blur_filter = get_blur_kernel(id=id_blur)
        physics = dinv.physics.BlurFFT(
            img_size=(n_channels, x_true.shape[-2], x_true.shape[-1]),
            filter=blur_filter,
            device=device,
            noise_model=dinv.physics.GaussianNoise(sigma=noise_level),
        )
    elif problem_type == 'CT':
        physics = dinv.physics.Tomography(
            img_width=x_true.shape[-1],
            angles=100,
            circle=False,
            device=device,
            noise_model=dinv.physics.GaussianNoise(sigma=noise_level),
        )
    else:
        raise NotImplementedError

    return physics
