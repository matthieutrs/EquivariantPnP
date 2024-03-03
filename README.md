# Equivariant plug-and-play image reconstruction

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2312.01831)

[camera-ready (soon!)]()

[Matthieu Terris](https://matthieutrs.github.io/), [Thomas Moreau](https://tommoral.github.io/about.html), [Nelly Pustelnik](http://perso.ens-lyon.fr/nelly.pustelnik/), [Julián Tachella](https://tachella.github.io/).

To appear in CVPR 2024

## tl;dr

Enforcing equivariance of the denoiser to certain transformations within PnP/RED algorithms improves the stability and reconstruction quality of the algorithm. 

![flexible](https://github.com/matthieutrs/equivariant_pnp/blob/main/images/summary_eq.png)

[//]: # (## Abstract)

[//]: # ()
[//]: # (Plug-and-play algorithms constitute a popular framework for solving inverse imaging problems that rely on the implicit definition of an image prior via a denoiser. These algorithms can leverage powerful pre-trained denoisers to solve a wide range of imaging tasks, circumventing the necessity to train models on a per-task basis. Unfortunately, plug-and-play methods often show unstable behaviors, hampering their promise of versatility and leading to suboptimal quality of reconstructed images. In this work, we show that enforcing equivariance to certain groups of transformations &#40;rotations, reflections, and/or translations&#41; on the denoiser strongly improves the stability of the algorithm as well as its reconstruction quality. We provide a theoretical analysis that illustrates the role of equivariance on better performance and stability. We present a simple algorithm that enforces equivariance on any existing denoiser by simply applying a random transformation to the input of the denoiser and the inverse transformation to the output at each iteration of the algorithm. Experiments on multiple imaging modalities and denoising networks show that the equivariant plug-and-play algorithm improves both the reconstruction performance and the stability compared to their non-equivariant counterparts.)

## Method description

We consider algorithms where gradients (or proximal operators) of explicit priors are replaced by denoisers; these algorithms typically take the form (in the case of PnP)

``` math
x_{k+1} = \text{D}(x_k - \gamma \nabla f(x_k)),
```

where $\text{D}$ is a denoiser. In our paper, we show that enforcing equivariance of the denoiser with respect to a group of geometric transformations (such as rotations) can improve the Lipschitz constant of the denoiser, and hence the stability of the algorithm.
While a trivial way to enforce equivariance with respect to a group of transforms $\mathcal{G}$ is to perform an averaging of the denoiser's output over the group of transformations as $\text{D}\_{\mathcal{G}} = \frac{1}{|G|} \sum\_{g \in \mathcal{G}} T\_g^{-1} \text{D}(T\_g)$, we propose to use a Monte-Carlo estimation of the equivariant denoiser at each step of the algorithm. The resulting algorithm reads

``` math
\begin{align*}
&\text{Sample } g_k \sim \mathcal{G} \\
&\text{Set } \text{D}_{g_k}(x) = T_{g_k}^{-1} \text{D}(T_{g_k} x) \\
&x_{k+1} = \text{D}_{g_k}(x_k - \gamma \nabla f(x_k)).
\end{align*}
```


## Code
To reproduce the experiments, first download the test datasets and place them in your data folder. Next, update the `config/config.json` file to point to the correct data folder. There, there are two folders to specify:
- `ROOT_DATASET`: the folder within which the [CBSD68](https://huggingface.co/datasets/deepinv/CBSD68) and [set3c](https://huggingface.co/datasets/deepinv/set3c) datasets are located;
- `PATH_MRI_DATA`: the path to the fastMRI .pt dataset.

Then, you can run the following scripts to reproduce the experiments:

<details>
<summary><strong>PnP</strong> (click to expand) </summary>

#### without equivariance

On the set3c dataset for the motion blur problem, with the drunet model:

```bash

python running_pnp.py --problem='motion_blur' --model_name='drunet' --rand_rotations=0 --dataset_name='set3c' --results_folder='table_4/' --compute_lip=0 --sigma_den=0.02 --noise_level=0.01

```

#### with equivariance

On the set3c dataset for the motion blur problem, with the drunet model, and with equivariance:

```bash

python running_pnp.py --problem='motion_blur' --model_name='drunet' --rand_rotations=1 --dataset_name='set3c' --results_folder='table_4/' --compute_lip=0 --sigma_den=0.02 --noise_level=0.01

```

</details>

<details>
<summary><strong>RED</strong> (click to expand) </summary>

#### without equivariance
On the set3c dataset for the super-resolution blur problem, with the drunet model (Fig. 6 of the paper):
```bash
python running_red.py --problem='sr' --model_name='drunet' --rand_translations=0 --dataset_name='set3c' --sigma_den=0.015 --sr=2
```
#### with equivariance
On the set3c dataset for the motion blur problem, with the drunet model, and with equivariance (Fig. 6 of the paper):
```bash
python running_red.py --problem='sr' --model_name='drunet' --rand_translations=1 --dataset_name='set3c' --sigma_den=0.015 --sr=2
```
Feel free to change problem and models!
</details>

<details>
<summary><strong>ULA</strong> (click to expand) </summary>

#### without equivariance
On the BSD68 dataset for the super-resolution blur problem, with the drunet model (Fig. 8 of the paper):
```bash
python running_ula.py --problem='motion_blur' --model_name='drunet' --rand_translations=0 --dataset_name='subset_BSD20' --sigma_den=0.019
```
#### with equivariance
On the BSD10 dataset for the motion blur problem, with the drunet model, and with equivariance (Fig. 8 of the paper):
```bash
python running_ula.py --problem='motion_blur' --model_name='drunet' --rand_translations=1 --dataset_name='subset_BSD20' --sigma_den=0.019
```
Feel free to change problem and models!
</details>

## Requirements

This code was tested with the following packages:
- torch 2.2
- deepinverse 0.1.1

The deepinverse package can be installed with `pip install deepinverse` or by cloning the [repository](https://github.com/deepinv/deepinv).
