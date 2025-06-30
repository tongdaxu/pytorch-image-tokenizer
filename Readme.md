# Pytorch Image Tokenizer
* A fork of https://github.com/Stability-AI/generative-models, with focus on tokenizers
    * Practical size implementation and training code for popular tokenizers, such as VQ, FSQ, LFQ, BSQ
    * Both stable diffusion unet and bsq vit backbone support
    * With pre-trained model and benchmark on ImageNet 256x256 

# Usage 
## Prequisites
* dependency in environment.yaml
    ```bash
    conda env create --file=environment.yaml
    conda activate tokenizer
    ```
## Installation
* from source
    ```bash
    pip install .
    ```

## Prepare your dataset
* It is recommend to list the dataset in advanced using
    ```bash
    python scripts/create_dataset_list.py --root $PATH_TO_DATASET_FOLDER --ext $IMAGE_EXTENSION --out $PATH_TO_OUTFILE
    ```
* It is not mandatory, just speed up training

## Training Tokenizers using default config
* modify the yaml file according to your system, pay special attention to "trainer-device", "trainer-num_nodes", "data-train-params-root"
* __Gaussian VAE__ with stable diffusion UNet
    ```bash
    python main.py --config sd3unet_gaussian_kl_0.64.yaml --wandb
    ```

* __FSQ__ with stable diffusion UNet
    ```bash
    python main.py --config sd3unet_fsq_16.yaml --wandb
    ```

* __LFQ__ with stable diffusion UNet
    ```bash
    python main.py --config sd3unet_lfq_16.yaml --wandb
    ```
* check ./configs/ for more

## Evaluating Tokenizers
* usage
    ```bash
    python -m torch.distributed.launch --standalone --use-env \
    --nproc-per-node=8 eval.py \
    --bs=32 \
    --base=$PATH_TO_YAML_CONFIG \
    --ckpt=$PATH_TO_CKPT \
    --dataset=$PATH_TO_DATASET_FOLDER
    ```

# Pre-trained models and benchmark
* All models are trained with ImageNet train set, on 8xA100 GPU for around 30 epochs, which takes around 24 hours 
* All models available in https://huggingface.co/xutongda/pytorch-image-tokenizer-models

| spec          | config                  | model                    | PSNR  | SSIM  | LPIPS | rFID  |
|---------------|-------------------------|--------------------------|-------|-------|-------|-------|
| VQ 2^16x1024  | sd3unet_vq_16.yaml     | sd3unet_vq_16.ckpt       |23.90| 0.683 | 0.112 | 1.961 |
| LFQ 2^16x1024 | sd3unet_lfq_16.yaml     | sd3unet_lfq_16.ckpt      | 22.65 | 0.635 | 0.141 | 3.523 |
| FSQ 2^16x1024 | sd3unet_fsq_16.yaml | sd3unet_fsq_16.ckpt  | 26.87 | 0.785 | 0.072 | 1.161 |
| BSQ 2^16x1024 | sd3unet_bsq_16.yaml     | sd3unet_bsq_16.ckpt      | 25.62 | 0.754 | 0.086 | 1.080 |

# Reference
* main structure is a fork from: https://github.com/Stability-AI/generative-models
* bsq, vit and evaluation from: https://github.com/zhaoyue-zephyrus/bsq-vit
* lfq from: https://github.com/TencentARC/SEED-Voken
* vq from: https://github.com/ai-forever/MoVQGAN
* [VQ NIPS 17] Neural Discrete Representation Learning
* [LFQ ICLR 24] Language Model Beats Diffusion: Tokenizer is key to visual generation
* [FSQ ICLR 24] Finite Scalar Quantization: VQ-VAE Made Simple
* [BSQ ICLR 25] Image and Video Tokenization with Binary Spherical Quantization
 