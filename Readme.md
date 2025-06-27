# Pytorch Image Tokenizer
* A fork of https://github.com/Stability-AI/generative-models, with focus on tokenizers
    * Practical size training example for popular tokenizers, such as SD-VAE, VQ, FSQ, LFQ, BSQ
    * Both stable diffusion unet and bsq vit backbone support
    * [TODO] With pre-trained model and benchmark on ImageNet 256x256 
# Usage 
## Prequisites
## Installation
* from source
    ```bash
    pip install .
    ```

## Prepare your dataset
* It is recommend to list the dataset in advanced using
    ```bash
    python scripts/create_dataset_list.py --root $DATASET_FOLDER --ext $IMAGE_EXTENSION --out $OUTFILE
    ```

## Training Tokenizers using default config
* __Gaussian VAE__ with stable diffusion UNet
    ```bash
    python main.py --config sd3unet_gaussian_kl_0.64.yaml --wandb
    ```

* __FSQ__ with stable diffusion UNet
    ```bash
    python main.py --config sd3unet_fsq_888555.yaml --wandb
    ```

## Benchmark
* All models are trained with ImageNet train set, on 8xA100 GPU