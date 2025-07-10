# Apex Diffusion

Apex Diffusion is a project that aims to provide a unified interface for running diffusion models.

## Features

- Unified interface for running diffusion models
- Support for multiple models
- Support for multiple modes

## Setup

You will need to install the following dependencies before starting anything:

```bash
pip install torch torchaudio torchvision
```

### You will additionaly the following thirdparty libraries which will be kept in the `thirdparty` directory:
- `diffusers`
- `flash-attn`
- `sage-attention`

**If for some reason, the thirdparty folder is not present, you can run the following command to install it:**

```bash
./src/scripts/create_thirdparty.sh
cd thirdparty/diffusers
pip install -e .
```

or manually:

```bash
mkdir -p thirdparty
git clone https://github.com/huggingface/diffusers.git thirdparty/diffusers
cd thirdparty/diffusers
pip install -e .
```

After that you can install the dependencies for the project:

```bash
pip install -r requirements.txt
```

## Installing Optional Dependencies
** If your machine supports it, you can install sageattention and flash-attn to speed up the inference process. **

```bash
pip install thirdparty/flash-attn
pip install thirdparty/sage-attention
```

## Running the project
The current work has been completed on an H100 GPU. This may make it prohibitively difficult to run on other machines. 
Here is a step by step guide for how you can run each model with `torch.int8` precision.

You will need to use the CLI tool to easily run the models. The command structure is designed to be simple and intuitive, much like Docker.

```bash
apex run <base/model:tag|yaml-path> [args]
```

## Example

```bash
apex run wan/t2v:1.3b --prompt "A beautiful sunset over a calm ocean" --duration 16f --num-videos 1 --seed 42 --offload=True \
    --save_path=./output.mp4 \
    --transformer_dtype=float8_e4m3fn \
    --text_encoder_dtype=float8_e4m3fn \
    --vae_dtype=float16 \
    --offload=True
```

## Managing Models

### Searching for new models
To search for all available models in our registry, you can use the `search` command.

```bash
apex search [query]
```
If you omit the query, it will show a list of all public models. You can also search within a specific model base.

### Listing local models

To see all the models you have downloaded locally on your machine:

```bash
apex model list
```
or for a specific model base, you can run the following command:

```bash
apex model list <model-base-name>
```

The available model bases are:
- `wan`
- `hunyuan`
- `cogvideox`
- `ltx`
- `stepvideo`
- `mochi`

To list available model bases you can also run the following command:
```bash
apex base list
```

### Downloading a model

To download a specific model from the registry, use the `pull` command:

```bash
apex pull <base/model:tag>
```

For example:
```bash
apex pull wan/t2v:1.3b
```

## Available Models

To list all available models, you can run the following command:

```bash
apex models list
```
or for a specific model base, you can run the following command:

```bash
apex model list <model-base-name>
```

The available model bases are:
- `wan`
- `hunyuan`
- `cogvideox`
- `ltx`
- `stepvideo`
- `mochi`

To list available model bases you can also run the following command:
```bash
apex model-base list
```

To download a specific model, you can run the following command:

```bash
apex pull model <model-name>
```
