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
To test that the code effectively, you can run the following command:
```bash
python3 -m src.engine.wan_engine
```

This will attempt to run the `wan_vace_1.3b` model.