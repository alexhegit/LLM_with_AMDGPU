# LLM_with_AMDGPU

Sharing what I have done about running LLM with AMD GPU. 
The AMD dGPU(Radeon GPU) and iGPU will be used first to show how to run LLM which you may easy to own. YES, I means they are more cheaper than NVIDIA GPU.

## Radeon GPU + ROCm
From PyTorch2.0, ROCm is out-of-box with PyTorch and Huggingface to use AMD dGPU(MIxxx and Radeon GPU).

Platform:
- AMD Ryzen 7900 + AMD Radeon Pro W7900
- Ubuntu22.04
- Key libs
    - python 3.11.5
    - torch-rocm	v2.2.0.dev20231211+rocm5.7
    - transformers 	v4.36.2 or newer
    - accelerate	v0.25.0

Software Envrionment Setup:
1. Install the ROCm packages at first(drivers, basic libs, tools). Please refert to https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/install-overview.html.

2. Install the PyTorch-ROCm version.
The ROCm is out-of-box with PyTorch2.x. Please go to https://pytorch.org/ for details.

e.g.

```pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm5.7```

3. Install other libs like transformers, accelerate and anything need by the models. You should refer to every model card or repo project to finish it.

4. Download and run inference the model refer to Hugggingface guide:w
for each one.

I upload the inference python code in this repo for each model. Bellow is the checklist for quick to know what models are work with ROCm.

LLM+ROCm inference checklist(verified with Huggingface)
- chatglm3-6b
- baichuan2-13b
- microsoft/phi-2 (support flash-attn)
- Mistral-7B-v1.0
- openai/whispe
- llama2-7B-hf

## AMD iGPU + vulkan
*to-do*

## AMD GPU + MLC-LLM
*to-do*



