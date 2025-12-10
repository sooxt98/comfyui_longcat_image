# ComfyUI LongCat-Image Integration

This custom node integrates the [LongCat-Image](https://github.com/meituan-longcat/LongCat-Image) pipeline into ComfyUI, enabling text-to-image generation and image editing with the LongCat-Image models.

## Features

- **Text-to-Image Generation**: Generate high-quality images from text prompts using LongCat-Image models
- **Image Editing**: Edit existing images with instruction-based prompts using LongCat-Image-Edit models
- **Chinese Text Support**: Excellent Chinese text rendering capabilities
- **Efficient**: Only 6B parameters with competitive performance

## Installation

### 1. Install Dependencies

```bash
cd custom_nodes/comfyui_longcat_image
pip install -r requirements.txt
```

### 2. Install LongCat-Image Package

```bash
pip install git+https://github.com/meituan-longcat/LongCat-Image.git
```

### 3. (Optional) Install SageAttention for Speed Boost

For ~2x faster inference, install SageAttention:

```bash
pip install sageattention
```

**Requirements:** CUDA-capable NVIDIA GPU with PyTorch CUDA support.

### 4. Download Models

Download the models using huggingface-cli:

```bash
pip install "huggingface_hub[cli]"

# For text-to-image
huggingface-cli download meituan-longcat/LongCat-Image --local-dir models/diffusion_models/LongCat-Image

# For image editing
huggingface-cli download meituan-longcat/LongCat-Image-Edit --local-dir models/diffusion_models/LongCat-Image-Edit

# For fine-tuning (optional)
huggingface-cli download meituan-longcat/LongCat-Image-Dev --local-dir models/diffusion_models/LongCat-Image-Dev
```

## Available Nodes

### LongCat-Image Model Loader

Loads a LongCat-Image model for use with other nodes.

**Inputs:**
- `model_path`: Path to the model directory (e.g., "LongCat-Image" or "LongCat-Image-Edit")
- `dtype`: Data type for model weights (bfloat16, float16, float32)
- `enable_cpu_offload`: Enable CPU offload to save VRAM (false/true, default: true)
- `attention_backend`: Choose attention backend - "default" or "sage" (default: default)

**Outputs:**
- `LONGCAT_PIPE`: Pipeline object for use with generation nodes

#### Low VRAM Support

The model loader supports low VRAM mode via the `enable_cpu_offload` option:

- **Disabled**: All models loaded to GPU at once
  - Faster inference
  - Requires more VRAM (typically ~24GB+)
  
- **Enabled (default)**: Models offloaded to CPU when not in use
  - Slower inference (due to model transfers)
  - Requires only ~17-19GB VRAM
  - Prevents Out-of-Memory errors on lower-end GPUs

**When to use CPU offload:**
- GPUs with less than 24GB VRAM
- When experiencing OOM errors
- When running multiple models simultaneously

#### SageAttention Backend

The model loader supports an optional SageAttention backend for improved inference speed:

- **default**: Uses PyTorch's standard scaled dot product attention
  - Works on all systems (CPU/GPU)
  - Standard performance
  
- **sage**: Uses [SageAttention](https://github.com/thu-ml/SageAttention) for accelerated attention computation
  - **~2x faster inference speed** compared to default attention
  - Requires CUDA-capable GPU
  - Requires the `sageattention` package (see installation section above)
  - Automatically falls back to default attention for unsupported operations

**To use SageAttention:**

1. Install the sageattention package:
```bash
pip install sageattention
```

2. Set `attention_backend` to "sage" in the Model Loader node

**Requirements:**
- CUDA-capable NVIDIA GPU
- PyTorch with CUDA support
- The `sageattention` package installed

### LongCat-Image Text to Image

Generates images from text prompts.

**Inputs:**
- `LONGCAT_PIPE`: Pipeline from the model loader
- `prompt`: Text description of the image to generate
- `negative_prompt`: Things to avoid in the generated image
- `width`: Image width (default: 1344)
- `height`: Image height (default: 768)
- `steps`: Number of inference steps (default: 50)
- `guidance_scale`: CFG scale (default: 4.5)
- `seed`: Random seed
- `enable_cfg_renorm`: Enable CFG renormalization (true/false)
- `enable_prompt_rewrite`: Enable built-in prompt rewriting (true/false)

**Outputs:**
- `IMAGE`: Generated image

### LongCat-Image Edit

Edits images based on instruction prompts.

**Inputs:**
- `LONGCAT_PIPE`: Pipeline from the model loader (must be an edit model)
- `image`: Input image to edit
- `prompt`: Edit instruction
- `negative_prompt`: Things to avoid in the edited image
- `steps`: Number of inference steps (default: 50)
- `guidance_scale`: CFG scale (default: 4.5)
- `seed`: Random seed

**Outputs:**
- `IMAGE`: Edited image

## Example Workflows

Example workflow JSON files are provided in this directory:
- `example_workflow_t2i.json` - Text-to-image generation workflow
- `example_workflow_edit.json` - Image editing workflow

You can load these workflows in ComfyUI by dragging and dropping the JSON file onto the canvas.

### Text-to-Image

1. Add a **LongCat-Image Model Loader** node
   - Set `model_path` to "LongCat-Image"
   
2. Add a **LongCat-Image Text to Image** node
   - Connect the loader output to the pipeline input
   - Enter your prompt
   - Adjust settings as needed

3. Add a **Save Image** node to save the output

### Image Editing

1. Add a **LongCat-Image Model Loader** node
   - Set `model_path` to "LongCat-Image-Edit"
   
2. Add a **Load Image** node to load your input image

3. Add a **LongCat-Image Edit** node
   - Connect the loader output to the pipeline input
   - Connect the image to edit
   - Enter your edit instruction (e.g., "将猫变成狗" - "change the cat to a dog")

4. Add a **Save Image** node to save the output

## Model Information

| Model | Type | Description |
|-------|------|-------------|
| LongCat-Image | Text-to-Image | Final release model for out-of-the-box inference |
| LongCat-Image-Dev | Text-to-Image | Mid-training checkpoint, suitable for fine-tuning |
| LongCat-Image-Edit | Image Editing | Specialized model for image editing |

## Performance

- **Parameters**: 6B (highly efficient)
- **Supported Resolutions**: 768x1344 and variations
- **Chinese Text Support**: Industry-leading Chinese dictionary coverage
- **Quality**: Competitive with much larger models

### Attention Backend Performance

| Backend | Speed | Requirements | When to Use |
|---------|-------|--------------|-------------|
| default | 1x (baseline) | Any system | General use, CPU inference |
| sage | ~2x faster | CUDA GPU + sageattention package | Maximum speed on NVIDIA GPUs |

**Note**: SageAttention provides approximately 2x speed improvement for attention operations on CUDA GPUs while maintaining output quality.

### VRAM Requirements

| Mode | VRAM Required | Speed | When to Use |
|------|---------------|-------|-------------|
| Standard (CPU offload disabled) | ~24GB+ | Faster | High-end GPUs (e.g., RTX 3090, 4090, A100) |
| Low VRAM (CPU offload enabled) | ~17-19GB | Slower | Mid-range GPUs (e.g., RTX 3080, 4080) |

**Note**: The Low VRAM mode uses CPU offloading to transfer models between CPU and GPU as needed, reducing VRAM usage at the cost of slower inference speed.

## Tips

- For better results, use a strong LLM for prompt engineering
- The model has excellent Chinese text rendering capabilities
- Enable prompt rewriting for enhanced generation quality
- Default guidance scale of 4.5 works well for most cases

## License

LongCat-Image is licensed under Apache 2.0. See the [LongCat-Image repository](https://github.com/meituan-longcat/LongCat-Image) for more information.

## References

- [LongCat-Image GitHub](https://github.com/meituan-longcat/LongCat-Image)
- [LongCat-Image on Hugging Face](https://huggingface.co/meituan-longcat/LongCat-Image)
- [Technical Report](https://github.com/meituan-longcat/LongCat-Image/blob/main/assets/LongCat_Image_Technical_Report.pdf)
