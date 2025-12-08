"""
ComfyUI custom nodes for LongCat-Image
https://github.com/meituan-longcat/LongCat-Image
"""

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from comfy_api.latest._io import comfytype, ComfyTypeIO
import torch
import folder_paths
import os


# Define custom type for LongCat pipeline
@comfytype(io_type="LONGCAT_PIPE")
class LongCatPipe(ComfyTypeIO):
    """Custom type for LongCat-Image pipeline"""
    Type = dict


class LongCatImageModelLoader(io.ComfyNode):
    """
    Load LongCat-Image models for text-to-image or image editing.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LongCatImageModelLoader",
            display_name="LongCat-Image Model Loader",
            category="loaders",
            inputs=[
                io.String.Input(
                    "model_path",
                    default="",
                    multiline=False,
                    tooltip="Path to LongCat-Image model directory (e.g., 'LongCat-Image', 'LongCat-Image-Edit')"
                ),
                io.Combo.Input(
                    "dtype",
                    options=["bfloat16", "float16", "float32"],
                    default="bfloat16",
                    tooltip="Data type for model weights"
                ),
                io.Combo.Input(
                    "enable_cpu_offload",
                    options=["false", "true"],
                    default="true",
                    tooltip="Enable CPU offload to save VRAM (~17-19GB required). Slower but prevents OOM on low VRAM GPUs."
                ),
            ],
            outputs=[
                LongCatPipe.Output(
                    display_name="LongCat Pipeline",
                    tooltip="LongCat-Image pipeline for generation"
                ),
            ],
        )

    @classmethod
    def execute(cls, model_path, dtype, enable_cpu_offload) -> io.NodeOutput:
        try:
            from transformers import AutoProcessor
            from longcat_image.models import LongCatImageTransformer2DModel
            from longcat_image.pipelines import LongCatImagePipeline, LongCatImageEditPipeline
        except ImportError as e:
            raise ImportError(
                f"Failed to import LongCat-Image dependencies. "
                f"Please install: pip install -r custom_nodes/comfyui_longcat_image/requirements.txt\n"
                f"Error: {e}"
            )

        # Find model path
        if not model_path:
            raise ValueError("model_path is required")
        
        # Check if it's an absolute path
        if os.path.isabs(model_path):
            checkpoint_dir = model_path
        else:
            # Try to find in standard ComfyUI model directories
            base_paths = [
                os.path.join(folder_paths.models_dir, "diffusion_models"),
                os.path.join(folder_paths.models_dir, "checkpoints"),
                folder_paths.models_dir,
            ]
            
            checkpoint_dir = None
            for base_path in base_paths:
                potential_path = os.path.join(base_path, model_path)
                if os.path.exists(potential_path):
                    checkpoint_dir = potential_path
                    break
            
            if checkpoint_dir is None:
                raise ValueError(
                    f"Model not found at '{model_path}'. "
                    f"Please provide a valid path or place the model in: {base_paths[0]}"
                )

        # Map dtype string to torch dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map[dtype]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load text processor
        text_processor = AutoProcessor.from_pretrained(
            checkpoint_dir,
            subfolder='tokenizer'
        )

        # Determine if this is an edit model by checking the path
        is_edit_model = "edit" in model_path.lower()
        
        # Determine CPU offload setting
        cpu_offload = enable_cpu_offload == "true"

        # Load transformer
        transformer = LongCatImageTransformer2DModel.from_pretrained(
            checkpoint_dir,
            subfolder='transformer',
            torch_dtype=torch_dtype,
            use_safetensors=True
        )
        
        # Only move transformer to device if CPU offload is disabled
        if not cpu_offload:
            transformer = transformer.to(device)

        # Load appropriate pipeline
        if is_edit_model:
            pipe = LongCatImageEditPipeline.from_pretrained(
                checkpoint_dir,
                transformer=transformer,
                text_processor=text_processor,
                torch_dtype=torch_dtype,
            )
        else:
            pipe = LongCatImagePipeline.from_pretrained(
                checkpoint_dir,
                transformer=transformer,
                text_processor=text_processor,
                torch_dtype=torch_dtype,
            )
        
        # Apply CPU offload or move to device based on user preference
        if cpu_offload:
            # Enable CPU offload to save VRAM (Requires ~17-19 GB); slower but prevents OOM
            pipe.enable_model_cpu_offload()

        else:
            # Load all models to GPU at once (Faster inference but requires more VRAM)
            pipe.to(device, torch_dtype)

        pipeline_data = {
            "pipe": pipe,
            "device": device,
            "dtype": torch_dtype,
            "is_edit": is_edit_model,
        }

        return io.NodeOutput(pipeline_data)


class LongCatImageTextToImage(io.ComfyNode):
    """
    Generate images using LongCat-Image text-to-image pipeline.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LongCatImageTextToImage",
            display_name="LongCat-Image Text to Image",
            category="sampling",
            inputs=[
                LongCatPipe.Input(
                    "longcat_pipeline",
                    display_name="LongCat Pipeline",
                    tooltip="LongCat-Image pipeline from loader"
                ),
                io.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    tooltip="Text prompt for image generation"
                ),
                io.String.Input(
                    "negative_prompt",
                    default="",
                    multiline=True,
                    tooltip="Negative prompt"
                ),
                io.Int.Input(
                    "width",
                    default=1344,
                    min=64,
                    max=4096,
                    step=64,
                    tooltip="Image width"
                ),
                io.Int.Input(
                    "height",
                    default=768,
                    min=64,
                    max=4096,
                    step=64,
                    tooltip="Image height"
                ),
                io.Int.Input(
                    "steps",
                    default=50,
                    min=1,
                    max=200,
                    step=1,
                    display_mode=io.NumberDisplay.number,
                    tooltip="Number of inference steps"
                ),
                io.Float.Input(
                    "guidance_scale",
                    default=4.5,
                    min=0.0,
                    max=20.0,
                    step=0.1,
                    tooltip="Guidance scale for classifier-free guidance"
                ),
                io.Int.Input(
                    "seed",
                    default=43,
                    min=0,
                    max=0xffffffffffffffff,
                    tooltip="Random seed for generation"
                ),
                io.Combo.Input(
                    "enable_cfg_renorm",
                    options=["true", "false"],
                    default="true",
                    tooltip="Enable CFG renormalization"
                ),
                io.Combo.Input(
                    "enable_prompt_rewrite",
                    options=["true", "false"],
                    default="true",
                    tooltip="Enable built-in prompt rewriting using text encoder"
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip="Generated image"
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        longcat_pipeline,
        prompt,
        negative_prompt,
        width,
        height,
        steps,
        guidance_scale,
        seed,
        enable_cfg_renorm,
        enable_prompt_rewrite,
    ) -> io.NodeOutput:
        if not longcat_pipeline:
            raise ValueError("longcat_pipeline input is required")

        pipeline = longcat_pipeline["pipe"]
        
        if longcat_pipeline.get("is_edit", False):
            raise ValueError("This is an edit pipeline. Use LongCatImageEdit node instead.")

        # Convert string bools to actual bools
        cfg_renorm = enable_cfg_renorm == "true"
        prompt_rewrite = enable_prompt_rewrite == "true"

        # Generate image
        generator = torch.Generator("cpu").manual_seed(seed)
        
        result = pipeline(
            prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            num_images_per_prompt=1,
            generator=generator,
            enable_cfg_renorm=cfg_renorm,
            enable_prompt_rewrite=prompt_rewrite,
        )
        
        image = result.images[0]
        
        # Convert PIL image to tensor format expected by ComfyUI
        import numpy as np
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]
        
        return io.NodeOutput(image_tensor)


class LongCatImageEdit(io.ComfyNode):
    """
    Edit images using LongCat-Image-Edit pipeline.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LongCatImageEdit",
            display_name="LongCat-Image Edit",
            category="sampling",
            inputs=[
                LongCatPipe.Input(
                    "longcat_pipeline",
                    display_name="LongCat Pipeline",
                    tooltip="LongCat-Image-Edit pipeline from loader"
                ),
                io.Image.Input(
                    "image",
                    tooltip="Input image to edit"
                ),
                io.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    tooltip="Edit instruction prompt"
                ),
                io.String.Input(
                    "negative_prompt",
                    default="",
                    multiline=True,
                    tooltip="Negative prompt"
                ),
                io.Int.Input(
                    "steps",
                    default=50,
                    min=1,
                    max=200,
                    step=1,
                    display_mode=io.NumberDisplay.number,
                    tooltip="Number of inference steps"
                ),
                io.Float.Input(
                    "guidance_scale",
                    default=4.5,
                    min=0.0,
                    max=20.0,
                    step=0.1,
                    tooltip="Guidance scale for classifier-free guidance"
                ),
                io.Int.Input(
                    "seed",
                    default=43,
                    min=0,
                    max=0xffffffffffffffff,
                    tooltip="Random seed for generation"
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip="Edited image"
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        longcat_pipeline,
        image,
        prompt,
        negative_prompt,
        steps,
        guidance_scale,
        seed,
    ) -> io.NodeOutput:
        if not longcat_pipeline:
            raise ValueError("longcat_pipeline input is required")

        pipeline = longcat_pipeline["pipe"]
        
        if not longcat_pipeline.get("is_edit", False):
            raise ValueError("This is not an edit pipeline. Use LongCatImageTextToImage node instead.")

        # Convert tensor to PIL Image
        from PIL import Image
        import numpy as np
        
        # ComfyUI images are in format [B, H, W, C] in range [0, 1]
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np).convert('RGB')

        # Generate edited image
        generator = torch.Generator("cpu").manual_seed(seed)
        
        result = pipeline(
            pil_image,
            prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            num_images_per_prompt=1,
            generator=generator,
        )
        
        edited_image = result.images[0]
        
        # Convert PIL image back to tensor
        image_np = np.array(edited_image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]
        
        return io.NodeOutput(image_tensor)


class LongCatImageExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LongCatImageModelLoader,
            LongCatImageTextToImage,
            LongCatImageEdit,
        ]


async def comfy_entrypoint() -> LongCatImageExtension:
    """ComfyUI calls this to load the extension and its nodes."""
    return LongCatImageExtension()
