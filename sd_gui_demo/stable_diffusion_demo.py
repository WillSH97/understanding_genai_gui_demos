from PIL import Image
from tqdm.auto import tqdm
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, LMSDiscreteScheduler, StableDiffusionPipeline, UniPCMultistepScheduler
from torchvision import transforms

torch_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu")

torch_dtype = torch.float16 if torch_device in ["cuda", "mps"] else torch.float32

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", 
    torch_dtype=torch_dtype, 
    use_safetensors=True, 
    safety_checker = None).to(torch_device)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

pipe.enable_model_cpu_offload()

def StableDiffusion(uncond_embeddings, text_embeddings, height, width, num_inference_steps, guidance_scale, seed):
    batch_size=1

    generator = None

    if seed:
        generator=torch.manual_seed(seed)
    
    output = pipe(
        prompt = text_embeddings,
        negative_prompt = uncond_embeddings,
        height = height,
        width = width,
        num_inference_steps = num_inference_steps,
        guidance_scale = guidance_scale,
        generator = generator
    ).images[0]
    return output

