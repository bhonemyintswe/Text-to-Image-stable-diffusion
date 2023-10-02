from transformers import pipeline, set_seed
import warnings
warnings.filterwarnings("ignore")
device = "cuda"

import torch
from diffusers import StableDiffusionPipeline
from diffusers import DiffusionPipeline
import matplotlib.pyplot as plt

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16") 
pipe = pipe.to(device)

def plot_image(prompet):
    image1 = pipe(prompt=prompet).images[0]
    plt.subplot(1, 1, 1)
    plt.imshow(image1)
    plt.title('stable-diffusion-xl-base-1.0')
    plt.axis('off')

plot_image('Samurai warrior, ultra realistic, Black and Red, Japanese flag in background, portrait')
