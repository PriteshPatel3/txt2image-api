from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
import yaml
import io
import base64
from PIL import Image


with open("config.yaml", 'r') as file:
    config = yaml.safe_load(file)

MODEL_ID = config['model']['id']
prompt = "a photo of an astronaut riding a horse on mars"

def pipeline(prompt, cuda='cuda'):
    # Use the Euler scheduler here instead
    scheduler = EulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, scheduler=scheduler, torch_dtype=torch.float16)
    pipe = pipe.to(cuda)

    image = pipe(prompt).images[0]
    
    # Convert the image to a byte array
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Return the base64 encoded image
    return f"data:image/png;base64,{img_str}"
    return image
    # image.save("astronaut_rides_horse.png")
    # return image
