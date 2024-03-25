# Installing required libraries
#pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 torchdata==0.5.1 torchtext==0.14.1
#pip install diffusers==0.14
#pip install -q accelerate transformers xformers
#pip install -q opencv-contrib-python
#pip install -q controlnet_aux

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, DEISMultistepScheduler, EulerAncestralDiscreteScheduler
import torch
import cv2  # OpenCV
from PIL import Image
import numpy as np

def grid_img(imgs, rows=1, cols=3, scale=1):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    w, h = int(w * scale), int(h * scale)

    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        img = img.resize((w, h), Image.ANTIALIAS)
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

# Edge detection with Canny Edge
controlnet_canny_model = 'lllyasviel/sd-controlnet-canny'
control_net_canny = ControlNetModel.from_pretrained(controlnet_canny_model, torch_dtype=torch.float16)

pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5',
                                                         controlnet=control_net_canny,
                                                         torch_dtype=torch.float16)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

# Loading and processing the image
img = Image.open('/content/bird.jpg')

def canny_edge(img, low_threshold=100, high_threshold=200):
    img_array = np.array(img)
    edges = cv2.Canny(img_array, low_threshold, high_threshold)
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    canny_img = Image.fromarray(edges)
    return canny_img

canny_img = canny_edge(img)

prompt = "realistic photo of a blue bird with purple details, high quality, natural light"
neg_prompt = ""

seed = 777
generator = torch.Generator(device="cuda").manual_seed(seed)

imgs = pipe(
    prompt,
    canny_img,
    negative_prompt=neg_prompt,
    generator=generator,
    num_inference_steps=20,
)

grid_img([imgs.images[0]])

# Pose estimation using OpenPose
from controlnet_aux import OpenposeDetector
pose_model = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

img_pose = Image.open('/content/pose01.jpg')
pose = pose_model(img_pose)
grid_img([img_pose, pose])

# ControlNet model with OpenPose
controlnet_pose_model = ControlNetModel.from_pretrained('thibaud/controlnet-sd21-openpose-diffusers', torch_dtype=torch.float16)
sd_controlpose = StableDiffusionControlNetPipeline.from_pretrained('stabilityai/stable-diffusion-2-1-base',
                                                                   controlnet=controlnet_pose_model,
                                                                   torch_dtype=torch.float16)

sd_controlpose.enable_model_cpu_offload()
sd_controlpose.enable_attention_slicing()
sd_controlpose.enable_xformers_memory_efficient_attention()
sd_controlpose.scheduler = DEISMultistepScheduler.from_config(sd_controlpose.scheduler.config)

seed = 777
generator = torch.Generator(device="cuda").manual_seed(seed)

prompt = "professional photo of a young woman in the street, wearing a coat, sharp focus, insanely detailed, photorealistic, sunset, side light"
neg_prompt = "ugly, tiling, closed eyes, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face"

imgs = sd_controlpose(
    prompt,
    pose,
    negative_prompt=neg_prompt,
    num_images_per_prompt=4,
    generator=generator,
    num_inference_steps=20,
)

grid_img(imgs.images)

# Further testing and improvements
prompt_list = [
    "oil painting walter white wearing a suit and black hat and sunglasses, face portrait, in the desert, realistic, vivid",
    "oil painting walter white wearing a jedi brown coat, face portrait, wearing a hood, holding a cup of coffee, in another planet, realistic, vivid",
    "professional photo of walter white wearing a space suit, face portrait, in mars, realistic, vivid",
    "professional photo of walter white in the kitchen, face portrait, realistic, vivid"
]
neg_prompt_list = [neg_prompt] * len(prompt_list)

imgs = sd_controlpose(
    prompt_list,
    pose,
    negative_prompt=neg_prompt_list,
    generator=generator,
    num_inference_steps=20,
)

grid_img(imgs.images)

# Applying different schedulers
sd_controlpose.scheduler = EulerAncestralDiscreteScheduler.from_config(sd_controlpose.scheduler.config)

imgs = sd_controlpose(
    prompt_list,
    pose,
    negative_prompt=neg_prompt_list,
    generator=generator,
    num_inference_steps=20,
)

grid_img(imgs.images)

# Additional image loading and pose extraction
urls = ["yoga1.jpeg", "yoga2.jpeg", "yoga3.jpeg", "yoga4.jpeg"]
imgs = [Image.open(url) for url in urls]
poses = [pose_model(img) for img in imgs]

grid_img(imgs + poses)

# Final image generation with DEIS scheduler
prompt_list = [
    "oil painting walter white wearing a suit and black hat and sunglasses, face portrait, in the desert, realistic, vivid",
    "oil painting walter white wearing a jedi brown coat, face portrait, wearing a hood, holding a cup of coffee, in another planet, realistic, vivid",
    "professional photo of walter white wearing a space suit, face portrait, in mars, realistic, vivid",
    "professional photo of walter white in the kitchen, face portrait, realistic, vivid"
]

imgs = sd_controlpose(
    prompt_list,
    poses,
    negative_prompt=neg_prompt_list,
    generator=generator,
    num_inference_steps=30,
)

grid_img(imgs.images)
