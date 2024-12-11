import os
import torch
import fsspec
import numpy as np
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, AutoencoderKL
import einops
import math
from models import ACDiT_models
import argparse
from torchvision.utils import save_image
import random

def samples_case(labels, model, vae, cfg, sample_step, args):
    scheduler = DDIMScheduler()
    sample_scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config)
    output = model.sample(labels, sample_scheduler, sample_step, cfg=cfg, target_shape=(labels.size(0), args.ar_len, args.block_size, args.vae_latent_size*args.patch_size**2), dtype=torch.float)
    block_h = block_w = int(math.isqrt(args.block_size))    
    new_h = new_w = int(math.isqrt(output.size(1)))
    latent = einops.rearrange(output, 'b (new_h new_w) (block_h block_w) c -> b new_h new_w block_h block_w c', block_h=block_h, block_w=block_w, new_h=new_h, new_w=new_w)
    latent = einops.rearrange(latent, 'b new_h new_w block_h block_w c -> b (new_h block_h new_w block_w) c')
    latent = einops.rearrange(latent, 'N (h1 w1) (h2 w2 C) -> N C (h1 h2) (w1 w2)', h1=(args.image_size//8)//args.patch_size, w1=(args.image_size//8)//args.patch_size, h2=args.patch_size, w2=args.patch_size)
    with torch.no_grad():
        samples = vae.decode(latent / vae.config.scaling_factor).sample
    for i, sample in enumerate(samples):
        save_image(sample, f"sample_img_{labels[i].item()}.png", normalize=True, value_range=(-1, 1))
    
if  __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default='./ACDiT-H-img.pt')
    parser.add_argument("--vae_path", type=str, default="facebook/DiT-XL-2-256")
    parser.add_argument("--cfg", type=float, default=1.5)
    parser.add_argument("--sample_step", type=int, default=25)
    parser.add_argument("--global_seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.global_seed)
    random.seed(args.global_seed)
    np.random.seed(args.global_seed)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    args_model = ckpt['args']
    model = ACDiT_models[args_model.model](
        latent_size=args_model.vae_latent_size*args_model.patch_size**2,
        block_size=args_model.block_size,
        num_classes=1000,
        ar_len=args_model.ar_len,
        temporal_len=args_model.num_frames,
        spatial_len= (args_model.image_size // args_model.vae_patch_pixels // args_model.patch_size)**2,
        nd_split = args_model.nd_split,
        square_block=True
    ).to('cuda')
    model.load_state_dict(ckpt['model'])

    vae = AutoencoderKL.from_pretrained(args.vae_path, subfolder='vae').to('cuda')
    labels = [207, 360, 387, 974, 88, 979, 417, 279]
    labels = torch.tensor(labels).to('cuda')
    samples_case(labels, model, vae, args.cfg, args.sample_step, args_model)
    

