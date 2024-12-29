import os
import torch
import numpy as np
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, AutoencoderKL
import einops
import math
from models import ACDiT_models
import argparse
import imageio
import pandas as pd
import random

def samples_case(labels, model, vae, cfg, sample_step, args):
    scheduler = DDIMScheduler()
    sample_scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config)
    output = model.sample(labels, sample_scheduler, sample_step, cfg=cfg, target_shape=(labels.size(0), args.ar_len, args.block_size, args.vae_latent_size*args.patch_size**2), dtype=torch.float)
    latent = einops.rearrange(output, 'N T B C -> N (T B) C')
    latent = einops.rearrange(latent, 'N (T B) C -> N T B C', T=args_model.num_frames)
    latent = einops.rearrange(latent, 'N T (h1 w1) (h2 w2 C) -> N T C (h1 h2) (w1 w2)', h1=(args_model.image_size//args_model.vae_patch_pixels )//args_model.patch_size, w1=(args_model.image_size//args_model.vae_patch_pixels )//args_model.patch_size, h2=args_model.patch_size, w2=args_model.patch_size)

    extra_shape, image_shape = latent.shape[:2], latent.shape[2:]
    latent = latent.reshape(-1, *image_shape)
    with torch.no_grad():
        output = vae.decode(latent / vae.config.scaling_factor).sample
    output = output.reshape(labels.size(0), -1, *output.shape[1:])
    for i, output_i in enumerate(output):
        output_i = ((output_i * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
        output_filename = f'output_video_{labels[i].item()}.mp4'
        imageio.mimwrite(output_filename, output_i, fps=4, quality=9)
    return output

if  __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--vae_path", type=str, default="facebook/DiT-XL-2-256")
    parser.add_argument("--save_dir", type=str, default="samples")
    parser.add_argument("--model", type=str, choices=list(ACDiT_models.keys()), default="ArDiT-B/2")
    parser.add_argument("--global_batch_size", type=int, default=256)
    parser.add_argument("--cfg", type=float, default=1.5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sample_step", type=int, default=25)
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--eval_ema", action="store_true")
    args = parser.parse_args()
    
    torch.manual_seed(args.global_seed)
    random.seed(args.global_seed)
    np.random.seed(args.global_seed)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    args_model = ckpt['args']
    model = ACDiT_models[args_model.model](
        latent_size=args_model.vae_latent_size*args_model.patch_size**2,
        block_size=args_model.block_size,
        num_classes=101,
        ar_len=args_model.ar_len,
        temporal_len=args_model.num_frames, # for position embedding's temporal dimension, different from ar_len
        spatial_len= (args_model.image_size // args_model.vae_patch_pixels // args_model.patch_size)**2,  # for position embedding's spatial dimension, different from block_size
        nd_split = args_model.nd_split,
        square_block=False
    ).to('cuda')
    model.load_state_dict(ckpt['model'])
    vae = AutoencoderKL.from_pretrained(args.vae_path, subfolder='vae').to('cuda')
    
    labels = [27, 36, 38, 97, 88, 79, 41, 29]
    labels = torch.tensor(labels).to('cuda')
    samples_case(labels, model, vae, args.cfg, args.sample_step, args_model)
