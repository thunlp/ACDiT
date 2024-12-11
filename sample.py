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
    samples = vae.decode(latent / vae.config.scaling_factor).sample
    for i, sample in enumerate(samples):
        save_image(sample, f"output_{labels[i].item()}.png", normalize=True, value_range=(-1, 1))
    
if  __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default='hdfs://haruna/home/byte_data_seed/ssd_hldy/user/hujinyi/ardiff', help='please specify a hdfs disk path, if not, local path')
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

    with fsspec.open(args.ckpt) as f:
        ckpt = torch.load(f, map_location="cpu")
    args_model = ckpt['args']
    model = ACDiT_models[args.model](
        latent_size=args_model.vae_latent_size*args_model.patch_size**2,
        block_size=args_model.block_size,
        num_classes=args_model.num_classes,
        ar_len=args_model.ar_len,
        temporal_len=args_model.num_frames, # for position embedding's temporal dimension, different from ar_len
        spatial_len= (args_model.image_size // args_model.vae_patch_pixels // args_model.patch_size)**2,  # for position embedding's spatial dimension, different from block_size
        nd_split = args_model.nd_split,
        square_block=args_model.square_block or args_model.dataset == 'ImageNet'
    ).to('cuda')
    model.load_state_dict(ckpt)
    vae = AutoencoderKL.from_pretrained(args.vae_path, subfolder='vae').to('cuda')
    os.makedirs(args.save_dir, exist_ok=True)
    labels = [207, 360, 387, 974, 88, 979, 417, 279]
    labels = torch.tensor(labels).to('cuda')
    samples_case(labels, model, vae, args.cfg, args.sample_step, args_model)