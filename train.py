import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusers import DDPMScheduler
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
import time
import argparse
import logging
import os

import fsspec
import json
import math
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from models import DiT_models
from diffusers import AutoencoderKL, DDIMScheduler, DPMSolverMultistepScheduler
from torchvision.utils import save_image
from sample_ddp import distributed_sampling
import wandb
import einops
from inception import InceptionV3
from functools import partial
from torch.optim import lr_scheduler

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data.to(torch.float32), alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

#################################################################################
#                                  Training Loop                                #
#################################################################################



def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    args.batch_size_per_rank = int(args.global_batch_size / dist.get_world_size())
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()} on device cuda:{device}.")
        
    # Setup an experiment folder:
    model_string_name = args.model.replace('/', '_')
    experiment_dir = f"{args.results_dir}_{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")
    if rank == 0:
        wandb.init(project=args.project_name, name=args.results_dir)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    
    model = DiT_models[args.model](
        latent_size=args.vae_latent_size*args.patch_size**2,
        block_size=args.block_size,
        num_classes=args.num_classes,
        use_rope=args.use_rope,
        ar_len=args.ar_len,
        temporal_len=args.num_frames, # for position embedding's temporal dimension, different from ar_len
        spatial_len=(args.image_size // args.vae_patch_pixels // args.patch_size)**2,  # for position embedding's spatial dimension, different from block_size
        spatial_2d=args.spatial_2d,
        nd_split=args.nd_split,
        square_block=args.square_block or args.dataset == 'ImageNet'
    ).to(device)

    if dist.get_rank() == 0:
        print("="*30)
        print(model)
        print("="*30)
        
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device).to(torch.float32)  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    vae = AutoencoderKL.from_pretrained(args.vae_path, subfolder='vae').to(device).eval()

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_model = InceptionV3([block_idx]).to(device)

    if dist.get_rank() == 0:
        print("="*30)
        print(model)
        print("="*30)
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2))
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params}")

    scheduler = DDPMScheduler()
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    model = DDP(model.to(device), device_ids=[device])

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    step_per_epoch = len(dataloader)
    total_steps = args.epochs * step_per_epoch
    if args.lr_scheduler == "wsd":
        def lambda_wsd(step, stages, end_lr_exponent=6):
            if step < stages[0]:
                return step /stages[0]
            elif step < stages[1]:
                return 1.0
            else:
                ratio = (step - stages[1])/(stages[2] - stages[1])
                return 0.5**(ratio*end_lr_exponent)
        lrs = lr_scheduler.LambdaLR(opt, lr_lambda=partial(lambda_wsd, stages=[int(args.lr_warmup*total_steps), int(0.85*total_steps), total_steps], end_lr_exponent=6))
    elif args.lr_scheduler == "cosine":
        def lambda_cosine(step, stages, end_lr_ratio=0):
            if step < stages[0]:
                return step /stages[0]
            else:
                ratio = (step - stages[0])/(stages[1] - stages[0])
                return end_lr_ratio + 0.5 * (1 - end_lr_ratio) * (1 + math.cos(ratio*math.pi))
        lrs = lr_scheduler.LambdaLR(opt, lr_lambda=partial(lambda_cosine, stages=[int(args.lr_warmup*total_steps), total_steps], end_lr_ratio=0.01))
    else:
        lrs = None

    logger.info(f"total steps: {total_steps}")


    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params}")
    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    logger.info(f"Training for {args.epochs} epochs...")
    vae_scaling_factor = vae.config.scaling_factor
    vae.scaling_factor = vae.config.scaling_factor

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            if args.dataset == 'ImageNet':
                x = x.unsqueeze(1) # [N, C, H, W] -> [N, 1, C, H, W]

            batch_size, num_frames = x.shape[:2]
            with torch.inference_mode():
                # Map input images to latent space + normalize latents:
                image_shape = x.shape[-3:]
                latent = vae.encode(x.view(-1, *image_shape)).latent_dist.sample().mul_(vae_scaling_factor)
                _, _, latent_h, latent_w = latent.size()
                latent = einops.rearrange(latent, 'N C (h1 h2) (w1 w2) -> N (h1 w1) (h2 w2 C)', h2=args.patch_size, w2=args.patch_size)
            latent = latent.clone() # other wise it is an inference tensor

            if args.square_block:
                assert math.isqrt(args.block_size) ** 2 == args.block_size
                latent = einops.rearrange(latent, 'N (H W) C-> N H W C', H=latent_h//args.patch_size, W=latent_w//args.patch_size)
                block_h = block_w = int(math.isqrt(args.block_size))
                latent = einops.rearrange(latent, 'N (new_h block_h) (new_w block_w) c -> N new_h new_w block_h block_w c', block_h=block_h, block_w=block_w)
                latent = einops.rearrange(latent, '(N T) new_h new_w block_h block_w c -> N (T new_h new_w) (block_h block_w) c', T=args.num_frames)
            else:
                latent = einops.rearrange(latent, '(N T) B C -> N T B C', N=batch_size)
                latent = einops.rearrange(latent, 'N T B C -> N (T B) C')
                latent = einops.rearrange(latent, 'N (T B) C-> N T B C', B=args.block_size) # For multi-frames
            
            N, T, B, C = latent.shape
            noise = torch.randn_like(latent)
            t = torch.randint(0, scheduler.num_train_timesteps, (batch_size, T))
            noised_latent = scheduler.add_noise(latent.reshape(-1, 1, B, C), noise.reshape(-1, 1, B, C), t.reshape(-1)).reshape(N, T, B, C)
            
                
            opt.zero_grad()
            with torch.cuda.amp.autocast(dtype=ptdtype):
                model_output = model(latent, noised_latent, t, y)
                noise = einops.rearrange(noise, 'N T B C-> N (T B) C', B=args.block_size)
                loss = F.mse_loss(model_output, noise)
            
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if args.max_grad_norm != 0.0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.module.parameters(), args.max_grad_norm)
            
            if rank == 0:
                wandb.log({'loss': loss.item(), 'grad_norm': grad_norm, 'lr': opt.param_groups[0]['lr']}, step=train_steps)

            scaler.step(opt)
            scaler.update()
            if lrs is not None:
                lrs.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, LR: {opt.param_groups[0]['lr']}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()

            # Eval FiD
        if (epoch + 1) % args.eval_every == 0:
            model.eval()
            logger.info(f"Sampling images for eval fid")
            fid_value_cfg = distributed_sampling(dist.get_rank(), dist.get_world_size(), model.module, vae, inception_model, batch_size=args.batch_size_per_rank, images_per_label=args.images_per_label, total_labels=args.num_classes, device=device, cfg=args.cfg, args=args)
            fid_value_wocfg = distributed_sampling(dist.get_rank(), dist.get_world_size(), model.module, vae, inception_model, batch_size=args.batch_size_per_rank, images_per_label=args.images_per_label, total_labels=args.num_classes, device=device, cfg=1, args=args)
            fid_value_cfg_ema = distributed_sampling(dist.get_rank(), dist.get_world_size(), ema, vae, inception_model, batch_size=args.batch_size_per_rank, images_per_label=args.images_per_label, total_labels=args.num_classes, device=device, cfg=args.cfg, args=args)
            fid_value_wocfg_ema = distributed_sampling(dist.get_rank(), dist.get_world_size(), ema, vae, inception_model, batch_size=args.batch_size_per_rank, images_per_label=args.images_per_label, total_labels=args.num_classes, device=device, cfg=1, args=args)

            if rank == 0:
                logger.info(f"Fid value at {train_steps} without cfg: {fid_value_wocfg}; with cfg: {fid_value_cfg}; EMA: without cfg: {fid_value_wocfg_ema}; with cfg: {fid_value_cfg_ema}")
                wandb.log({'fid_value_wocfg': fid_value_wocfg, f"fid_value_withcfg{args.cfg}": fid_value_cfg, 'EMA/fid_value_wocfg': fid_value_wocfg_ema, f"EMA/fid_value_withcfg{args.cfg}": fid_value_cfg_ema}, step=train_steps)

            if rank == 0 and args.dataset == 'ImageNet':
                labels = torch.randint(0, args.num_classes, (16, )).to(device)
                ddim = DDIMScheduler()
                sample_scheduler = DPMSolverMultistepScheduler.from_config(ddim.config)
                with torch.no_grad():
                    output = model.module.sample(labels, sample_scheduler, 25, cfg=4, target_shape=(labels.size(0), args.ar_len, args.block_size, args.vae_latent_size*args.patch_size**2), dtype=torch.float)
                    if args.square_block:
                        block_h = block_w = int(math.isqrt(args.block_size))    
                        new_h = new_w = int(math.isqrt(output.size(1)))
                        latent = einops.rearrange(output, 'b (new_h new_w) (block_h block_w) c -> b new_h new_w block_h block_w c', block_h=block_h, block_w=block_w, new_h=new_h, new_w=new_w)
                        latent = einops.rearrange(latent, 'b new_h new_w block_h block_w c -> b (new_h block_h new_w block_w) c')
                        latent = einops.rearrange(latent, 'N (h1 w1) (h2 w2 C) -> N C (h1 h2) (w1 w2)', h1=(args.image_size//args.vae_patch_pixels)//args.patch_size, w1=(args.image_size//args.vae_patch_pixels)//args.patch_size, h2=args.patch_size, w2=args.patch_size)
                    else:
                        latent = einops.rearrange(output, 'N T B C -> N (T B) C', B=args.block_size)
                        latent = einops.rearrange(latent, 'N (h1 w1) (h2 w2 C) -> N C (h1 h2) (w1 w2)', h1=(args.image_size//args.vae_patch_pixels)//args.patch_size, w1=(args.image_size//args.vae_patch_pixels)//args.patch_size, h2=args.patch_size, w2=args.patch_size)
                    output = vae.decode(latent / vae_scaling_factor).sample
                save_image(output, 'output_image.png', nrow=8, normalize=True, value_range=(-1, 1))
                wandb.log({
                        'recon': wandb.Image('output_image.png')
                    }, step = train_steps
                )
            model.train()
        # Save DiT checkpoint:
        if (epoch + 1) % args.ckpt_every == 0 and 'test' not in args.results_dir:
            if rank == 0:
                checkpoint = {
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "lrs": lrs.state_dict() if lrs is not None else {},
                    "args": args
                }
                save_epoch = epoch + 1
                checkpoint_path = f"{checkpoint_dir}/epoch_{save_epoch:04d}.pt"
                with fsspec.open(checkpoint_path, 'wb') as f:
                    torch.save(checkpoint, f)

                training_info = {
                    'training_step': train_steps,
                    'epoch': save_epoch
                }
                with fsspec.open(f"{checkpoint_dir}/training_info.json", 'w') as f:
                    json.dump(training_info, f)
                logger.info(f"Saved checkpoint in {checkpoint_dir}")
        dist.barrier()
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--vae_path", type=str, default="sdxl-vae")
    parser.add_argument("--vae_patch_pixels", type=int, default=8, choices=[8,16])
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="ACDiT_132M")
    parser.add_argument("--dataset", type=str, choices=['UCF101', 'ImageNet'], default='ImageNet')
    parser.add_argument("--num_classes", type=int, default=101)
    parser.add_argument("--num_scales", type=int, default=8)
    parser.add_argument("--num_frames", type=int, default=16) # set 1 for image
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-2, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global_batch_size", type=int, default=256)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--images_per_label", type=int, default=25)
    parser.add_argument("--mixed_precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--image_size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--patch_size", default=1, type=int, help="patch_size x patch_size of vae latents forms an input feature")
    parser.add_argument("--vae_latent_size", default=4, type=int, help="vae's latent feature size")
    parser.add_argument("--block_size", default=16, type=int, help="vae's latent feature size")
    parser.add_argument("--square_block", action="store_true")
    parser.add_argument("--ar_len", type=int, default=16)
    parser.add_argument("--use_rope", action="store_true")
    parser.add_argument("--project_name", type=str, default="acdit_ucf")
    parser.add_argument("--spatial_2d", action="store_true")
    parser.add_argument("--nd_split", type=str, default=None, help="The split for rope emb in different dmension (temporal[optional], spatial1, spatial2[optional]) input a list separated by _  e.g., 2_1_1 for [2, 1, 1]. Position ids should also follow the same order")
    parser.add_argument("--lr_scheduler", type=str, default="wsd", choices=["wsd", "cosine", "constant"])
    parser.add_argument("--lr_warmup", type=float, default=0.0)
    args = parser.parse_args()
    print(args)
    main(args)