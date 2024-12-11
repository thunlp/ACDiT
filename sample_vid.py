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

# @torch.no_grad()
# def generation(args, model=None, vae=None, savefile=True, rank=None, world_size=None):
#     ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.eval_dtype]
#     if vae is None:
#         vae_config = VAE_CONFIG[args.vae_path]
#         vae = AutoencoderKL.from_pretrained(args.vae_path, subfolder=vae_config["subfolder"]).to('cuda').to(ptdtype)
#     if model is None:
#         ckpt = args.ckpt #'hdfs://haruna/home/byte_data_seed/ssd_hldy/user/hujinyi/ardiff/arvideodit_b_2_ditvae_fixarange-2024-08-24-18-40-59/000-ArVideoDiT-B_2/checkpoints/0030000.pt'
#         if ckpt.startswith("hdfs"):
#             hdfs_file = hopen(ckpt)
#             buffer = io.BytesIO()
#             buffer.write(hdfs_file.read())
#             buffer.seek(0)
#             hdfs_file.close()
#             checkpoint = torch.load(buffer, map_location="cpu")
#         else:
#             checkpoint = torch.load(ckpt,  map_location="cpu")

#         args_model = checkpoint['args']
#         model = DiT_models[args_model.model](
#             latent_size=args_model.vae_latent_size*args_model.patch_size**2,
#             block_size=args_model.block_size,
#             num_classes=args.num_classes,
#             use_rope=args_model.use_rope,
#             no_qk_norm=args_model.no_qk_norm,
#             ar_len=args_model.ar_len,
#             temporal_len=args_model.num_frames, # for position embedding's temporal dimension, different from ar_len
#             spatial_len=(args_model.image_size // args_model.vae_patch_pixels // args_model.patch_size)**2,  # for position embedding's spatial dimension, different from block_size
#             spatial_2d=args_model.spatial_2d,
#             nd_split=args_model.nd_split,
#             square_block=args_model.square_block or args_model.dataset == 'ImageNet'
#         ).to('cuda')

#         if args.eval_ema:
#             model.load_state_dict(checkpoint['ema'])
#         else:
#             model.load_state_dict(checkpoint['model'])
        
#     else:
#         args_model = args

#     scheduler = DDIMScheduler()
#     if args.sample_scheduler == "DDIM":
#         sample_scheduler = DDIMScheduler(rescale_betas_zero_snr=True, timestep_spacing="trailing")
#     elif args.sample_scheduler == "DPMSolver":
#         sample_scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config)
#     elif args.sample_scheduler == "DDPM":
#         sample_scheduler = DDPMScheduler()
#     all_labels = torch.arange(0, args.num_classes).to("cuda")


#     if args.dataset == "UCF101":
#         label2name = pd.read_csv("data/ucf101.label2class.csv", header=None)
#     else:
#         label2name = None

#     if world_size == 1 or world_size is None:
#         batch_size_per_rank = args.global_batch_size
#         n_chunk = (args.num_classes - 1)//batch_size_per_rank + 1
#         labels_this_rank = all_labels
#         index_offset = 0
#     else:
#         batch_size_per_rank = args.global_batch_size // world_size
#         classes_per_rank = (args.num_classes - 1)//world_size + 1
#         labels_this_rank = all_labels[classes_per_rank*rank: classes_per_rank*(rank+1)]
#         index_offset = classes_per_rank*rank
#         n_chunk = (len(labels_this_rank) - 1)//batch_size_per_rank + 1
        
        
#     from datetime import datetime
#     nowstr = datetime.now().strftime("%m%d%H%M")

#     os.makedirs(name=f"generated/{nowstr}", exist_ok=True)

#     all_output_filename = []
#     generated_num = 0
#     for cid in tqdm.tqdm(range(n_chunk)):
#         labels = labels_this_rank[batch_size_per_rank*cid : batch_size_per_rank*(cid+1)]
#         if len(labels) == 0:
#             continue
#         print('core generate')
#         output = core_generate(model=model, 
#                                 vae=vae, 
#                                 labels=labels, 
#                                 sample_scheduler=sample_scheduler, 
#                                 target_shape=(labels.size(0), args_model.ar_len, args_model.block_size, args_model.vae_latent_size*args_model.patch_size**2), 
#                                 ptdtype=ptdtype, 
#                                 args=args,
#                                 args_model=args_model, 
#                                 device=labels.device,
#                                 unconditional=False, 
#                                 with_tqdm=(True if rank==0 else False))

#         if savefile:
#             for i, output_i in enumerate(output):
#                 labelstr = label2name.iloc[int(labels[i].cpu())][1] if label2name is not None else ""
#                 output_i = ((output_i * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
                
#                 if output_i.shape[0] == 1: # is image
#                     output_filename = f'generated/{nowstr}/image{index_offset + generated_num+ i}{labelstr}_{args.inference_step}.png'
#                     imageio.imwrite(output_filename, output_i[0])
#                 else:
#                     output_filename = f'generated/{nowstr}/video{index_offset + generated_num + i}{labelstr}_{args.inference_step}.mp4'
#                     imageio.mimwrite(output_filename, output_i, fps=4, quality=9)
#                 all_output_filename.append(output_filename)
                

    
#     # if world_size > 1:
#     #     if savefile:
#     #         all_output_filename = "&".join(all_output_filename) if len(all_output_filename) > 0 else ""
#     #         gathered_output_filename = [None for i in range(world_size)]
#     #         dist.all_gather_object(gathered_output_filename, all_output_filename)
#     #         gathered_output_filename = [x.split("&") for x in gathered_output_filename if len(x) > 0]
#     #         gathered_output_filename = [y for x in gathered_output_filename for y in x]
#     #     else:
#     #         gathered_output_filename = []
#     # else:
#     #     gathered_output_filename = all_output_filename if savefile else []
#     return all_output_filename


# if  __name__=="__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--train_dataset_path", type=str, default='hdfs://haruna/home/byte_data_seed/ssd_hldy/user/shengdinghu/video_data/ucf101/*.parquet')
#     parser.add_argument("--valid_dataset_path", type=str, default='hdfs://haruna/home/byte_data_seed/ssd_hldy/user/shengdinghu/video_data/ucf101/*.parquet')
#     parser.add_argument("--image_tag", type=str, default='video')
#     parser.add_argument("--label_tag", type=str, default='label')
#     parser.add_argument("--ckpt", type=str, default='hdfs://haruna/home/byte_data_seed/ssd_hldy/user/hujinyi/ardiff', help='please specify a hdfs disk path, if not, local path')
#     parser.add_argument("--vae_path", type=str, default="sdxl-vae")
#     parser.add_argument("--vae_patch_pixels", type=int, default=8, choices=[8, 1], help="1 for pixel level diffusion")
#     parser.add_argument("--results_dir", type=str, default="results")
#     parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="ArDiT-B/2")
#     parser.add_argument("--dataset", type=str, choices=['UCF101', 'ImageNet', 'visualtext'], default='UCF101')
#     parser.add_argument("--image_size", type=int, choices=[256, 512, 16], default=256, help="16 for visual text")
#     parser.add_argument("--num_classes", type=int, default=101)
#     parser.add_argument("--num_scales", type=int, default=8)
#     parser.add_argument("--pretrained_ckpt", type=str, default=None)
#     parser.add_argument("--num_frames", type=int, default=1) # set 1 for image
#     parser.add_argument("--lr", type=float, default=1e-4)
#     parser.add_argument("--weight_decay", type=float, default=5e-2, help="Weight decay to use.")
#     parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
#     parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
#     parser.add_argument("--epochs", type=int, default=1400)
#     parser.add_argument("--global_batch_size", type=int, default=256)
#     parser.add_argument("--global_seed", type=int, default=0)
#     parser.add_argument("--cfg", type=float, default=4.0)
#     parser.add_argument("--num_workers", type=int, default=4)
#     parser.add_argument("--images_per_label", type=int, default=25)
#     parser.add_argument("--mixed_precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
#     parser.add_argument("--eval_dtype", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
#     parser.add_argument("--patch_size", default=1, type=int, help="patch_size x patch_size of vae latents forms an input feature")
#     parser.add_argument("--vae_latent_size", default=4, type=int, help="vae's latent feature size")
#     parser.add_argument("--block_size", default=16, type=int, help="vae's latent feature size")
#     parser.add_argument("--block_use_2d_pos", action="store_true", help="vae's latent feature size")
#     parser.add_argument("--block_pos_seq_integrate", default="add", type=str, choices=['add', 'cat'], help="whether to concate pos_emb")
#     parser.add_argument("--square_block", action="store_true")
#     parser.add_argument("--use_rope", action="store_true")
#     parser.add_argument("--only_eval", action="store_true")
#     parser.add_argument("--project_name", type=str, default="acdit_ucf")
#     parser.add_argument("--ar_len", type=int, default=1)
#     parser.add_argument("--inference_step", type=int, default=100)
#     parser.add_argument("--frame_interval", type=int, default=3)
#     parser.add_argument("--no_qk_norm", action="store_true")
#     parser.add_argument("--spatial_2d", action="store_true")
#     parser.add_argument("--nd_split", type=str, default=None, help="The split for rope emb in different dmension (temporal[optional], spatial1, spatial2[optional]) input a list separated by _  e.g., 2_1_1 for [2, 1, 1]. Position ids should also follow the same order")
#     parser.add_argument("--do_eval", action="store_true")
#     parser.add_argument("--eval_ema", action="store_true")
#     parser.add_argument("--max_eval_num", type=int, default=2048, help="for UCF fvd")
#     parser.add_argument("--gen_num", type=int, default=128, help="for online generation")
#     parser.add_argument("--sample_scheduler", type=str, default="DPMSolver", choices=["DPMSolver", "DDIM", "DDPM"], help="sample scheduler")



#     args = parser.parse_args()
#     world_size = int(os.environ.get("WORLD_SIZE", 1))
#     if world_size > 1:
#         init_ddp()
#         rank = dist.get_rank()
#     else:
#         rank = 0
#     if args.do_eval:
#         evaluate(args)
#     else:
#         generation(args, rank=rank, world_size=world_size)