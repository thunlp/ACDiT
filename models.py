import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Attention, Mlp
import einops

from diffusers.utils.torch_utils import randn_tensor
from rope import RopeND

try:
    from torch.nn.attention.flex_attention import flex_attention, BlockMask
    flex_attention = torch.compile(flex_attention)
    from torch.nn.attention.flex_attention import create_block_mask
    from functools import partial
    USE_FLEX_ATTENTION = True
    print("Use flex attention!!!!")
except:
    USE_FLEX_ATTENTION = False
    print('Not use flex attention')

torch._dynamo.config.optimize_ddp=False


def modulate(x, shift, scale):
    # Suppose x is (B, N, D), shift is (B, D), scale is (B, D)
    x = x * (scale + 1) + shift
    return x

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000, dtype=torch.float32):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=dtype) / half
        ).to(device=t.device, dtype=dtype)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype=torch.bfloat16):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size, dtype=dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class SkipCausalAttention(Attention):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        rope=None,
        qk_norm=True,
        **block_kwargs,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, **block_kwargs)
        self.caching, self.cached_k, self.cached_v = False, None, None
        self.rope = rope

    def set_caching(self, flag):
        self.caching, self.cached_k, self.cached_v = flag, None, None

    def forward(self, x, position_ids=None, attention_mask=None, block_size=None, cache=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q, k = self.rope(q, k, position_ids)

        if self.caching:
            if cache:
                if self.cached_k is None:
                    self.cached_k = k[:, :, :block_size, :]
                    self.cached_v = v[:, :, :block_size, :]
                    self.cached_x = x
                else:
                    self.cached_k = torch.cat((self.cached_k, k[:, :, :block_size, :]), dim=2)
                    self.cached_v = torch.cat((self.cached_v, v[:, :, :block_size, :]), dim=2)

            if self.cached_k is not None:
                k = torch.cat((self.cached_k, k[:, :, -block_size:, :]), dim=2)
                v = torch.cat((self.cached_v, v[:, :, -block_size:, :]), dim=2)

        if not USE_FLEX_ATTENTION or not isinstance(attention_mask, BlockMask):
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.attn_drop.p
            )
        else:
            x = flex_attention(q, k, v, block_mask=attention_mask)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CustomDiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, rope=None, qk_norm=True,   **block_kwargs):
        super().__init__()
        
        self.norm1 = LlamaRMSNorm(hidden_size, eps=1e-6)
        self.attn = SkipCausalAttention(hidden_size, num_heads=num_heads, qkv_bias=True, norm_layer=LlamaRMSNorm, qk_norm=qk_norm, rope=rope, **block_kwargs)
        self.norm2 = LlamaRMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, attention_mask=None, cond_length=0, block_size=None, cache=False, position_ids=None):
        dtype = x.dtype

        N, T, _, C = c.shape  # for inference its T, _ = 1, 1
        N, TB, C = x[:, cond_length:].shape  # for inference TB = 1*B
        B = TB // T

        ada_c_list = self.adaLN_modulation(c).chunk(6, dim=-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [ada_c.repeat(1, 1, B, 1).reshape(N, TB, C) for ada_c in ada_c_list]
        norm_x1 = self.norm1(x.to(torch.float32)).to(dtype)
        attn_input_x = torch.cat((norm_x1[:, :cond_length], modulate(norm_x1[:, cond_length:], shift_msa, scale_msa)), dim=1)
        attn_output_x = self.attn(attn_input_x, attention_mask=attention_mask, block_size=block_size, cache=cache, position_ids=position_ids)
        x = x + torch.cat((attn_output_x[:, :cond_length], gate_msa * attn_output_x[:, cond_length:]), dim=1)
        
        norm_x2 = self.norm2(x.to(torch.float32)).to(dtype)
        gate_input_x = torch.cat((norm_x2[:, :cond_length], modulate(norm_x2[:, cond_length:], shift_mlp, scale_mlp)), dim=1)
        gate_output_x = self.mlp(gate_input_x)
        x = x + torch.cat((gate_output_x[:, :cond_length], gate_mlp * gate_output_x[:, cond_length:]), dim=1)
        return x

    def set_caching(self, flag):
        self.attn.set_caching(flag)

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, outputsize):
        super().__init__()
        self.norm_final = LlamaRMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, outputsize, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        N, T, _, C = c.shape  # for inference its T, _ = 1, 1
        N, TB, C = x.shape  # for inference TB = 1*B
        B = TB // T
        ada_c_list = self.adaLN_modulation(c).chunk(2, dim=-1)
        shift, scale = [ada_c.repeat(1, 1, B, 1).reshape(N, TB, C) for ada_c in ada_c_list]
        norm_x = self.norm_final(x.to(torch.float32)).to(x.dtype)
        x = modulate(norm_x, shift, scale)
        x = self.linear(x)
        return x

def skip_causal_attn_mask_mod_gen(b, h, q_idx, kv_idx, block_size, len1):
    q_idx = q_idx // block_size
    kv_idx = kv_idx // block_size
    mask = torch.where(((q_idx < len1) & (kv_idx < len1) & (q_idx >= kv_idx))| ((q_idx >= len1) & (kv_idx < len1) & ((q_idx - len1) > kv_idx)) | (q_idx == kv_idx) , True, False)
    return mask

class ACDiT(nn.Module):
    def __init__(
        self,
        patch_size=2,
        latent_size=16,
        block_size=256,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=101,
        learn_sigma=False,
        num_frames=16,
        no_qk_norm=False,
        ar_len=4,
        temporal_len=1,
        spatial_len=1024,
        nd_split=None,
        square_block=True
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.mlp_ratio = mlp_ratio
        self.depth = depth

        self.latent_size = latent_size
        self.block_size = block_size
        self.qk_norm = not no_qk_norm
        self.ar_len = ar_len
        self.temporal_len = temporal_len 
        self.spatial_len = spatial_len
        self.square_block = square_block
        assert self.spatial_len * self.temporal_len == self.block_size * self.ar_len, f"Split block invalid: {self.spatial_len} * {self.temporal_len} != {self.block_size} * {self.ar_len}"

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.x_embedder = nn.Linear(in_features=self.latent_size, out_features=self.hidden_size, bias=False)
        
        if self.temporal_len == 1: # pure image, no need for temporal dim
            nd = 2
            h = w = math.isqrt(self.spatial_len) 
            self.max_lens = [h, w] # a bit redundancy for future extrapolation 
            self.nd_split = [1, 1] if nd_split is None else [int(x) for x in nd_split.split("_")]
        else:
            nd = 3
            h = w = math.isqrt(self.spatial_len) 
            self.max_lens  = [self.temporal_len, h, w]
            self.nd_split = [2, 3, 3] if nd_split is None else [int(x) for x in nd_split.split("_")] # 16 dim for temporal, 24 for h, w each

        self.rope = RopeND(nd=nd, nd_split=self.nd_split, max_lens=self.max_lens)
        def create_index_tensor(max_lens):
            ranges = [torch.arange(m) for m in max_lens]
            grids = torch.meshgrid(*ranges, indexing='ij')
            return torch.stack(grids).reshape(len(max_lens), -1)
        self.position_ids_precompute = create_index_tensor(self.max_lens)
        if square_block:
            position_idx = torch.arange(self.spatial_len)
            h = w = int(math.isqrt(self.spatial_len))
            position_idx = position_idx.view(h, w)
            block_h = block_w = int(math.isqrt(self.block_size))
            position_idx = einops.rearrange(position_idx, '(new_h block_h) (new_w block_w)-> new_h new_w block_h block_w', block_h=block_h, block_w=block_w)
            position_idx = einops.rearrange(position_idx, 'new_h new_w block_h block_w -> (new_h new_w) (block_h block_w)')
            position_idx = position_idx.view(-1)
            self.position_ids_precompute = self.position_ids_precompute[:, position_idx]
        self.blocks = nn.ModuleList([
            CustomDiTBlock(self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio, rope=self.rope, qk_norm=self.qk_norm) for _ in range(self.depth)
        ])
        self.final_layer = FinalLayer(self.hidden_size, self.latent_size)
        self.initialize_weights()
        self.init_flex_attn()

    def init_flex_attn(self):
        if USE_FLEX_ATTENTION:
            skip_causal_attn_mask_mod = partial(skip_causal_attn_mask_mod_gen, block_size=self.block_size, len1=self.ar_len)
            self.flex_attnmask = create_block_mask(skip_causal_attn_mask_mod, B=None, H=None, Q_LEN=2*self.ar_len*self.block_size, KV_LEN=2*self.ar_len*self.block_size)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        try:
            w = self.x_embedder.proj.weight.data
        except:
            w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        try:
            nn.init.constant_(self.x_embedder.proj.bias, 0)
        except:
            if self.x_embedder.bias is not None:
                nn.init.constant_(self.x_embedder.bias, 0)


        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def build_attention_mask(self, T, B, device):
        size = T * B
        m_noise_noise = torch.zeros(size, size)
        for i in range(T):
            start_idx = i * B
            end_idx = start_idx + B
            m_noise_noise[start_idx:end_idx, start_idx:end_idx] = torch.ones(B, B)
        m_noise_clean = torch.zeros(size, size)
        for i in range(T):
            for j in range(i + 1, T):
                start_col = i * B
                end_col = start_col + B
                start_row = j * B
                end_row = start_row + B
                m_noise_clean[start_row:end_row, start_col:end_col] = 1
        m_clean_noise = torch.zeros(size, size)
        m_clean_clean = torch.zeros(size, size)
        for i in range(T):
            start_idx = i * B
            end_idx = start_idx + B
            m_clean_clean[start_idx:end_idx, :end_idx] = 1

        attn_mask = torch.zeros(2 * size, 2 * size)
        attn_mask[:size, :size] = m_clean_clean
        attn_mask[:size, size:] = m_clean_noise
        attn_mask[size:, :size] = m_noise_clean
        attn_mask[size:, size:] = m_noise_noise
        return attn_mask.bool().to(device)

    def build_inference_attention_mask(self, block_id, B, device):
        attention_mask = torch.ones(2 * B, B * (block_id + 1))
        attention_mask[:B, -B:] = torch.zeros(B, B)
        attention_mask = attention_mask.bool().to(device)
        return attention_mask

    def forward(self, clean_x, noised_x, t, y):
        N, T, B, _ = clean_x.size()

        if t.dim() == 1:
            t = t.unsqueeze(1).repeat(1, T)
        
        t = self.t_embedder(t.reshape(-1), dtype=clean_x.dtype).reshape(N, T, -1)
        if y is None:
            y = torch.tensor([self.num_classes] * N, device=t.device)
        y = self.y_embedder(y, True)
        clean_x = self.x_embedder(clean_x)
        noised_x = self.x_embedder(noised_x)

        cond = t[:, :, None, :] + y[:, None, None, :]
        position_ids=torch.cat([self.position_ids_precompute , self.position_ids_precompute ], dim=-1) # for ACDiT clean and noise

        clean_x = einops.rearrange(clean_x, 'N T B C -> N (T B) C')
        noised_x = einops.rearrange(noised_x, 'N T B C -> N (T B) C')
        # clean x First!!!
        x = torch.cat((clean_x, noised_x), dim=1)   
        if not USE_FLEX_ATTENTION:
            attention_mask = self.build_attention_mask(T, B, x.device)
        else:
            attention_mask = self.flex_attnmask

        for block in self.blocks:
            x = block(x, cond, attention_mask, cond_length=T*B, position_ids=position_ids)
        x = self.final_layer(x[:, T * B:, :], cond)
        return x

    @torch.no_grad()
    def inference_forward(self, clean_x, noised_x, t, y, block_id, dtype=torch.bfloat16, cache=False):
        t = self.t_embedder(t, dtype=dtype)
        y = self.y_embedder(y, False)
        B = noised_x.size(2)
        Tc = clean_x.size(1) if clean_x is not None else 0
        if clean_x is not None:
            clean_x = self.x_embedder(clean_x)
            clean_x = einops.rearrange(clean_x, 'N T B C -> N (T B) C')
        
        noised_x = self.x_embedder(noised_x)
        noised_x = einops.rearrange(noised_x, 'N T B C -> N (T B) C')
        
        noise_ids = self.position_ids_precompute[:, block_id*self.block_size:(block_id+1)*self.block_size]  # torch.tensor([[block_id for j in range(B)], [j for j in range(B)]])
        if clean_x is not None and cache:
            clean_ids = self.position_ids_precompute[:, (block_id-1)*self.block_size:block_id*self.block_size]  #torch.tensor([[block_id - 1 for j in range(B)], [j for i in range(Tc) for j in range(B)]])
            position_ids = torch.cat([clean_ids, noise_ids], dim=-1) # for AC DiT clean and noise
        else:
            position_ids = noise_ids

        cond = t[:, None, None, :] + y[:, None, None, :] # T=1
        if clean_x is not None and cache:
            x = torch.cat((clean_x, noised_x), dim=1)
            attention_mask = self.build_inference_attention_mask(block_id, B, x.device)
        else:
            x = noised_x
            attention_mask = None
        cond_length = 0 if block_id == 0 or not cache else B
        for block in self.blocks:
            x = block(x, cond, attention_mask, position_ids=position_ids, cond_length=cond_length, block_size=B, cache=cache)
        x = self.final_layer(x[:, -B:, :], cond)
        return x

    @torch.no_grad()
    def sample(self, y, scheduler, num_inference_steps, cfg, target_shape, generator=None, dtype=torch.bfloat16):
        N, T, B, C = target_shape
        scheduler.set_timesteps(num_inference_steps, device=y.device)
        if cfg > 1:
            class_null = torch.tensor([self.num_classes] * N, device=y.device)
            y = torch.cat([y, class_null], 0)

        clean_latents = []
        for block in self.blocks:
            block.set_caching(True)
        for block_id in range(T):
            clean_x = clean_latents[-1] if len(clean_latents) > 0 else None
            noised_x = randn_tensor((N, 1, B, C), device=y.device, generator=generator, dtype=dtype)
            scheduler.set_timesteps(num_inference_steps, device=y.device)
            cache_flag = block_id > 0
            for t in scheduler.timesteps:
                if cfg > 1:
                    noised_x_double = torch.cat([noised_x, noised_x], dim=0)
                    clean_x_double = torch.cat([clean_x, clean_x], dim=0) if len(clean_latents) > 0 else None
                else:
                    noised_x_double = noised_x
                    clean_x_double = clean_x
                noised_x_double = scheduler.scale_model_input(noised_x_double, t)
                timesteps = torch.tensor([t] * noised_x_double.size(0), device=y.device)
                noise_pred = self.inference_forward(
                    clean_x_double, noised_x_double, timesteps, y, block_id, dtype, cache=cache_flag
                )
                if cfg > 1:
                    cond_eps, uncond_eps = noise_pred.chunk(2)
                    noise_pred = uncond_eps + cfg * (cond_eps - uncond_eps)
                noised_x = scheduler.step(noise_pred, t, noised_x.squeeze(1)).prev_sample.unsqueeze(1)
                cache_flag = False
            clean_latents.append(noised_x)
        clean_latents = torch.cat(clean_latents, dim=1)
        for block in self.blocks:
            block.set_caching(False)
        return clean_latents


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/inblock_pos_embed.py

def get_2d_sincos_inblock_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    inblock_pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    inblock_pos_embed = get_2d_sincos_inblock_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        inblock_pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), inblock_pos_embed], axis=0)
    return inblock_pos_embed


def get_2d_sincos_inblock_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_inblock_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_inblock_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_inblock_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_seq_pos_embed(embed_dim, length):
    pos = torch.arange(0, length).unsqueeze(1)
    return get_1d_sincos_inblock_pos_embed_from_grid(embed_dim, pos)


#################################################################################
#                                   ACDiT Configs                               #
#################################################################################

def ACDiT_132M(**kwargs):
    return ACDiT(depth=12, hidden_size=768, num_heads=12, **kwargs) # = B

def ACDiT_460M(**kwargs):
    return ACDiT(depth=24, hidden_size=1024, num_heads=16, **kwargs) # =L

def ACDiT_677M(**kwargs):
    return ACDiT(depth=28, hidden_size=1152, num_heads=18, **kwargs) # =XL

def ACDiT_954M(**kwargs):
    return ACDiT(depth=32, hidden_size=1280, num_heads=20, **kwargs) # H

ACDiT_models = {
    'ACDiT-XL': ACDiT_677M,
    'ACDiT-L':  ACDiT_460M,
    'ACDiT-B':  ACDiT_132M,
    'ACDiT-H':  ACDiT_954M
}