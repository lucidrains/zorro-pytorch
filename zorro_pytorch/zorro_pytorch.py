from enum import Enum
import functools
from functools import wraps

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from beartype import beartype
from beartype.typing import Tuple, Optional, Union

from torchaudio.transforms import Spectrogram

# constants

class TokenTypes(Enum):
    AUDIO = 0
    VIDEO = 1
    FUSION = 2

# functions

def exists(val):
    return val is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def round_down_nearest_multiple(n, divisor):
    return n // divisor * divisor

def pair(t):
    return (t, t) if not isinstance(t, tuple) else t

def cum_mul(it):
    return functools.reduce(lambda x, y: x * y, it, 1)

def divisible_by(numer, denom):
    return (numer % denom) == 0

# decorators

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

# bias-less layernorm

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# geglu feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        GEGLU(),
        nn.Linear(inner_dim, dim, bias = False)
    )

# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        context = None,
        attn_mask = None
    ):
        x = self.norm(x)
        kv_x = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(kv_x).chunk(2, dim = -1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        if exists(attn_mask):
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# main class

class Zorro(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        num_fusion_tokens = 16,
        audio_patch_size: Union[int, Tuple[int, int]] = 16,
        video_patch_size: Union[int, Tuple[int, int]] = 16,
        video_temporal_patch_size = 2,
        video_channels = 3,
        spec_n_fft = 128,
        spec_power = 2,
        spec_win_length = 24,
        spec_hop_length = None,
        spec_pad = 0,
        spec_center = True,
        spec_pad_mode = 'reflect',
        spec_aug_stretch_factor = 0.8,
        spec_aug_freq_mask = 80,
        spec_aug_time_mask = 80,
        return_token_types: Tuple[TokenTypes] = (TokenTypes.AUDIO, TokenTypes.VIDEO, TokenTypes.FUSION)
    ):
        super().__init__()
        self.max_return_tokens = len(return_token_types)

        self.return_token_types = return_token_types
        return_token_types_tensor = torch.tensor(list(map(lambda t: t.value, return_token_types)))
        self.register_buffer('return_token_types_tensor', return_token_types_tensor, persistent = False)

        self.return_tokens = nn.Parameter(torch.randn(self.max_return_tokens, dim))
        self.attn_pool = Attention(dim = dim, dim_head = dim_head, heads = heads)

        # audio input

        self.audio_patch_size = audio_patch_height, audio_patch_width = pair(audio_patch_size)

        self.spec = Spectrogram(
            n_fft = spec_n_fft,
            power = spec_power,
            win_length = spec_win_length,
            hop_length = spec_hop_length,
            pad = spec_pad,
            center = spec_center,
            pad_mode = spec_pad_mode
        )

        audio_input_dim = cum_mul(self.audio_patch_size)
        self.audio_to_tokens = nn.Sequential(
            Rearrange('b (h p1) (w p2) -> b h w (p1 p2)', p1 = audio_patch_height, p2 = audio_patch_width),
            nn.LayerNorm(audio_input_dim),
            nn.Linear(audio_input_dim, dim),
            nn.LayerNorm(dim)
        )

        # video input

        self.video_patch_size = (video_temporal_patch_size, *pair(video_patch_size))

        video_input_dim = cum_mul(self.video_patch_size) * video_channels
        video_patch_time, video_patch_height, video_patch_width = self.video_patch_size

        self.video_to_tokens = nn.Sequential(
            Rearrange('b c (t p1) (h p2) (w p3) -> b t h w (c p1 p2 p3)', p1 = video_patch_time, p2 = video_patch_height, p3 = video_patch_width),
            nn.LayerNorm(video_input_dim),
            nn.Linear(video_input_dim, dim),
            nn.LayerNorm(dim)
        )

        # fusion tokens

        self.fusion_tokens = nn.Parameter(torch.randn(num_fusion_tokens, dim))

        # transformer

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = LayerNorm(dim)

    def forward(
        self,
        *,
        audio,
        video,
        return_token_indices: Optional[Tuple[int]] = None
    ):
        batch, device = audio.shape[0], audio.device
    
        # validate video can be patched

        assert all([divisible_by(numer, denom) for denom, numer in zip(self.video_patch_size, tuple(video.shape[-3:]))]), f'video shape {video.shape[-3:]} needs to be divisible by {self.video_patch_size}'

        # automatically crop if audio does not yield a 2d spectrogram that is divisible by patch sizes

        audio = self.spec(audio)

        height, width = audio.shape[-2:]
        patch_height, patch_width = self.audio_patch_size

        rounded_height, rounded_width = map(lambda args: round_down_nearest_multiple(*args), ((height, patch_height), (width, patch_width)))

        if (height, width) != (rounded_height, rounded_width): # just keep printing to be annoying until it is fixed
            print_once(f'spectrogram yielded shape of {(height, width)}, but had to be cropped to {(rounded_height, rounded_width)} to be patchified for transformer')

        audio = audio[..., :rounded_height, :rounded_width]

        # to tokens

        audio_tokens = self.audio_to_tokens(audio)

        video_tokens = self.video_to_tokens(video)

        fusion_tokens = repeat(self.fusion_tokens, 'n d -> b n d', b = batch)

        # construct all tokens

        audio_tokens, fusion_tokens, video_tokens = map(lambda t: rearrange(t, 'b ... d -> b (...) d'), (audio_tokens, fusion_tokens, video_tokens))

        tokens, ps = pack((
            audio_tokens,
            fusion_tokens,
            video_tokens
        ), 'b * d')

        # construct mask (thus zorro)

        token_types = torch.tensor(list((
            *((TokenTypes.AUDIO.value,) * audio_tokens.shape[-2]),
            *((TokenTypes.FUSION.value,) * fusion_tokens.shape[-2]),
            *((TokenTypes.VIDEO.value,) * video_tokens.shape[-2]),
        )), device = device, dtype = torch.long)

        token_types_attend_from = rearrange(token_types, 'i -> i 1')
        token_types_attend_to = rearrange(token_types, 'j -> 1 j')

        # the logic goes
        # every modality, including fusion can attend to self

        zorro_mask = token_types_attend_from == token_types_attend_to

        # fusion can attend to everything

        zorro_mask = zorro_mask | token_types_attend_from == TokenTypes.FUSION.value

        # and both specific modalities like audio and video can attend to fusion

        zorro_mask = zorro_mask | token_types_attend_to == TokenTypes.FUSION.value

        # attend and feedforward

        for attn, ff in self.layers:
            tokens = attn(tokens, attn_mask = zorro_mask) + tokens
            tokens = ff(tokens) + tokens

        tokens = self.norm(tokens)

        # final attention pooling - each modality pool token can only attend to its own tokens

        return_tokens = self.return_tokens
        return_token_types_tensor = self.return_token_types_tensor

        if exists(return_token_indices):
            assert len(set(return_token_indices)) == len(return_token_indices), 'all indices must be unique'
            assert all([indice < self.max_return_tokens for indice in return_token_indices]), 'indices must range from 0 to max_num_return_tokens - 1'

            return_token_indices = torch.tensor(return_token_indices, dtype = torch.long, device = device)

            return_token_types_tensor = return_token_types_tensor[return_token_indices]
            return_tokens = return_tokens[return_token_indices]

        return_tokens = repeat(return_tokens, 'n d -> b n d', b = batch)
        pool_mask = rearrange(return_token_types_tensor, 'i -> i 1') == token_types_attend_to

        pooled_tokens = self.attn_pool(return_tokens, context = tokens, attn_mask = pool_mask) + return_tokens

        return pooled_tokens
