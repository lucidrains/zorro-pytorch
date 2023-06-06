<img src="./zorro.png" width="450px"></img>

## Zorro - Pytorch

Implementation of <a href="https://arxiv.org/abs/2301.09595">Zorro</a>, Masked Multimodal Transformer, in Pytorch. This is a Deepmind work that claims a special masking strategy within a transformer help them achieve SOTA on a few multimodal benchmarks.

## Appreciation

- <a href="https://stability.ai/">Stability.ai</a> for the generous sponsorship to work and open source cutting edge artificial intelligence research

## Install

```bash
$ pip install zorro-pytorch
```

## Usage

```python
import torch
from zorro_pytorch import Zorro, TokenTypes as T

model = Zorro(
    dim = 512,                        # model dimensions
    depth = 6,                        # depth
    dim_head = 64,                    # attention dimension heads
    heads = 8,                        # attention heads
    ff_mult = 4,                      # feedforward multiple
    num_fusion_tokens = 16,           # number of fusion tokens
    audio_patch_size = 16,            # audio patch size, can also be Tuple[int, int]
    video_patch_size = 16,            # video patch size, can also be Tuple[int, int]
    video_temporal_patch_size = 2,    # video temporal patch size
    video_channels = 3,               # video channels
    return_token_types = (
        T.AUDIO,
        T.AUDIO,
        T.FUSION,
        T.GLOBAL,
        T.VIDEO,
        T.VIDEO,
        T.VIDEO,
    ) # say you want to return 2 tokens for audio, 1 token for fusion, 3 for video - for whatever self-supervised learning, supervised learning, etc etc
)

video = torch.randn(2, 3, 8, 32, 32) # (batch, channels, time, height, width)
audio = torch.randn(2, 1024 * 10)    # (batch, time)

return_tokens = model(audio = audio, video = video) # (2, 6, 512) - all 6 tokes as indicated above is returned

# say you only want 1 audio and 1 video token, for contrastive learning

audio_token, video_token = model(audio = audio, video = video, return_token_indices = (0, 3)).unbind(dim = -2) # (2, 512), (2, 512)

```

## Citations

```bibtex
@inproceedings{Recasens2023ZorroTM,
  title  = {Zorro: the masked multimodal transformer},
  author = {Adri{\`a} Recasens and Jason Lin and Jo{\~a}o Carreira and Drew Jaegle and Luyu Wang and Jean-Baptiste Alayrac and Pauline Luc and Antoine Miech and Lucas Smaira and Ross Hemsley and Andrew Zisserman},
  year   = {2023}
}
```
