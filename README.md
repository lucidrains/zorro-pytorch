<img src="./zorro.png" width="450px"></img>

## Zorro - Pytorch (wip)

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
from zorro_pytorch import Zorro

model = Zorro(
    dim = 512,
    depth = 6
)

video = torch.randn(2, 3, 8, 32, 32)
audio = torch.randn(2, 1024 * 10)

return_tokens = model(audio = audio, video = video) # (2, 3, 512) - 1 audio, 1 video, 1 fusion - but customizable
```

## Citations

```bibtex
@inproceedings{Recasens2023ZorroTM,
  title  = {Zorro: the masked multimodal transformer},
  author = {Adri{\`a} Recasens and Jason Lin and Jo{\~a}o Carreira and Drew Jaegle and Luyu Wang and Jean-Baptiste Alayrac and Pauline Luc and Antoine Miech and Lucas Smaira and Ross Hemsley and Andrew Zisserman},
  year   = {2023}
}
```
