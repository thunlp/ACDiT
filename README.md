<h1 align="center"> ACDiT: Interpolating Autoregressive Conditional Modeling and Diffusion Transformer
</h1>

This repository contains the official implementation of **ACDiT**, an innovative model combining the strengths of **autoregressive modeling** and **diffusion transformers**. ACDiT introduces a flexible blockwise generation mechanism, achieving superior performance in both image and video generation tasks. 

<div align="center">
<a href='https://arxiv.org/pdf/2412.07720'><img src='https://img.shields.io/badge/Paper-PDF-orange'></a>
</div>

## Overview

**ACDiT** (**A**utoregressive **C**onditional **Di**ffusion **T**ransformer) interpolates between token-wise autoregressive modeling and full-sequence diffusion by introducing a block-based paradigm. Inherent advantages include:
- Simultaneously learns the causal interdependence across blocks with autoregressive modeling and the non-causal dependence within blocks with diffusion modeling.
- Endowed with clean continuous visual input.
- Makes full use of KV-Cache for flexible autoregressive generation.

<img src="images/Comparison.png" alt="Generation Process of ACDiT" width="600"/>

The generation process of ACDiT, where pixels in each block are denoised simultaneously conditioned on previously generated clean contexts.

### Skip-Causal Attention Mask (SCAM)
| (a) SCAM for training                        | (b) Inference Process                          | (c) 3D view of ACDiT                            |
|---------------------------------------------|-----------------------------------------------|------------------------------------------------|
| ![SCAM Training](images/scam.png)      | ![SCAM Inference](images/inference_process.png)       | ![3D View of ACDiT](images/3dview.png)      |

ACDiT is easy to implement, as simple as adding a Skip-Causal Attention Mask to the current DiT architecture during training, as shown in (a), where each noised block can only attend previous clean blocks and itself. During inference, ACDiT utilizes KV-Cache for efficient autoregressive inference.

## Implementation
To implement the SCAM for both customization and efficiency, we use [FlexAttention](https://pytorch.org/blog/flexattention) provided by [Pytorch 2.5](https://pytorch.org/blog/pytorch2-5/). The training codes will be released soon.

## Model Zoo 🤗
We provide the model weights for ACDiT-XL/H-img/vid through the download links below.
| Model Name          | Image           |    Video  |
|---------------------|-----------------|-----------|
| **ACDiT-XL** | [ACDiT-XL-img](https://huggingface.co/JamesHujy/ACDiT/blob/main/ACDiT-XL-img.pt) |   [ACDiT-XL-vid](https://huggingface.co/JamesHujy/ACDiT/blob/main/ACDiT-XL-vid.pt)      |
| **ACDiT-H**  | [ACDiT-H-img](https://huggingface.co/JamesHujy/ACDiT/blob/main/ACDiT-H-img.pt)  |   [ACDiT-H-vid](https://huggingface.co/JamesHujy/ACDiT/blob/main/ACDiT-H-vid.pt)       |

## Setup
To set up the runtime environment for this project, install the required dependencies using the provided requirements.txt file:
```bash
pip install -r requirements.txt
```

## Sampling
After downloading the checkpoints, you can use the following scripts to generate image or videos:
```bash
python3 sample_img.py --ckpt ACDiT-H-img.pt
```
```bash
python3 sample_vid.py --ckpt ACDiT-H-vid.pt
```

## Evaluation
Following the evaluation protocal of [DiT](https://github.com/facebookresearch/DiT), we use [ADM's evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to compute FID, Inception Score and other metrics. 

## Acknowledgements
This code is mainly built upon [DiT](https://github.com/facebookresearch/DiT) repository.

## License

This project is licensed under the MIT License.

## Citation
If our work assists your research, feel free to give us a star ⭐ or cite us using:
```bibtex
@article{ACDiT,
  title={ACDiT: Interpolating Autoregressive Conditional Modeling and Diffusion Transformer},
  author={Hu, Jinyi and Hu, Shengding and Song, Yuxuan and Huang, Yufei and Wang, Mingxuan and Zhou, Hao and Liu, Zhiyuan and Ma, Wei-Ying and Sun, Maosong},
  journal={arXiv preprint arXiv:2412.07720},
  year={2024}
}
```