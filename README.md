# Awesome Deep Learning Watermarking Papers

A curated list of deep learning-based watermarking papers for audio and image, with links to code repositories. Papers with notebook notes are marked with .

---

## Table of Contents

- [Audio Watermarking](#audio-watermarking)
- [Image Watermarking](#image-watermarking)
- [Video Watermarking](#video-watermarking)
- [LLM/Text Watermarking](#llmtext-watermarking)
- [Foundational](#foundational)
- [Surveys & Benchmarks](#surveys--benchmarks)

---

### Audio Watermarking

| Title | Venue | Date | Code | Note | Supplement |
|:------|:-----:|:----:|:----:|:----:|:----------:|
|  [AudioSeal: Proactive Detection of Voice Cloning with Localized Watermarking](https://arxiv.org/abs/2401.17264) (Meta) | arXiv | 2024-01 | [![Star](https://img.shields.io/github/stars/facebookresearch/audioseal.svg?style=social&label=Star)](https://github.com/facebookresearch/audioseal) | [Note](paper/深度学习音频水印/AudioSeal/note.md) / [Experiment](paper/深度学习音频水印/AudioSeal/experiment/) | [![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://facebookresearch.github.io/audioseal/) |
|  [SilentCipher: Deep Audio Watermarking](https://arxiv.org/abs/2406.03822) | arXiv | 2024-06 | [![Star](https://img.shields.io/github/stars/sony/silentcipher.svg?style=social&label=Star)](https://github.com/sony/silentcipher) | [Note](paper/深度学习音频水印/SilentCipher/note.md) | - |
|  [WavMark: Watermarking for Audio Generation](https://arxiv.org/abs/2308.12770) | arXiv | 2023-08 | [![Star](https://img.shields.io/github/stars/wavmark/wavmark.svg?style=social&label=Star)](https://github.com/wavmark/wavmark) | [Note](paper/综述与基准/WavMark/note.md) | - |
| [Self Voice Conversion as an Attack against Neural Audio Watermarking](https://arxiv.org/abs/2601.20432) | arXiv | 2026-01 | - | - | - |
| [Latent-Mark: An Audio Watermark Robust to Neural Resynthesis](https://arxiv.org/abs/2603.05310) | arXiv | 2026-03 | - | - | - |
| [XAttnMark: Learning Robust Audio Watermarking with Cross-Attention](https://arxiv.org/abs/2502.04230) | arXiv | 2025-02 | - | - | [![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://liuyixin-louis.github.io/xattnmark/) |
| [Multi-bit Audio Watermarking](https://arxiv.org/abs/2510.01968) | arXiv | 2025-10 | - | - | - |
| [HarmonicAttack: An Adaptive Cross-Domain Audio Watermark Removal](https://arxiv.org/abs/2511.21577) | arXiv | 2025-11 | - | - | - |
| [WAVE: Watermarking Audio with Key Enrichment](https://arxiv.org/abs/2506.05891) | arXiv | 2025-06 | [![Star](https://img.shields.io/github/stars/thuhcsi/WAKE.svg?style=social&label=Star)](https://github.com/thuhcsi/WAKE) | - | - |
| [P2Mark: Plug-and-play Parameter-level Watermarking for Neural Speech Generation](https://arxiv.org/abs/2504.05197) | arXiv | 2025-04 | - | - | - |
| [A Comprehensive Real-World Assessment of Audio Watermarking Algorithms: Will They Survive Neural Codecs?](https://arxiv.org/abs/2505.19663) | arXiv | 2025-05 | - | - | - |
| [WaveVerify: Audio Watermarking for Media Authentication and Combatting Deepfakes](https://doi.org/10.1109/ijcb65343.2025.11411372) | IEEE IJCB 2025 | 2025 | [![Star](https://img.shields.io/github/stars/pujariaditya/WaveVerify.svg?style=social&label=Star)](https://github.com/pujariaditya/WaveVerify) | - | - |
| [END^2: Robust Dual-Decoder Watermarking Framework Against Non-Differentiable Distortions](https://ojs.aaai.org/index.php/AAAI/article/view/32060) | AAAI 2025 | 2025-02 | - | - | - |
| [Speech Watermarking with Discrete Intermediate Representations](https://ojs.aaai.org/index.php/AAAI/article/view/34600) | AAAI 2025 | 2025-02 | - | - | - |
| [IDEAW: Robust Neural Audio Watermarking with Invertible Dual-Embedding](https://aclanthology.org/2024.emnlp-main.258/) | EMNLP 2024 | 2024-11 | [![Star](https://img.shields.io/github/stars/PecholaL/IDEAW.svg?style=social&label=Star)](https://github.com/PecholaL/IDEAW) | - | - |
| [Latent Watermarking of Audio Generative Models](https://arxiv.org/abs/2409.02915) | arXiv | 2024-09 | - | - | - |
| [Audio Codec Augmentation for Robust Collaborative Watermarking](https://arxiv.org/abs/2409.13382) | arXiv | 2024-09 | [![Star](https://img.shields.io/github/stars/ljuvela/collaborative-watermarking-with-codecs.svg?style=social&label=Star)](https://github.com/ljuvela/collaborative-watermarking-with-codecs) | - | - |
| [WMCodec: End-to-End Neural Speech Codec with Deep Watermarking](https://arxiv.org/abs/2409.12121) | arXiv | 2024-09 | [![Star](https://img.shields.io/github/stars/zjzser/WMCodec.svg?style=social&label=Star)](https://github.com/zjzser/WMCodec) | - | - |
| [AudioMarkBench: Benchmarking Robustness of Audio Watermarking](https://arxiv.org/abs/2406.06979) | arXiv | 2024-06 | [![Star](https://img.shields.io/github/stars/moyangkuo42/AudioMarkBench.svg?style=social&label=Star)](https://github.com/moyangkuo42/AudioMarkBench) | [Note](paper/综述与基准/AudioMarkBench2024/note.md) | - |
| [TraceableSpeech: Proactively Traceable Text-to-Speech with Watermarking](https://arxiv.org/abs/2406.04840) | arXiv | 2024-06 | [![Star](https://img.shields.io/github/stars/zjzser/TraceableSpeech.svg?style=social&label=Star)](https://github.com/zjzser/TraceableSpeech) | - | - |
| [HiFi-GANw: Watermarked Speech Synthesis via Fine-Tuning of HiFi-GAN](https://arxiv.org/abs/2402.10153) | arXiv | 2024-02 | - | - | - |
| [AudioStamp: Deep Learning Based Watermarking for Digital Audio Copyright Protection](https://www.internationaljournalssrg.org/IJECE/2024/Volume11-Issue7/IJECE-V11I7P111.pdf) | IJECE 2024 | 2024 | - | - | - |
| [Detecting Voice Cloning Attacks via Timbre Watermarking](https://arxiv.org/abs/2312.03410) | arXiv | 2023-12 | [![Star](https://img.shields.io/github/stars/TimbreWatermarking/TimbreWatermarking.svg?style=social&label=Star)](https://github.com/TimbreWatermarking/TimbreWatermarking) | - | - |
| [DeAR: A Deep-Learning-Based Audio Re-recording Resilient Watermarking](https://ojs.aaai.org/index.php/AAAI/article/view/26550) | AAAI 2023 | 2023-02 | - | - | - |
| [AudioQR: Deep Neural Audio Watermarks for QR Code](https://www.ijcai.org/proceedings/2023/0687.pdf) | IJCAI 2023 | 2023 | [![Star](https://img.shields.io/github/stars/xinghua-qu/AudioQR.svg?style=social&label=Star)](https://github.com/xinghua-qu/AudioQR) | - | - |
| [Neural Audio Watermarking](https://arxiv.org/abs/2401.17890) | arXiv | 2024 | - | [Note](paper/深度学习音频水印/NeuralAudioWatermarking_2024/note.md) | - |

### Image Watermarking

| Title | Venue | Date | Code | Note | Supplement |
|:------|:-----:|:----:|:----:|:----:|:----------:|
|  [PAI: Plug-and-play Watermarking for AIGC Images](paper/图像水印/PAI/note.md) | arXiv | - | - | [Note](paper/图像水印/PAI/note.md) | - |
| [Stable Signature is Unstable: Removing Image Watermark from Diffusion Models](https://arxiv.org/abs/2405.07145) | arXiv | 2024-05 | - | - | - |
| [The Stable Signature: Rooting Watermarks in Latent Diffusion Models](https://arxiv.org/abs/2303.15435) (Meta) | ICCV 2023 | 2023-03 | [![Star](https://img.shields.io/github/stars/facebookresearch/stable_signature.svg?style=social&label=Star)](https://github.com/facebookresearch/stable_signature) | [Note](paper/图像水印/StableSignature2023/note.md) | - |
| [Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust](https://arxiv.org/abs/2305.20030) | NeurIPS 2023 | 2023-05 | [![Star](https://img.shields.io/github/stars/YuxinWenRick/tree-ring-watermark.svg?style=social&label=Star)](https://github.com/YuxinWenRick/tree-ring-watermark) | [Note](paper/图像水印/TreeRing_Watermarks2023/note.md) | - |
| [Flow-Based Robust Watermarking with Invertible Noise Layer](https://ojs.aaai.org/index.php/AAAI/article/view/25633) | AAAI 2023 | 2023-02 | - | - | - |
| [PIMoG: Screen-shooting Noise-Layer Simulation for Deep-Learning-Based Watermarking](https://dl.acm.org/doi/10.1145/3503161.3548049) | ACM MM 2022 | 2022 | [![Star](https://img.shields.io/github/stars/FangHanNUS/PIMoG-An-Effective-Screen-shooting-Noise-Layer-Simulation-for-Deep-Learning-Based-Watermarking-Netw.svg?style=social&label=Star)](https://github.com/FangHanNUS/PIMoG-An-Effective-Screen-shooting-Noise-Layer-Simulation-for-Deep-Learning-Based-Watermarking-Netw) | - | - |
| [Markpainting: Adversarial Machine Learning Meets Inpainting](https://arxiv.org/abs/2106.04229) | ICML 2021 | 2021-06 | - | - | - |
| [ReDMark: Framework for Residual Diffusion Watermarking on Deep Networks](https://arxiv.org/abs/1810.07248) | arXiv | 2018-10 | - | - | - |

### Foundational

#### Diffusion Model Learning Resources

| Resource | Description |
|:---------|:------------|
| [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) | DDPM 逐行代码讲解（HuggingFace 博客） |

| Title | Venue | Date | Code | Note | Supplement |
|:------|:-----:|:----:|:----:|:----:|:----------:|
| [HiDDeN: Hiding Data with Deep Networks](https://arxiv.org/abs/1807.09937) | ECCV 2018 | 2018-07 | [![Star](https://img.shields.io/github/stars/jirenz/HiDDeN.svg?style=social&label=Star)](https://github.com/jirenz/HiDDeN) | - | - |
| [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (DDPM) | NeurIPS 2020 | 2020-06 | [![Star](https://img.shields.io/github/stars/hojonathanho/diffusion.svg?style=social&label=Star)](https://github.com/hojonathanho/diffusion) | [Note](paper/生成式AI基础/DDPM/note.md) | - |
| [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) (LDM) | CVPR 2022 | 2022-04 | [![Star](https://img.shields.io/github/stars/CompVis/latent-diffusion.svg?style=social&label=Star)](https://github.com/CompVis/latent-diffusion) | [Note](paper/生成式AI基础/LDM/note.md) | - |

### Surveys & Benchmarks

| Title | Venue | Date | Code | Note | Supplement |
|:------|:-----:|:----:|:----:|:----:|:----------:|
|  [SoK: How Robust is Audio Watermarking in Generative AI Models](paper/综述与基准/SoK_AudioWatermarking/note.md) | arXiv | 2024 | - | [Note](paper/综述与基准/SoK_AudioWatermarking/note.md) | [![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://sokaudiowm.github.io/) |
| [SoK: On the Role and Future of AIGC Watermarking in the Era of Gen-AI](https://arxiv.org/abs/2411.11478) | arXiv | 2024-11 | - | - | - |
| [Deep Learning-Based Watermarking Techniques Challenges: A Review of Current and Future Trends](https://link.springer.com/article/10.1007/s00034-024-02651-z) | Circuits, Systems, and Signal Processing | 2024 | - | - | - |
| [Digital Watermarking -- A Meta-Survey and Techniques for Fake News Detection](https://doi.org/10.1109/ACCESS.2024.3373784) | IEEE Access | 2024 | - | - | - |
| [AudioMarkBench: Benchmarking Robustness of Audio Watermarking](https://arxiv.org/abs/2406.06979) | arXiv | 2024-06 | [![Star](https://img.shields.io/github/stars/moyangkuo42/AudioMarkBench.svg?style=social&label=Star)](https://github.com/moyangkuo42/AudioMarkBench) | [Note](paper/综述与基准/AudioMarkBench2024/note.md) | - |
| [Survey: Deep Learning-Based Audio Watermarking Techniques](https://link.springer.com/article/10.1007/s00034-024-02651-z) | Survey | 2023 | - | [Note](paper/综述与基准/Survey_DL_Audio_Watermarking/note.md) | - |

### Video Watermarking

| Title | Venue | Date | Code | Note | Supplement |
|:------|:-----:|:----:|:----:|:----:|:----------:|
| [VideoSeal](https://arxiv.org/abs/2502.08725) (Meta) | arXiv | 2025-02 | - | [Note](paper/视频水印/VideoSeal2025/note.md) | - |

### LLM/Text Watermarking

| Title | Venue | Date | Code | Note | Supplement |
|:------|:-----:|:----:|:----:|:----:|:----------:|
| [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226) | ICML 2023 | 2023-01 | [![Star](https://img.shields.io/github/stars/jwkirchenbauer/lm-watermarking.svg?style=social&label=Star)](https://github.com/jwkirchenbauer/lm-watermarking) | [Note](paper/LLM水印/Kirchenbauer2023_LLM_Watermark/note.md) | - |
| [Adaptive Text Watermark](https://arxiv.org/abs/2312.02249) | arXiv | 2024 | - | [Note](paper/LLM水印/Lai2024_Adaptive_Text_Watermark/note.md) | - |
| [On the Reliability of Watermarks for Large Language Models](https://arxiv.org/abs/2306.04634) | ICLR 2024 | 2024 | - | [Note](paper/LLM水印/Reliability_LLM_Watermarks/note.md) | - |
| [Undetectable Watermarks for Language Models](https://arxiv.org/abs/2306.09194) | FOCS 2024 | 2024 | - | [Note](paper/LLM水印/Christ2024_Undetectable_Watermarks/note.md) | - |
| [Distortion-Free Watermarks Are Not Possible](https://arxiv.org/abs/2311.04399) | arXiv | 2024 | - | [Note](paper/LLM水印/Goldwasser2024_Distortion_Free_Impossible/note.md) | - |

### Backdoor Watermarking

| Title | Venue | Date | Code | Note | Supplement |
|:------|:-----:|:----:|:----:|:----:|:----------:|
|  [Watermark: Clean-label Backdoor Watermarking for Audio Dataset Protection](paper/后门水印/Watermark_Backdoor_WM/note.md) | - | - | - | [Note](paper/后门水印/Watermark_Backdoor_WM/note.md) | - |

---

>  = has notebook note in [paper/](paper/) directory
