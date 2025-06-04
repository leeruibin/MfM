<!-- ![MfM-logo](assets/MfM_logo.png) -->
<div align="center">
  <img src="assets/MfM_logo.jpeg" alt="MfM-logo" width="50%">
</div>
<div align="center">



<a href="https://leeruibin.github.io/MfMPage/"><img src="https://img.shields.io/static/v1?label=Project&message=Page&color=blue&logo=github-pages"></a> &ensp;
<a href="https://arxiv.org/abs/2506.01758"><img alt="paper" src="https://img.shields.io/badge/arXiv-2506.01758-b31b1b.svg"></a> &ensp;
<a href="https://github.com/SandAI-org/MAGI-1/LICENSE"><img alt="license" src="https://img.shields.io/badge/License-Apache2.0-green?logo=Apache"></a> &ensp;
# Many-for-Many: Unify the Training of Multiple Video and Image Generation and Manipulation Tasks
</div>



> Tao Yang<sup>1</sup>, Ruibin Li<sup>1,2</sup>, Yangming Shi<sup>1</sup>, Yuqi Zhang<sup>1</sup>, Qide Dong<sup>1</sup>, Haoran Cheng<sup>1</sup>, Weiguo Feng<sup>1</sup>, Shilei Wen<sup>1</sup>, Bingyue Peng<sup>1</sup>, Lei Zhang<sup>2</sup>
> 
> <sup>1</sup>ByteDance, <sup>2</sup>The Hong Kong Polytechnic University, 
> 
> In this work, we introduce a unified framework, namely **many-for-many**, which leverages the available training data from many different visual generation and manipulation tasks to train a single model for those different tasks. Specifically, we design a lightweight adapter to unify the different conditions in different tasks, then employ a joint image-video learning strategy to progressively train the model from scratch. Our joint learning leads to a unified visual generation and manipulation model with improved video generation performance. In addition, we introduce depth maps as a condition to help our model better perceive the 3D space in visual generation. Two versions of our model are trained with different model sizes (8B and 2B), each of which can perform more than 10 different tasks. In particular, our 8B model demonstrates highly competitive performance in video generation tasks compared to open-source and even commercial engines. 🚀✨

<img src='./assets/visual_result.png'>

<!-- This repository contains the code for the MfM model, pre-trained weights and inference code. You can find more information on our paper <a href="https://arxiv.org/abs/2506.01758"><img alt="paper" src="https://img.shields.io/badge/arXiv-2506.01758-b31b1b.svg"></a> . 🚀✨ -->

## 📺 Demo Video
<div align="center">
  <video src="https://github.com/user-attachments/assets/f1ddd1fd-1c2b-44e7-94dc-9f62963ab147" width="70%" poster=""> </video>
</div>

## 📮 Architecture

<img src='./assets/arch.png'>

  <!-- Option 2: If you have a GIF version -->
  <!-- <img src="./assets/MfM_demo.gif" alt="MfM Demo" width="70%"> -->
<!-- </div> -->

## 🔥 Latest News

- Inference code and model weights, coming soon.

## 📌 Progress Checklist
<!-- ✅ -->

- [x] **⬜️ Inference Code**  
- [x] **⬜️ Model Weights**

<!-- ## Introduction







## Architecture -->



## ✍️ Citation

If you find our code or model useful in your research, please cite:

```bibtex
@article{yang2025MfM,
  title={Many-for-Many: Unify the Training of Multiple Video and Image Generation and Manipulation Tasks},
  author={Tao Yang, Ruibin Li, Yangming Shi, Yuqi Zhang, Qide Dong, Haoran Cheng, Weiguo Feng, Shilei Wen, Bingyue Peng, Lei Zhang},
  year={2025},
  booktitle={arXiv preprint arXiv:2506.01758},
}
```