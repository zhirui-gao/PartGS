# [ICCV 2025] PartGS

<h2 align="center">

Learning Self-supervised Part-aware  <br> 3D Hybrid Representations
of 2D Gaussians and Superquadrics

[Zhirui Gao](https://scholar.google.com/citations?user=IqtwGzYAAAAJ&hl=zh-CN), [Renjiao Yi](https://renjiaoyi.github.io/), [Huang Yuhang](https://scholar.google.com/citations?user=OAULSygAAAAJ&hl=zh-CN), [Wei Chen](https://openreview.net/profile?id=~Wei_Chen35),  [Chenyang Zhu](https://www.zhuchenyang.net/), [Kai Xu](https://kevinkaixu.net/)

[![arXiv](https://img.shields.io/badge/arXiv-2402.04717-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2408.10789)
[![Project page](https://img.shields.io/badge/Project-Page-brightgreen)](https://arxiv.org/abs/2408.10789)
[![Dataset](https://img.shields.io/badge/HF-Dataset-yellow)](https://arxiv.org/abs/2408.10789)

<p>
    <img width="" alt="romm0" src="./assets/scan63.gif", style="width: 80%;">
</p>

<p align="center">
<span style="font-size: 10 px">
PartGS enables both the block-level and point-
level part-aware reconstructions,  preserving both part decomposition and reconstruction precision.
</span>
</p>


<p>
    <img width="730" alt="pipeline", src="./assets/pipeline.png">
</p>

</h4>

This repository contains the official implementation of the paper: [Self-supervised Learning of Hybrid Part-aware 3D Representation
of 2D Gaussians and Superquadrics](https://arxiv.org/abs/2408.10789), which is accepted to ICCV 2025.
PartGS is a self-supervised part-aware reconstruction framework that integrates 2D Gaussians and superquadrics to
parse objects and scenes into an interpretable decomposition, leveraging multi-view image inputs to uncover 3D structural
information

If you find this repository useful to your research or work, it is really appreciated to star this repository✨ and cite our paper 📚.

Feel free to contact me (gzrer2018@gmail.com) or open an issue if you have any questions or suggestions.


## 🔥 See Also

You may also be interested in our other works:
- [**[ICCV 2025] CurveGaussian**](https://zhirui-gao.github.io/CurveGaussian/):  A novel bi-directional coupling framework between parametric curves and edge-oriented Gaussian components, enabling direct optimization of parametric curves through differentiable Gaussian splatting.

- [**[TCSVT 2025] PoseProbe**](https://github.com/zhirui-gao/PoseProbe): A novel approach of utilizing everyday objects commonly found in both images and real life, as pose probes, to tackle few-view NeRF reconstruction using only 3 to 6 unposed scene images.


- [**[CVMJ 2024] DeepTm**](https://github.com/zhirui-gao/Deep-Template-Matching): An accurate template matching method based on differentiable coarse-to-fine correspondence refinement, especially designed for planar industrial parts.





## 📢 News
- **2025-06-27**: The paper is available on arXiv.
- **2025-06-26**: PartGS is accepted to ICCV 2025.


## 📋 TODO

- [ ] Release the training and evaluation code.


## 🔧 Installation



## 📊 Dataset


## 👀 Visual Results


### DTU Dataset
<p align="center">
    <img width="" alt="40" src="./assets/scan40.gif" style="width: 70%;">
    <img width="" alt="55" src="./assets/scan69.gif", style="width: 70%;">

</p>

### ShapeNet Dataset 

<p align="center">
    <img width="" alt="chair" src="./assets/1a6f615e8b1b5ae4dbbc9440457e303e.gif", style="width: 70%;">
    <img width="" alt="plane" src="./assets/1a32f10b20170883663e90eaf6b4ca52.gif", style="width: 70%;">
</p>


## 🚀 Usage

## 👊 Application


### Editing
<p align="center">
    <img width="" alt="chair" src="./assets/appli.png", style="width: 70%;">
</p>

## Simulation
<p align="center">
    <img width="" alt="chair" src="./assets/physics simulations.gif", style="width: 70%;">
</p>

## 📚 Citation
If you find our work helpful, please consider citing:
```bibtex
@misc{gao2025selfsupervisedlearninghybridpartaware,
      title={Self-supervised Learning of Hybrid Part-aware 3D Representation of 2D Gaussians and Superquadrics}, 
      author={Zhirui Gao and Renjiao Yi and Yuhang Huang and Wei Chen and Chenyang Zhu and Kai Xu},
      year={2025},
      eprint={2408.10789},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.10789}, 
}

```
---

