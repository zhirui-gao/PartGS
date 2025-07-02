# [ICCV 2025] PartGS

<h4 align="center">

Learning Self-supervised Part-aware  <br> 3D Hybrid Representations
of 2D Gaussians and Superquadrics

[Zhirui Gao](https://scholar.google.com/citations?user=IqtwGzYAAAAJ&hl=zh-CN), [Renjiao Yi](https://renjiaoyi.github.io/), [Huang Yuhang](https://scholar.google.com/citations?user=OAULSygAAAAJ&hl=zh-CN), [Wei Chen](https://openreview.net/profile?id=~Wei_Chen35),  [Chenyang Zhu](https://www.zhuchenyang.net/), [Kai Xu](https://kevinkaixu.net/)

[![arXiv](https://img.shields.io/badge/arXiv-2402.04717-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2408.10789)
[![Project page](https://img.shields.io/badge/Project-Page-brightgreen)](https://arxiv.org/abs/2506.21401)
[![Dataset](https://img.shields.io/badge/HF-Dataset-yellow)](https://arxiv.org/abs/2506.21401)

<p>
    <img width="" alt="romm0" src="./assets/room0.gif", style="width: 80%;">
</p>

<p align="center">
CurveGaussian enables compact 3D parametric <br>curve reconstruction from multi-view 2D edge maps.  
</p>


<p>
    <img width="730" alt="pipeline", src="./assets/pipeline.png">
</p>

</h4>

This repository contains the official implementation of the paper: [Learning Self-supervised Part-aware 3D Hybrid Representations
of 2D Gaussians and Superquadrics](https://arxiv.org/abs/2408.10789), which is accepted to ICCV 2025.
**PartGS** is a self-supervised **part**-aware reconstruction frame-
work that integrates 2D **G**aussians and **s**uperquadrics to
parse objects and scenes into an interpretable decomposition,
leveraging multi-view image inputs to uncover 3D structural
information.

If you find this repository useful to your research or work, it is really appreciated to star this repository✨ and cite our paper 📚.

Feel free to contact me (gzrer2018@gmail.com) or open an issue if you have any questions or suggestions.


## 🔥 See Also

You may also be interested in our other works:
- [**[ICCV 2025] CurveGaussian**](https://github.com/zhirui-gao/Curve-Gaussian):  A novel bi-directional coupling framework between parametric curves and edge-oriented Gaussian components, enabling direct optimization of parametric
curves through differentiable Gaussian splatting.

- [**[TCSVT 2025] PoseProbe**](https://github.com/zhirui-gao/PoseProbe): A novel approach of utilizing everyday objects commonly found in both images and real life, as pose probes, to tackle few-view NeRF reconstruction using only 3 to 6 unposed scene images.


- [**[CVMJ 2024] DeepTm**](https://github.com/zhirui-gao/Deep-Template-Matching): An accurate template matching method based on differentiable coarse-to-fine correspondence refinement, especially designed for planar industrial parts.





## 📢 News
- **2025-06-30**: The updated paper is available on arXiv.
- **2025-06-26**: PaperGS is accepted to ICCV 2025.


## 📋 TODO

- [ ] Release the training and evaluation code.


## 🔧 Installation



## 📊 Dataset


## 👀 Visual Results

DTU Dataset

<p align="center">
    <img width="" alt="romm0" src="./assets/room1.gif", style="width: 45%;">
    <img width="" alt="romm0" src="./assets/room2.gif", style="width: 45%;">
</p>


BlendedMVS Dataset 
<div style="text-align: center;">
  <video width="100%" autoplay loop muted>
    <source src="./assets/00000952_comparison.mp4" type="video/mp4">
  </video>

</div>

<div style="text-align: center;">
  <video width="100%" autoplay loop muted>
    <source src="./assets/00007025_comparison.mp4" type="video/mp4">
  </video>
</div>


## 🚀 Usage



## 📚 Citation
If you find our work helpful, please consider citing:
```bibtex
@misc{yi2025curveawaregaussiansplatting3d,
      title={Curve-Aware Gaussian Splatting for 3D Parametric Curve Reconstruction}, 
      author={Zhirui Gao. Renjiao Yi and Yaqiao Dai and Xuening Zhu and Wei Chen and Chenyang Zhu and Kai Xu},
      year={2025},
      eprint={2506.21401},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.21401}, 
}
```
