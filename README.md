# Time-domain-seismic-wave-modeling-using-PINN
This repository gives the codes for numerical solver independent seismic wave simulation using Task-decomposed Physics-informed Neural Networks.
**This repository reproduces the results of the paper "[Numerical solver independent seismic wave simulation using Task-decomposed Physics-informed Neural Networks.](https://ieeexplore.ieee.org/abstract/document/10229173)" IEEE Geoscience and Remote Sensing Letters 20, 3002905.

# Overview

We propose a task-decomposed (TD) training scheme of Physics-informed Neural Network (PINN) to perform the time-domain wave equation modeling. Besides this feature, we use analytical wavefield solutions (Fourier transformed from frequency-domain ones) as the initial condition to avoid the source singularity issue. 

Figure (a) shows the wavefield from the TD-PINN, and Figure (b) shows the wavefield from the finite-difference method (FDM).
![TDPINN_wavefie_gif](https://github.com/songc0a/Time-domain-seismic-wave-modeling-using-PINN/assets/31889731/a6a7bd80-e95c-4627-83e5-77b4371a7c8b)

# Code explanation

pre-training.ipynb: Stage1-pretraining

train_full.ipynb: Stage2-Full training

tra_physics.ipynb: Stage3-Physics enhanced training

# Citation information

If you find our codes and publications helpful, please kindly cite the following publications.

@article{zou2023numerical,
  title={Numerical solver independent seismic wave simulation using Task-decomposed Physics-informed Neural Networks},
  author={Zou, Jingbo and Liu, Cai and Song, Chao and Zhao, Pengfei},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2023},
  publisher={IEEE}
}

# contact information
If there are any problems, please contact me through my emails: chao.song@kaust.edu.sa;chaosong@jlu.edu.cn
