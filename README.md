# Time-domain-seismic-wave-modeling-using-PINN
This repository gives the codes for numerical solver independent seismic wave simulation using Task-decomposed Physics-informed Neural Networks.
**This repository reproduces the results of the paper "[Numerical solver independent seismic wave simulation using Task-decomposed Physics-informed Neural Networks.](https://ieeexplore.ieee.org/abstract/document/10229173)" IEEE Geoscience and Remote Sensing Letters 20, 3002905.

# Overview

We propose a task-decomposed (TD) training scheme of Physics-informed Neural Network (PINN) to perform the time-domain wave equation modeling. Besides this feature, we use analytical wavefield solutions (Fourier transformed from frequency-domain ones) as the initial condition to avoid the source singularity issue. 

Figure a shows the wavefield from the TD-PINN, and Figure b shows the wavefield from the finite-difference method (FDM).
![wavefield_com](https://github.com/songc0a/Time-domain-seismic-wave-modeling-using-PINN/assets/31889731/647347b2-de24-4497-b7df-1e9fcb6fa288)



# Code explanation

helm_solver_ffpinn_4D.py: Tensorflow code for solving the multifrequency-multisource scattered wavefields using Fourier feature PINN  
helm_solver_ffpinn_4D_test.py: Tensorflow code for a new velocity using the saved model
Sigsbee_sourceinput_data_generation_fre.m: Matlab code for generating training and test data  

# Citation information

If you find our codes and publications helpful, please kindly cite the following publications.

@article{song2023simulating,
  title={Simulating seismic multifrequency wavefields with the Fourier feature physics-informed neural network},
  author={Song, Chao and Wang, Yanghua},
  journal={Geophysical Journal International},
  volume={232},
  number={3},
  pages={1503--1514},
  year={2023},
  publisher={Oxford University Press}
}

# contact information
If there are any problems, please contact me through my emails: chao.song@kaust.edu.sa;chaosong@jlu.edu.cn
