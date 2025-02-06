# Poincaré Flow Neural Network: Learning Chaos In A Linear Way

This is the implementation of the paper Poincaré Flow Neural Network: Learning Chaos In A Linear Way.

Learning long-term behaviors in chaotic dynamical systems, such as turbulent flows and climate modelling, is challenging due to their inherent instability and unpredictability. These systems exhibit positive Lyapunov exponents, which significantly hinder accurate long-term forecasting. As a result, understanding long-term statistical behavior is far more valuable than focusing on short-term accuracy. While autoregressive deep sequence models have been applied to capture long-term behavior, they often lead to exponentially increasing errors in learned dynamics. To address this, we shift the focus from simple prediction errors to preserving an invariant measure in dissipative chaotic systems. These systems have attractors, where trajectories settle, and the invariant measure is the probability distribution on attractors that remains unchanged under dynamics. Existing methods generate long trajectories of dissipative chaotic systems by aligning invariant measures, but it is not always possible to obtain invariant measures for arbitrary datasets. We propose the Poincaré Flow Neural Network (PFNN), a novel operator learning framework designed to capture behaviors of chaotic systems without any explicit knowledge on the invariant measure.
PFNN employs an auto-encoder to map the chaotic system to a finite dimensional feature space, effectively linearizing the chaotic evolution.
It then learns the linear evolution operators to match the physical dynamics by addressing two critical properties in dissipative chaotic systems: (1) contraction, the system’s convergence toward its attractors, and (2) measure invariance, trajectories on the attractors following a probability distribution invariant to the dynamics.
Our experiments on a variety of chaotic systems, including Lorenz 96, Kuramoto-Sivashinsky equation and Navier–Stokes equation, demonstrate that PFNN has more accurate predictions and physical statistics compared to competitive baselines including the Fourier Neural Operator and the Markov Neural Operator.

Table of contents
=================
* [Dissipative Chaotic Systems State Forecasting Expresso](#dissipative-chaotic-systems-state-forecasting-expresso)
  * [Lorenz 96 (1D, Dimension 80)](#lorenz-96-1d-dimension-80)
  * [KS (1D, Dimension128)](#ks-1d-dimension128)
  * [Kolmogorov Flow (2D, Dimension 64 $\times$ 64)](#kolmogorov-flow-2d-dimension-64-x-64)


## Dissipative Chaotic Systems State Forecasting Expresso

### Lorenz 96 (1D, Dimension 80)

![1727441482180](image/README/1727441482180.png)

### Kuramoto-Sivashinsky (1D, Dimension128)

![1727441582689](image/README/1727441582689.png)

### Kolmogorov Flow (2D, Dimension 64 $\times$ 64)

Model performance in short-term forecasting accuracy in absolute error with states at step $\{2, 4, 8, 16, 32\}$.

<div style="width: 100%; display: table;">
  <div style="display: table-row;">
    <div style="display: table-cell; text-align: center; width: 34%;">
      <h6 style="font-size: 12px;">Ground Truth</h6>
      <img src="figures/ground_truth.gif" alt="GIF 1" style="width: 100%;">
    </div>
    <div style="display: table-cell; text-align: center; width: 34%;">
      <h6 style="font-size: 12px;">PFNN Prediction</h6>
      <img src="figures/PFNN_prediction.gif" alt="GIF 2" style="width: 100%;">
    </div>
    <div style="display: table-cell; text-align: center; width: 34%;">
      <h6 style="font-size: 12px;">Absolute Error</h6>
      <img src="figures/absolute_error.gif" alt="GIF 3" style="width: 100%;">
    </div>
  </div>
</div>

## Install and dependence

In the beginning, simply try to clone the repository.

Then, create the enviroment for PFNN via

```
conda env create -f environment.yml
conda activate PFNN
```
