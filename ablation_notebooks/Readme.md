# Notebooks for ablation study

Dear PFNN readers,

There are several notebooks in this folder for ablation study on 

    - the **Log-Sobolev constant** , **Feature dimension** and **Regularized coefficient** effects on PFNN models learning the Kuramoto–Sivashinsky equation.

If you are interested in the results of the ablation study data, please browse over the notebooks to find the *final box* of results.

If you wish to try the ablation results by yourself (such as changing seeds, trying different data amounts etc.), here are **the preparation steps** for you on a smooth run:

1. create an environment for PFNN (at least refer to the environment.yml and install all)
2. download the PFNN repository to your working folder
3. generate the data for the Kuramoto–Sivashinsky equation (refer to the ../lake/data_generation/KS folder, which needs a working matlab software, and generate **110-150k steps** (or 2k-step trajectories over 50-70 initial states)  data ideally and it shouldn't be very time-lengthy as it is a 1D case)
4. load data and models properly in the notebooks ([link to model weights](https://filebin.net/u7b41dqpn0jvz4sc))
5. ready to go!

Many thanks for your interest in PFNN!
