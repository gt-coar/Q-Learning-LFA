# Q-Learning-LFA

This repository contains the source code to reproduce all the numerical experiments as described in the paper ["Finite-Sample Analysis of Nonlinear Stochastic Approximation with Applications in Reinforcement Learning"](https://arxiv.org/pdf/1905.11425.pdf).

Here's a BibTeX entry that you can use to cite it in a publication:
```bibtex
@article{chen2019finite,
  title={Finite-sample analysis of nonlinear stochastic approximation with applications in reinforcement learning},
  author={Chen, Zaiwei and Zhang, Sheng and Doan, Thinh T and Clarke, John-Paul and Theja Maguluri, Siva},
  journal={arXiv e-prints},
  pages={arXiv--1905},
  year={2019}
}
```

# Requirements
* Python (>= 3.7)
* Numpy (>= 1.19.1)

# Usage
## Constant Step Size
1. Show convergence of Q-learning with linear function approximation for <img src="https://render.githubusercontent.com/render/math?math=\gamma \in \{0.7, 0.9, 0.97\}">.
```
cd constant_step_size
python convergence.py
```

2. Show exponentially fast convergence of Q-learning with linear function approximation for <img src="https://render.githubusercontent.com/render/math?math=\gamma = 0.7">.
```
cd constant_step_size
python rate_of_convergence.py
```

## Diminishing Step Sizes
Show convergence rate of Q-learning with linear function approximation for using diminishing step sizes <img src="https://render.githubusercontent.com/render/math?math=\xi \in \{0.4, 0.6, 0.8, 1.0\}">.
```
cd diminishing_step_size
python rate_of_convergence.py
```

# Maintainers
* [Sheng Zhang](https://github.com/xiaojianzhang) - shengzhang@gatech.edu
