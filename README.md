# Kolmogorov-Arnold Networks for Q-Learning

This repository presents a comparative study on the generalizability of Kolmogorov-Arnold Networks (KAN), Multi-Layer Perceptrons (MLP), and Long Short-Term Memory networks (LSTM) in Q-learning for reinforcement learning environments.

## Goal

The primary goal of this project is to evaluate and compare the generalization abilities of KAN with MLP, Kan-Based LSTM with LSTM in various Q-learning environments, focusing on their performance under in-distribution and out-of-distribution conditions.

## Methodology

The study was conducted using three reinforcement learning algorithms:
- **DQN (Deep Q Network)**
- **ADRQN (Action Specific Deep Recurrent Q Network)**
- **PPO (Proximal Policy Optimization)**

### Experimental Setup

- **Default Environment (In-Distribution):** CartPole environment with the following parameters:
  - Gravity: 9.8
  - Cart Mass: 1.0
  - Pole Mass: 0.1
  - Pole Length: 0.5
  - Force: 10.0
  - Time Step: 0.02
- **Out-of-Distribution Environment:** The environment parameters were altered by ±20% for generalization testing.

### Network Architectures

1. **Multi-Layer Perceptron (MLP)**
2. **Kolmogorov-Arnold Network (KAN)**
3. **LSTM and KAN-based LSTM**

## Experiments and Results

- **KAN vs. MLP:** KAN outperforms MLP in the default CartPole environment using DQN.
- **Environment Adaptability:** KAN and MLP show similar performance when environment parameters are changed by 20%.
- **KAN-LSTM vs. LSTM (ADRQN):** KAN-LSTM surpasses LSTM for CartPole under ADRQN.
- **KAN-LSTM vs. LSTM (PPO):** KAN-LSTM performs worse than LSTM under PPO.

Model performance was evaluated at regular intervals:
- Every 20 of 10,000 episodes for DQN
- Every 5 of 500 episodes for ADRQN

### Visual Results

The results are visualized across several metrics, demonstrating the effectiveness of each model in both in-distribution and out-of-distribution scenarios:
1. **Evaluation of MLP vs. KAN for Default Environment using DQN**
2. **Evaluation of MLP vs. KAN for Changing Environment using DQN**
3. **Evaluation of LSTM vs. KAN-based LSTM using ADRQN**
4. **Evaluation of LSTM vs. KAN-based LSTM using PPO**

## Future Work and Conclusion

The study shows that KAN networks outperform MLPs in generalization for the default CartPole environment when using DQN, while performing comparably under changing parameters. KAN-LSTM models show strong performance with ADRQN, though they underperform compared to LSTM with PPO. Future research could explore KAN-LSTM models with different algorithms and environments to further assess their generalization capabilities.

## References

1. Xu, Z., Wang, Y., Wang, S., Ruehle, F., Hallenborg, J., Stolpe, M., Hsu, T.Y., & Tegmark, M. (2023). KAN: Kolmogorov-Arnold Networks. arXiv:2306.11009.
2. Rivière, K. (2022). Kolmogorov-Arnold Networks as Radial Basis Function Networks. arXiv:abs/2306.02972.
3. Hausknecht, M., & Stone, P. (2015). Deep Recurrent Q-Learning for Partially Observable MDPs. arXiv:1507.06527.
4. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv:abs/1707.06347.


Recommended virtual environments: `conda` or `virtualenv`.

Activate your virtual environment and install dependencies
```[bash]
pip install --upgrade pip==21 setuptools==65.5.0 wheel==0.38.0 # Needed to install gym==0.21.0
pip install swig # Needs to be installed before requirements
pip install -r requirements.txt
```
