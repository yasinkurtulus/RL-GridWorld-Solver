# Reinforcement Learning: GridWorld Solver 🤖🧠

![Python](https://img.shields.io/badge/Python-3.6%2B-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Data%20Science-013243?logo=numpy&logoColor=white)
![RL](https://img.shields.io/badge/Domain-Reinforcement%20Learning-FF6F00)

This repository contains a custom **GridWorld Environment** and implements foundational Dynamic Programming algorithms to solve Markov Decision Processes (MDPs). 

It is designed to demonstrate how an AI agent can learn optimal paths in a deterministic grid environment using **Value Iteration** and **Policy Iteration** algorithms.

## 🚀 Features

- **Custom GridWorld Environment (`GridWorldEnv`)**: A highly modular 7x7 grid environment where rules, start/end states, ditches, and reward/penalty metrics can be customized.
- **Value Iteration**: An algorithm that computes the optimal state-value function iteratively to extract the perfect policy.
- **Policy Iteration**: An algorithm that starts with a random policy and continuously evaluates and improves it until convergence.
- **Gym-like API**: The environment uses standard reinforcement learning structures (`reset()`, `step()`, `compute_reward()`).

## 🧠 Algorithms Overview

### 1. Value Iteration
Value iteration focuses on finding the optimal value function first. It iteratively updates the value of each state using the **Bellman Optimality Equation**. Once the values converge (stop changing), it extracts the optimal policy in a single sweep.

### 2. Policy Iteration
Policy iteration focuses on the policy itself. It consists of two main loops:
- **Policy Evaluation:** Calculates the value of the grid strictly following the current policy.
- **Policy Improvement:** Greedily updates the policy based on the newly calculated values.
This dance continues until the policy becomes mathematically optimal and stops changing.

## 💻 Installation & Usage

1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
   cd your-repo-name