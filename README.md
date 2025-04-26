# HedgeGPT
Unlock next-generation option pricing and hedging with an AI-powered reinforcement-learning framework built on the Black–Scholes model

## 1) What Exactly Is HedgeGPT?
HedgeGPT is a sophisticated trading and risk-management engine that leverages Q-learning to price and dynamically hedge European options. Rather than relying purely on closed-form formulas, HedgeGPT uses large-scale Monte Carlo simulation of underlying asset paths, encodes the market state with B-spline basis functions, and trains a Q-learner to discover optimal hedging strategies in a simulated Black–Scholes world. The result is an AI agent that learns to replicate option payoffs and manage risk in a fully automated, data-driven fashion.

## 2) Key Features

**2.1) Monte Carlo Backbone**

* Simulates tens of thousands of asset‐price paths under risk‐neutral dynamics to capture path dependency and nonlinear payoffs in a single, unified framework.

**2.2) B-Spline State Encoding**

* Transforms log-prices into a compact, continuous feature space, enabling smooth, high-resolution representation of market states.

**2.3) Q-Learning Hedging Agent**

* Learns optimal hedge ratios by maximizing a variance-based reward, striking the right balance between hedging P&L and portfolio variance.

**2.4) Variance-Based Reward Function**

* Penalizes portfolio volatility while rewarding accurate payoff replication, adapting dynamically to changing market conditions.

**2.5) Transparent Black–Scholes Benchmark**

* Directly compares the agent’s learned option price to the analytic Black–Scholes price, providing an intuitive gauge of model performance.

## 3) Why Use HedgeGPT?
Whether you’re a quant researcher, risk manager, or developer building automated strategies, HedgeGPT provides:

* Deep AI Integration: Combines reinforcement learning with classical finance theory.

* Risk-Aware Pricing: Explicit variance control yields robust option valuations.

* Extensibility: Swap in different reward functions, market simulators, or basis encodings.

* Educational Value: See firsthand how an AI agent discovers the famed delta‐hedging strategy.

 ## 4) How to Run the Project
Follow these steps to run the project:

**4.1) Clone the Repository**

```python
git clone https://github.com/YourUsername/HedgeGPT.git
```

**4.2) Navigate to the Project Directory**

```python
cd HedgeGPT
```

**4.3) Install the Required Packages**

```python
pip install -r requirements.txt
```

**4.4) Run the Script**

```python
python main.py
```

## 5) Core Formulas and Computations

Below is a detailed rundown of every mathematical formula used in main.py, organized by the five main computation stages

**5.1) Asset‐Price Simulation (Geometric Brownian Motion)**

![Asset‐Price Simulation (Geometric Brownian Motion)](Geometric_Brownian_Motion.png)

