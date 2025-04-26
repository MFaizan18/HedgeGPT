# HedgeGPT
Unlock next-generation option pricing and hedging with an AI-powered reinforcement-learning framework built on the Black–Scholes model

## 1) What Exactly Is HedgeGPT?
HedgeGPT is a sophisticated trading and risk-management engine that leverages Q-learning to price and dynamically hedge European options. Rather than relying purely on closed-form formulas, HedgeGPT uses large-scale Monte Carlo simulation of underlying asset paths, encodes the market state with B-spline basis functions, and trains a Q-learner to discover optimal hedging strategies in a simulated Black–Scholes world. The result is an AI agent that learns to replicate option payoffs and manage risk in a fully automated, data-driven fashion.

## 2) Key Features

**2.1) Monte Carlo Backbone**

• Simulates tens of thousands of asset‐price paths under risk‐neutral dynamics

• Captures path dependency and nonlinear payoffs

**2.2) B-Spline State Encoding**

• Transforms log‐prices into a compact, continuous feature space

• Enables smooth, high-resolution representation of market states

**2.3) Q-Learning Hedging Agent**

• Learns optimal hedging ratios by maximizing a variance-based reward

• Balances hedging P&L against portfolio variance (risk aversion)

**2.4) Variance-Based Reward Function**

• Penalizes portfolio volatility while rewarding accurate replication

• Adapts dynamically to market fluctuations

**2.5) Transparent Black–Scholes Benchmark**

• Compares the agent’s learned price to the analytic Black–Scholes price

• Provides an intuitive performance gauge
