# HedgeGPT
Unlock next-generation option pricing and hedging with an AI-powered reinforcement-learning framework built on the Black–Scholes model

## 1) What Exactly Is HedgeGPT?
HedgeGPT is a sophisticated trading and risk-management engine that leverages Q-learning to price and dynamically hedge European options. Rather than relying purely on closed-form formulas, HedgeGPT uses large-scale Monte Carlo simulation of underlying asset paths, encodes the market state with B-spline basis functions, and trains a Q-learner to discover optimal hedging strategies in a simulated Black–Scholes world. The result is an AI agent that learns to replicate option payoffs and manage risk in a fully automated, data-driven fashion.

## 2) Key Features

**2.1) Monte Carlo Backbone**
• Simulates tens of thousands of asset‐price paths under risk‐neutral dynamics
• Captures path dependency and nonlinear payoffs
