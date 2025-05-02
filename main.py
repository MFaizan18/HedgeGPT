# Copyright 2025 Mohammed Faizan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import numpy as np
import pandas as pd
from scipy.stats import norm
import bspline.splinelab as splinelab
from scipy.interpolate import splev

def prompt_option_parameters():
    """
    Prompt for option inputs, with:
      – μ, σ, r: accept decimal (0.05) or percent (5 → 0.05)
      – S₀, K, M: accept only absolute values (no percent conversion)
    """

    def get_pct_or_decimal(prompt, name, lower=0.0, upper=1.0):
        while True:
            try:
                raw = float(input(prompt))
            except ValueError:
                print(f"  → Please enter a valid number for {name}.")
                continue

            # treat >1 as percent
            if raw > 1:
                print(f"  ↳ You entered {raw:.2f}. Interpreting as {raw:.2f}% → {raw/100:.4f}.")
                raw /= 100

            if not (lower <= raw <= upper):
                print(f"  → {name} should be between {lower:.2f} and {upper:.2f}.")
                continue

            return raw

    def get_absolute(prompt, name, lower=1e-6, upper=1e6):
        while True:
            try:
                val = float(input(prompt))
            except ValueError:
                print(f"  → Please enter a valid number for {name}.")
                continue

            if not (lower <= val <= upper):
                print(f"  → {name} should be between {lower} and {upper}.")
                continue

            return val

    print("📈 Risk-Neutral QLBS / Black–Scholes Comparison\n")

    S0 = get_absolute("1) Initial stock price S₀ (e.g. 50, 100): ", "S₀", 0.01, 1e6)
    mu = get_pct_or_decimal("2) Expected drift μ (e.g. 0.05 or 5): ", "μ", -1.0, 1.0)
    sigma = get_pct_or_decimal("3) Volatility σ (e.g. 0.15 or 15): ", "σ", 1e-4, 5.0)
    r = get_pct_or_decimal("4) Risk-free rate r (e.g. 0.03 or 3): ", "r", 0.0, 1.0)
    K = get_absolute("5) Strike price K (e.g. 80, 100): ", "K", 0.01, 1e6)
    M = get_absolute("6) Time to maturity M in years (e.g. 0.25 = 3 months): ",
                     "M", 1e-4, 50.0)  # allow up to, say, 50 years

    while True:
        option_type = input("7) Option type: 'call' or 'put': ").strip().lower()
        if option_type in ("call","put"):
            break
        print("  → Please enter either 'call' or 'put'.")

    print("\nRunning risk-neutral pricing (λ = 0) vs. Black–Scholes for comparison:")
    print(f"  S₀ = {S0}, μ = {mu}, σ = {sigma}, r = {r}, K = {K}, M = {M}, Type = {option_type.upper()}\n")

    return S0, mu, sigma, r, K, M, option_type

def terminal_payoff(ST, K, option_type='call'):
    if option_type=='call':
        return max(ST-K, 0)
    else:
        return max(K-ST, 0)

def function_A_vec(t, delta_S_hat, data_mat, reg_param):
    phi_t = data_mat[t,:,:]
    dS_hat_t = delta_S_hat.iloc[:,t].astype(float)
    dS_hat_sq = dS_hat_t**2
    A_mat = np.dot(phi_t.T, dS_hat_sq.values[:,None]*phi_t)
    A_mat += reg_param * np.eye(phi_t.shape[1])
    return A_mat

def function_B_vec(t, Pi_hat, delta_S_hat, delta_S, data_mat, gamma, risk_lambda):
    phi_t = data_mat[t,:,:]
    pi_next = Pi_hat.iloc[:,t+1].values.astype(float)
    dS_hat_t = delta_S_hat.iloc[:,t].values.astype(float)
    delta_S_t = delta_S.iloc[:,t].values.astype(float)
    if risk_lambda>0:
        penalty_term = (1.0/(2*gamma*risk_lambda))*delta_S_t
    else:
        penalty_term = 0.0
    term = pi_next*dS_hat_t + penalty_term
    return np.dot(phi_t.T, term)

def function_C_vec(t, data_mat, reg_param):
    phi_t = data_mat[t,:,:]
    C_mat = np.dot(phi_t.T, phi_t)
    C_mat += reg_param * np.eye(phi_t.shape[1])
    return C_mat

def function_D_vec(t, Q, R, data_mat, gamma):
    phi_t = data_mat[t,:,:]
    term = (R.iloc[:,t] + gamma*Q.iloc[:,t+1]).values.astype(float)
    return np.dot(phi_t.T, term)

def main():
    # 0) prompt
    S0, mu, sigma, r, K, M, option_type = prompt_option_parameters()

    # 1) parameters
    risk_lambda = 0.0
    N_MC        = 20000
    T           = 24
    delta_t     = M / T
    gamma       = np.exp(-r*delta_t)
    reg_param   = 1e-3
    np.random.seed(42)

    # 2) simulate S
    S = pd.DataFrame([], index=range(1, N_MC+1), columns=range(T+1))
    S.loc[:,0] = S0
    RN = pd.DataFrame(np.random.randn(N_MC,T), index=range(1, N_MC+1), columns=range(1, T+1))
    for t in range(1, T+1):
      S.loc[:,t] = S.loc[:,t-1] * np.exp((mu - 1/2 * sigma**2) * delta_t + sigma * np.sqrt(delta_t) * RN.loc[:,t])

    # 3) delta's
    delta_S = S.loc[:,1:T].values - np.exp(r * delta_t) * S.loc[:,0:T-1]
    delta_S_hat = delta_S.apply(lambda x: x - np.mean(x), axis=0)

    # 4) state X
    X = - (mu - 1/2 * sigma**2) * np.arange(T+1) * delta_t + np.log(S.astype(float))

    # 5) spline & basis
    X_min = np.min(np.min(X))
    X_max = np.max(np.max(X))
    p = 4
    ncolloc = 24
    tau = np.linspace(X_min, X_max, ncolloc)
    k = splinelab.aptknt(tau, p)
    degree = p - 1
    n_basis = len(k) - p
    num_t_steps = T + 1
    N_MC = X.shape[0]
    flat_x = X.values.flatten(order='F')
    flat_basis = np.zeros((flat_x.size, n_basis))
    for j in range(n_basis):
      coeffs = np.zeros(n_basis)
      coeffs[j] = 1.0
      flat_basis[:, j] = splev(flat_x, (k, coeffs, degree))
    data_mat_t = flat_basis.reshape((num_t_steps, N_MC, n_basis))

    # 6) replicate Pi, Pi_hat, a
    Pi = pd.DataFrame(index=range(1, N_MC+1), columns=range(T+1))
    Pi.iloc[:, -1] = S.iloc[:, -1].apply(lambda x: terminal_payoff(x, K, option_type))
    Pi_hat = pd.DataFrame(index=range(1, N_MC+1), columns=range(T+1))
    Pi_hat.iloc[:, -1] = Pi.iloc[:, -1] - np.mean(Pi.iloc[:, -1])
    a = pd.DataFrame(index=range(1, N_MC+1), columns=range(T+1))
    a.iloc[:, -1] = 0


    for t in range(T - 1, -1, -1):
        A_mat = function_A_vec(t, delta_S_hat, data_mat_t, reg_param)
        B_vec = function_B_vec(t, Pi_hat, delta_S_hat, delta_S, data_mat_t, gamma, risk_lambda)
        phi = np.dot(np.linalg.inv(A_mat), B_vec)
        a.iloc[:, t] = np.dot(data_mat_t[t, :, :], phi)
        Pi.iloc[:, t] = gamma * (Pi.iloc[:, t+1] - a.iloc[:, t] * delta_S.iloc[:, t])
        Pi_hat.iloc[:, t] = Pi.iloc[:, t] - Pi.iloc[:, t].mean()

    # 7) rewards
    R = pd.DataFrame(np.nan, index=range(1, N_MC+1), columns=range(T+1), dtype=np.float64)
    R.iloc[:,-1] = -risk_lambda * np.var(Pi.iloc[:,-1].values.astype(float))

    for t in range(T):
        R.loc[:, t] = (
            gamma * a.loc[:, t].values.astype(float) * delta_S.loc[:, t].values.astype(float)
            - risk_lambda * np.var(Pi.loc[:, t].values.astype(float))
        )

    # 8) Q‐learning
    Q = pd.DataFrame(np.nan, index=range(1, N_MC+1), columns=range(T+1), dtype=np.float64)
    Q.iloc[:,-1] = (-Pi.iloc[:,-1] - risk_lambda * np.var(Pi.iloc[:,-1])).astype(float)


    for t in range(T-1, -1, -1):
        C_mat = function_C_vec(t,data_mat_t,reg_param)
        D_vec = function_D_vec(t, Q,R,data_mat_t,gamma)
        omega = np.dot(np.linalg.inv(C_mat), D_vec)
        Q.loc[:,t] = np.dot(data_mat_t[t,:,:], omega)

    option_price = abs(Q.loc[:,0].mean())

    # 9) Black–Scholes
    def bs_price_call():
        d1 = (np.log(S0/K)+(r+0.5*sigma**2)*M)/(sigma*np.sqrt(M))
        d2 = d1 - sigma*np.sqrt(M)
        return S0*norm.cdf(d1)-K*np.exp(-r*M)*norm.cdf(d2)
    def bs_price_put():
        d1 = (np.log(S0/K)+(r+0.5*sigma**2)*M)/(sigma*np.sqrt(M))
        d2 = d1 - sigma*np.sqrt(M)
        return K*np.exp(-r*M)*norm.cdf(-d2)-S0*norm.cdf(-d1)

    print("\n✅  Pricing complete!\n")
    print(f"RL Q-Learner price (Our Model) = {option_price:.2f}")

    if option_type == 'call':
        print(f"Black–Scholes (call) = {bs_price_call():.2f}")
    elif option_type == 'put':
        print(f"Black–Scholes (put) = {bs_price_put():.2f}")



if __name__ == "__main__":
    main()

