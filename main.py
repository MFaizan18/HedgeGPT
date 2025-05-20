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
      – S₀, K: absolute values
      – M: enter in years (fractional for months/weeks), with trading-day feedback
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

    def get_maturity(prompt):
        print("Examples:")
        print("  • For 3 months (~63 trading days), enter 0.25")
        print("  • For 6 weeks (~31 trading days), enter 0.115")
        print("  • For 9 months (~189 trading days), enter 0.75")
        print("  • For 1.5 years (~378 trading days), enter 1.5")
        print("  • For 2 years (~504 trading days), enter 2\n")
        while True:
            try:
                val = float(input(prompt))
            except ValueError:
                print("  → Please enter a valid decimal or integer for M.")
                continue
            if not (1e-4 <= val <= 50):
                print("  → M should be between 0.0001 and 50 (years).")
                continue

            # feedback in calendar months and trading days
            months = val * 12
            trading_days = val * 252
            if val < 1:
                print(f"  ↳ You selected {val:.4f} years ≈ "
                      f"{months:.1f} months (~{trading_days:.0f} trading days).")
            else:
                yrs = int(val)
                rem_months = (val - yrs) * 12
                print(f"  ↳ You selected {yrs} year(s) + {rem_months:.1f} months "
                      f"(~{trading_days:.0f} trading days).")
            return val


    S0 = get_absolute("1) Initial stock price S₀ (e.g. 50, 100): ", "S₀")
    mu = get_pct_or_decimal("2) Expected drift μ (e.g. 0.05 or 5): ", "μ", -1.0, 1.0)
    sigma = get_pct_or_decimal("3) Volatility σ (e.g. 0.15 or 15): ", "σ", 1e-4, 5.0)
    r = get_pct_or_decimal("4) Risk-free rate r (e.g. 0.03 or 3): ", "r", 0.0, 1.0)
    K = get_absolute("5) Strike price K (e.g. 80, 100): ", "K")
    M = get_maturity("6) Time to maturity M in years (e.g. see examples above): ")
   

    while True:
        option_type = input("7) Option type: 'call' or 'put': ").strip().lower()
        if option_type in ("call", "put"):
            break
        print("  → Please enter either 'call' or 'put'.")

    # 11) print header for DP pricing
    

    print(f"  S₀ = {S0}, μ = {mu}, σ = {sigma}, r = {r}, "
          f"K = {K}, M = {M} years, Type = {option_type.upper()}\n")

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
    penalty_term = (1.0/(2*gamma*risk_lambda))*delta_S_t
    term = pi_next*dS_hat_t + 0 # made 'penalty_term' Zero for pure hedge
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

def function_S_vec(t, S_t_mat, reg_param):
    S_t = np.array(S_t_mat[:, :, t], dtype=np.float64)  # ensure float64
    num_Qbasis = S_t.shape[0]
    S_mat_reg = S_t + reg_param * np.eye(num_Qbasis)
    return S_mat_reg
  
def function_M_vec(t, Q_star, R, Psi_mat_t, gamma):
    R_t = R.iloc[:, t].values  # Shape (N_MC,)
    Q_next = Q_star.loc[:, t + 1].values  # Shape (N_MC,)
    targets = R_t + gamma * Q_next  # Shape (N_MC,)
    Psi_np = Psi_mat_t  # Shape (num_Qbasis, N_MC)
    M_t = np.dot(Psi_np, targets)  # Shape (num_Qbasis,)
    return M_t

def main():
    # 0) prompt
    S0, mu, sigma, r, K, M, option_type = prompt_option_parameters()

    # 1) parameters
    risk_lambda = 0.001 # Risk-aversion parameter: adjust based on desired risk sensitivity
    N_MC        = 20000
    T           = 6
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
    ncolloc = 12
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

    on_policy_price = abs(Q.loc[:,0].mean())

    print("\n🚀 Running HedgeGPT Option Pricing Engine")
    print(f"  • Risk-aversion (λ) = {risk_lambda:.3f}")
    print("  • Methods: On-Policy DP and Off-Policy RL")
    print("  • Benchmark: Black–Scholes Formula\n")

    # 9) print on‑policy results
    print(f"✅ On-Policy Price       = {on_policy_price:.2f}")
  
    # 10) off‑policy initialization
    eta = 0.5
    np.random.seed(42)

    # Save on‑policy optimal actions
    a_dp = a.copy()

    # Initialize off-policy actions (a_op)
    a_op = pd.DataFrame(0.0, index=range(1, N_MC + 1), columns=range(T + 1), dtype=float)
    a_op.iloc[:, -1] = 0.0  # No action at terminal time

    # Initialize off-policy portfolios
    Pi_op = pd.DataFrame(0.0, index=range(1, N_MC + 1), columns=range(T + 1), dtype=float)
    Pi_op.iloc[:, -1] = S.iloc[:, -1].apply(lambda x: float(terminal_payoff(x, K, option_type)))

    # Initialize rewards
    R_op = pd.DataFrame(0.0, index=range(1, N_MC + 1), columns=range(T + 1), dtype=float)
    R_op.iloc[:, -1] = -risk_lambda * np.var(Pi_op.iloc[:, -1])

    # 11) Backward simulate off‑policy
    for t in range(T-1, -1, -1):
        a_star_t = a_dp.iloc[:, t]
        noise = np.random.uniform(1-eta, 1+eta, size=N_MC)
        a_op.iloc[:, t] = (a_star_t * noise).astype(float)
        delta_S_t = delta_S.iloc[:, t].values
        Pi_op.iloc[:, t] = (gamma * (Pi_op.iloc[:, t+1] - a_op.iloc[:, t] * delta_S_t)).astype(float)
        reward_term = (gamma * a_op.iloc[:, t] * delta_S_t).astype(float)
        risk_penalty = risk_lambda * np.var(Pi_op.iloc[:, t])
        R_op.iloc[:, t] = (reward_term - risk_penalty).astype(float)

    # 12) override on‑policy with off‑policy
    a = a_op.copy()
    Pi = Pi_op.copy()
    R  = R_op.copy()

    # 13) build Psi_mat & S_t_mat
    num_MC, num_TS = a.shape
    a_1_1 = a.values.reshape((1, num_MC, num_TS))
    a_1_2 = 0.5 * a_1_1**2
    ones_3d = np.ones((1, num_MC, num_TS))
    A_stack = np.vstack((ones_3d, a_1_1, a_1_2))
    data_mat_swap_idx = np.swapaxes(data_mat_t, 0, 2)
    A_2 = np.expand_dims(A_stack, axis=1)
    D_2 = np.expand_dims(data_mat_swap_idx, axis=0)
    Psi_mat = np.multiply(A_2, D_2).reshape(-1, N_MC, num_TS, order='F')
    num_Qbasis = Psi_mat.shape[0]
    S_t_mat = np.zeros((num_Qbasis, num_Qbasis, num_TS))
    for t in range(num_TS):
        P = Psi_mat[:, :, t]
        S_t_mat[:, :, t] = P.dot(P.T)

    # 14) initialize off‑policy Q
    # implied Q-function by input data (using the first form in Eq.(68))
    Q_RL = pd.DataFrame([], index=range(1, N_MC+1), columns=range(T+1))
    Q_RL.iloc[:,-1] = - Pi.iloc[:,-1] - risk_lambda * np.var(Pi.iloc[:,-1])

    # optimal Q-function with optimal action
    Q_star = pd.DataFrame([], index=range(1, N_MC+1), columns=range(T+1))
    Q_star.iloc[:,-1] = Q_RL.iloc[:,-1]

    # max_Q_star_next = Q_star.iloc[:,-1].values 
    max_Q_star = np.zeros((N_MC,T+1))
    max_Q_star[:,-1] = Q_RL.iloc[:,-1].values

    # 15) The backward loop
    for t in range(T-1, -1, -1):
        
        # calculate vector W_t
        S_mat_reg = function_S_vec(t,S_t_mat,reg_param) 
        M_t = function_M_vec(t,Q_star, R, Psi_mat[:,:,t], gamma)
        W_t = np.dot(np.linalg.inv(S_mat_reg), M_t)  
        
        # reshape to a matrix W_mat  
        W_mat = W_t.reshape((3, n_basis), order='F')  
            
        # make matrix Phi_mat
        Phi_mat = data_mat_t[t,:,:].T  

        # compute matrix U_mat of dimension N_MC x 3 
        U_mat = np.dot(W_mat, Phi_mat)
        
        # compute vectors U_W^0,U_W^1,U_W^2 as rows of matrix U_mat  
        U_W_0 = U_mat[0,:]
        U_W_1 = U_mat[1,:]
        U_W_2 = U_mat[2,:]
        
        # use hedges 'a_dp' computed as in DP approach:
        # in this way, errors of function approximation do not back-propagate. 
        # This provides a stable solution, 
        
        max_Q_star[:,t] = U_W_0 + a_dp.loc[:,t] * U_W_1 + 0.5 * (a_dp.loc[:,t]**2) * U_W_2       
      
        # update dataframes     
        Q_star.loc[:,t] = max_Q_star[:,t]
      
        # update the Q_RL solution given by a dot product of two matrices W_t Psi_t
        Psi_t = Psi_mat[:,:,t].T 
        Q_RL.loc[:,t] = np.dot(Psi_t, W_t)

    off_policy_price = abs(Q_RL.iloc[:, 0].mean())

    # 16) Black–Scholes
    def bs_price_call():
        d1 = (np.log(S0/K)+(r+0.5*sigma**2)*M)/(sigma*np.sqrt(M))
        d2 = d1 - sigma*np.sqrt(M)
        return S0*norm.cdf(d1)-K*np.exp(-r*M)*norm.cdf(d2)
    def bs_price_put():
        d1 = (np.log(S0/K)+(r+0.5*sigma**2)*M)/(sigma*np.sqrt(M))
        d2 = d1 - sigma*np.sqrt(M)
        return K*np.exp(-r*M)*norm.cdf(-d2)-S0*norm.cdf(-d1)

    # 17) print off‑policy results
    print(f"✅ Off-Policy Price      = {off_policy_price:.2f}")
    print(f"\n Black–Scholes = "
          f"{bs_price_call() if option_type=='call' else bs_price_put():.2f}")

if __name__ == "__main__":
    main()

