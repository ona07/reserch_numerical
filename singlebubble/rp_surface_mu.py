import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import os

# === 基本パラメータ ===
N = 3
P0 = 1
Pext = 2
rho = 1
R0 = 1
R_init = 1.0
Rdot_init = 0.0
t_span = (0, 10)
t_eval = np.linspace(*t_span, 1000)

# === m, sigma, mu のパターン ===
m = 3
sigma_values = [1, 5, 15]
mu_values = [0, 1]

# === RPE with surface tension and viscosity ===
def rp_with_sigma_mu(t, y, m, sigma, mu):
    R, Rdot = y
    if R <= 0:
        return [0, 0]
    term_pressure = (P0 / rho / (N - 2)) * (R0 / R)**m
    term_external = Pext / rho / (N - 2)
    term_surface = sigma * (N - 1) / rho / (N - 2) / R
    term_viscous = mu / (2 * rho) * Rdot
    Rddot = (term_pressure - term_external - term_surface - term_viscous - (N / 2) * Rdot**2) / R
    return [Rdot, Rddot]

# === 定常半径を計算 ===
def compute_equilibrium_radius(m, sigma):
    def f(R):
        term_pressure = (P0 / rho / (N - 2)) * (R0 / R)**m
        term_external = Pext / rho / (N - 2)
        term_surface = sigma * (N - 1) / rho / (N - 2) / R
        return term_pressure - term_external - term_surface

    try:
        sol = root_scalar(f, bracket=[0.01, 5.0], method='bisect')
        return sol.root if sol.converged else None
    except:
        return None

# === プロット設定 ===
fig, axes = plt.subplots(len(mu_values), len(sigma_values), figsize=(12, 8), sharey=True)

for j, mu in enumerate(mu_values):  # rows → mu
    for k, sigma in enumerate(sigma_values):  # cols → sigma
        ax = axes[j, k]
        sol = solve_ivp(
            lambda t, y: rp_with_sigma_mu(t, y, m, sigma, mu),
            t_span, [R_init, Rdot_init],
            t_eval=t_eval, rtol=1e-8, atol=1e-10
        )
        R_sol = sol.y[0]
        ax.plot(sol.t, R_sol, label="R(t)", linewidth=2.0)

        # 定常半径描画
        R_eq = compute_equilibrium_radius(m, sigma)
        if R_eq:
            ax.axhline(R_eq, color='r', linestyle='--', linewidth=1.5,
                       label=f"Equilibrium R ≈ {R_eq:.2f}")

        ax.set_title(f"μ = {mu}, σ = {sigma}", fontsize=20)
        ax.set_xlabel("Time", fontsize=12)
        if k == 0:
            ax.set_ylabel("Radius R(t)", fontsize=15)

        ax.tick_params(labelsize=13)
        ax.grid(True)
        ax.legend(fontsize=15, loc='upper right')

# === 全体調整 ===
plt.tight_layout()
plt.savefig("rp_surface_mu.png", dpi=300)
plt.show()
