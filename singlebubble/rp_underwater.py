import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import os
from joblib import Parallel, delayed

# === 定数パラメータ ===
N = 3
P0 = 1.01325e5  # Pa
rho = 1000      # kg/m^3
R0 = 2.0e-3     # m
R_init = R0
Rdot_init = 0.0
m = 3
t_span = (0, 0.1)  # sec
t_eval = np.linspace(*t_span, 1000)
sigma = 72.8e-3  # N/m

a = R_init / R0
def compute_Pext_for_p(p_target):
    return (p_target * P0) / (a ** m)

# 定常半径計算
def compute_equilibrium_radius(Pext):
    def f(R):
        term_pressure = (P0 / rho / (N - 2)) * (R0 / R)**m
        term_external = Pext / rho / (N - 2)
        term_surface = sigma * (N - 1) / rho / (N - 2) / R
        return term_pressure - term_external - term_surface
    sol = root_scalar(f, bracket=[1e-6, 1e-2], method='bisect')
    return sol.root

# === パラメータセット ===
p_values = [1.0, 1.5, 2.0, 3.0]
mu_values = [1.002e-3]  # Pa·s

def simulate_single_case(p, mu):
    Pext = compute_Pext_for_p(p)
    Req = compute_equilibrium_radius(Pext)

    def rp_with_p(t, y):
        R, Rdot = y
        if R <= 0:
            return [0, 0]
        term_pressure = (P0 / rho / (N - 2)) * (R0 / R)**m
        term_external = Pext / rho / (N - 2)
        term_surface = sigma * (N - 1) / rho / (N - 2) / R
        term_viscous = mu / (2 * rho) * Rdot
        Rddot = (term_pressure - term_external - term_surface - term_viscous - (N / 2) * Rdot**2) / R
        return [Rdot, Rddot]

    sol = solve_ivp(
        rp_with_p, t_span, [R_init, Rdot_init], t_eval=t_eval,
        method='Radau', rtol=1e-8, atol=1e-10
    )
    R_sol = sol.y[0] * 1e3  # mm
    t_sol = sol.t * 1e3     # ms

    return (p, mu, t_sol, R_sol, Req * 1e3)

# 並列実行
results = Parallel(n_jobs=-1)(delayed(simulate_single_case)(p, mu) for p in p_values for mu in mu_values)

# === 図の描画（2×2配置） ===
n_rows, n_cols = 2, 2
fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3))
axs = axs.reshape(-1)

for idx, result in enumerate(results):
    p, mu, t_sol, R_sol, Req = result
    ax = axs[idx]
    ax.plot(t_sol, R_sol, linewidth=0.5, label='R(t)')
    ax.axhline(Req, color='r', linestyle='--', label='Equilibrium')
    ax.set_xlabel("Time [ms]", fontsize=10)
    ax.set_ylabel("Radius [mm]", fontsize=10)
    ax.set_title(f"p={p}", fontsize=12)
    ax.grid(True)
    ax.set_ylim(0, 2.5)
    ax.legend(fontsize=8)

# 図の保存
plt.tight_layout()
filename = "rp_underwater.png"
plt.savefig(filename, dpi=300)
plt.show()
plt.close()

print(f"✅ Saved combined plot to {filename}")
