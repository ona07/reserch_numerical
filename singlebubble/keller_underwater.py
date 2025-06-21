import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

# ======================== 共通パラメータ設定 ========================
rho = 1000            # 密度 (kg/m^3)
S = 0.072             # 表面張力 (N/m)
mu = 1.02e-3          # 動粘性係数 (Pa·s)
kappa = 5/3           # 多方指数
c = 1450              # 音速 (m/s)
p_inf = 1e5           # 周囲圧力 (Pa)
R0 = 1e-3             # 初期半径 (5 mm)
p_g0 = 1000           # 初期気体圧 (Pa)
p_v = 2300            # 蒸気圧 (Pa)
t_span = (0, 0.1)       # 時間範囲 [s]
t_eval = np.linspace(*t_span, 2000)

# ======================== スケーリング係数のリスト ========================
p_values = [1.0, 1.5, 2.0, 3.0]

# ======================== プロット準備 ========================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

# ======================== pごとのループ処理 ========================
for idx, p in enumerate(p_values):
    term_A_list, term_B_list, term_C_list, time_record = [], [], [], []

    def pl(R, R_dot):
        term1 = p_g0 * (R0 / R)**(3 * kappa)
        term2 = -4 * mu * R_dot / R
        term3 = -2 * S / R
        term4 = -(p * p_inf - p_v)
        return term1 + term2 + term3 + term4

    def km_system(t, y):
        R, U = y
        p_l = pl(R, U)
        delta_R = R * 1e-3
        dpl_dt = (pl(R + delta_R, U) - p_l) / delta_R

        term_A = (1 + U / c) * p_l / rho
        term_B = R * dpl_dt / (rho * c)
        term_C = (3/2 - U / (2 * c)) * U**2
        denominator = (1 - U / c) * R
        R_ddot = (term_A + term_B - term_C) / denominator

        term_A_list.append(term_A)
        term_B_list.append(term_B)
        term_C_list.append(term_C)
        time_record.append(t)
        return [U, R_ddot]

    y0 = [R0, 0]
    sol = solve_ivp(km_system, t_span, y0, t_eval=t_eval, method='RK45',
                    rtol=1e-8, atol=1e-10)

    R_t = sol.y[0] * 1e3        # mm
    U_t = sol.y[1]              # m/s
    time = sol.t * 1e3          # ms

    ax1 = axes[idx]
    ax1.plot(time, R_t, color='tab:blue', linewidth=0.3, label='Radius [mm]')
    ax1.set_title(f'p = {p}')
    ax1.set_xlabel('Time [ms]')
    ax1.set_ylabel('Radius [mm]')
    ax1.grid(True)

plt.tight_layout()
plot_path = "keller_underwater.png"
plt.savefig(plot_path, dpi=300)
plt.show()
print(f"✅ Saved 2×2 radius plot to: {plot_path}")
