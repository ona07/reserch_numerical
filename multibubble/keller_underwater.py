import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# === パラメータ ===
rho = 1000          # 液体密度 [kg/m^3]
c = 1500            # 音速 [m/s]
S = 0.072           # 表面張力 [N/m]
mu = 1e-3           # 動粘性係数 [Pa·s]
kappa = 1.4         # ポリトロープ指数
p0 = 1e5            # 周囲圧 [Pa]
p_g0 = 1e5          # 初期ガス圧 [Pa]
R0 = 1e-3           # 初期半径 [m]
d = 5e-2            # 2気泡間距離 [m]

# === 音響相互作用のための履歴バッファ ===
history = {'t': [], 'R1': [], 'U1': [], 'R2': [], 'U2': []}

# === 有効圧力 pL ===
def pL(R, R_dot):
    p_g = p_g0 * (R0 / R)**(3 * kappa)
    return p_g - 4 * mu * R_dot / R - 2 * S / R

# === 相互作用圧 p_inf_i (遅延付き) ===
def p_inf(i, t, history, d):
    j = 1 - i  # 相手のインデックス
    delay = d / c
    t_delay = t - delay

    # 補間関数を定義（履歴があれば）
    if len(history['t']) < 2 or t_delay < history['t'][0]:
        return p0

    interp_R = interp1d(history['t'], history[f'R{j+1}'], fill_value="extrapolate")
    interp_U = interp1d(history['t'], history[f'U{j+1}'], fill_value="extrapolate")
    Rj = interp_R(t_delay)
    Uj = interp_U(t_delay)

    phi_j = Rj**2 * Uj
    g_j = 2 * Rj * Uj**2 + Rj**2 * 0  # dU/dt を省略または別途補完

    return p0 + rho * g_j / d  # 論文の圧力項の単純化版

# === Keller–Miksisの右辺 ===
def km_rhs(t, y):
    R1, U1, R2, U2 = y

    # 履歴に保存
    history['t'].append(t)
    history['R1'].append(R1)
    history['U1'].append(U1)
    history['R2'].append(R2)
    history['U2'].append(U2)

    # 気泡1のKM方程式
    pL1 = pL(R1, U1)
    p_inf1 = p_inf(0, t, history, d)
    dpL1_dt = 0  # 近似 or 無視
    denom1 = (1 - U1 / c) * R1
    term1 = (1 + U1 / c) * (pL1 - p_inf1) / rho
    term2 = R1 * dpL1_dt / (rho * c)
    term3 = (3/2 - U1 / (2 * c)) * U1**2
    R1_ddot = (term1 + term2 - term3) / denom1

    # 気泡2のKM方程式
    pL2 = pL(R2, U2)
    p_inf2 = p_inf(1, t, history, d)
    dpL2_dt = 0
    denom2 = (1 - U2 / c) * R2
    term1b = (1 + U2 / c) * (pL2 - p_inf2) / rho
    term2b = R2 * dpL2_dt / (rho * c)
    term3b = (3/2 - U2 / (2 * c)) * U2**2
    R2_ddot = (term1b + term2b - term3b) / denom2

    return [U1, R1_ddot, U2, R2_ddot]

# === 初期条件と数値積分 ===
y0 = [R0, 0, R0, 0]
t_span = (0, 0.1)
t_eval = np.linspace(*t_span, 1000)

sol = solve_ivp(km_rhs, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10)

# === 可視化 ===
R1 = sol.y[0] * 1e3
R2 = sol.y[2] * 1e3
t_ms = sol.t * 1e3

plt.plot(t_ms, R1, label='Bubble 1 [mm]')
plt.plot(t_ms, R2, label='Bubble 2 [mm]', linestyle='--')
plt.xlabel('Time [ms]')
plt.ylabel('Radius [mm]')
plt.title('Two-Bubble Interaction via Keller–Miksis')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("keller_underwater")
plt.show()
