import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# === パラメータ設定 ===
N = 3  # 空間次元

# 各分類の代表的 m 値
m_values = {
    'm < N': 2.0,     # B系列
    'm = N': 3.0,     # A系列
    'm > N': 5.0      # C系列
}

# p値リスト（各分類の境界や分岐を含む）
p_values = [1.0, 1.5, 3.0]

# 初期条件
r0 = 1.0
r_dot0 = 0.0

# 計算範囲
t_max = 10
t_eval = np.linspace(0, t_max, 1000)

# === RPEの微分方程式 ===
def rp_ode(t, y, N, m, p):
    r, r_dot = y
    if r <= 0:
        return [0, 0]  # collapse 保護
    drdt = r_dot
    d2rdt2 = (N - 2) * (r**(-m) - p - (N / 2) * r_dot**2) / r
    return [drdt, d2rdt2]

from scipy.signal import find_peaks
import numpy as np

def classify_solution(r_values, t_values, collapse_threshold=2e-1, steady_tol=1e-2, min_peaks=2):
    valid_r = r_values[~np.isnan(r_values)]

    if len(valid_r) < 5:
        return "Collapse"  # データが少なすぎれば崩壊とみなす

    # 振動の有無を確認
    peaks, _ = find_peaks(valid_r, prominence=steady_tol / 2)
    num_peaks = len(peaks)

    # 条件3: 最終的に小さい値に収束
    final_value_small = valid_r[-1] < collapse_threshold

    if final_value_small:
        return "Collapse"
    elif num_peaks >= min_peaks:
        return "Oscillatory"
    elif np.max(valid_r) - np.min(valid_r) < steady_tol:
        return "Steady"
    else:
        return "Other"

# === 可視化 ===
fig, axs = plt.subplots(len(m_values), len(p_values), figsize=(18, 10), sharex=True, sharey=True)
# fig.suptitle("Rayleigh–Plesset Equation Dynamics for All m, p Combinations", fontsize=16)

for i, (m_label, m) in enumerate(m_values.items()):
    for j, p in enumerate(p_values):
        ax = axs[i, j]
        try:
            sol = solve_ivp(
                fun=lambda t, y: rp_ode(t, y, N, m, p),
                t_span=(0, t_max),
                y0=[r0, r_dot0],
                t_eval=t_eval,
                rtol=1e-8,
                atol=1e-10
            )
            r_sol = sol.y[0]

            # 崩壊検知（r < ε）
            collapse_time = None
            for k, r_val in enumerate(r_sol):
                if r_val <= 1e-4:
                    collapse_time = sol.t[k]
                    r_sol[k:] = np.nan
                    break

            solution_type = classify_solution(r_sol, sol.t)
            label = f"{solution_type}" + (f" (t={collapse_time:.2f})" if collapse_time else "")
            ax.plot(sol.t, r_sol, label=label, linewidth=2.0) 
        except Exception as e:
            ax.text(0.5, 0.5, "Error", ha='center', va='center')
            continue


        ax.set_title(f"{m_label}, m={m}, p={p}", fontsize=20)
        ax.set_ylim(0, 1.5)
        if i == len(m_values) - 1:
            ax.set_xlabel("Time t", fontsize=15)
        if j == 0:
            ax.set_ylabel("r(t)", fontsize=15)
        ax.grid(True)
        ax.legend(fontsize=15)
        ax.tick_params(labelsize=15)  # 軸目盛のフォントサイズ

# 描画の最後に保存
# plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.tight_layout()
plt.savefig("rp_scaled.png", dpi=300)
plt.show()
