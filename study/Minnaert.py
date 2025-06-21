import numpy as np
import matplotlib.pyplot as plt

# === パラメータ設定 ===
gamma = 1.4             # 断熱指数（空気中の気泡）
p0 = 101325             # [Pa] 大気圧
rho = 1000              # [kg/m^3] 水の密度

# === 気泡半径の範囲（2mm〜12mm）===
R_vals = np.linspace(1e-4, 12e-3, 1000)  # [m]

# === Minnaertの式で共鳴周波数を計算 ===
f_vals = (1 / (2 * np.pi * R_vals)) * np.sqrt(3 * gamma * p0 / rho)

# === グラフ描画 ===
plt.figure(figsize=(9, 5))
plt.plot(R_vals * 1e3, f_vals, color='blue', lw=2)  # 横軸を mm に変換
plt.xlim(2, 12)
plt.ylim(0, 10000)

# === 軸目盛り設定 ===
xticks = np.arange(0, 13, 1)   # 0.5mm刻み
yticks = np.arange(0, 10001, 500)  # 500Hz刻み
plt.xticks(xticks)
plt.yticks(yticks)

# === グリッドとラベル ===
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("Bubble Radius $R_0$ [mm]")
plt.ylabel("Resonance Frequency $f$ [Hz]")
plt.title("Minnaert Resonance Frequency vs Bubble Radius")
plt.tight_layout()
plt.savefig("minnaert.png", dpi=300)
plt.show()