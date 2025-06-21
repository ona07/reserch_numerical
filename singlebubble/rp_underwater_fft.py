import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from scipy.fft import fft, fftfreq
import os

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

mu = 1.002e-4  # Pa·s（固定）
p = 3.0        # 対象のp値

# === 外部圧・定常半径の計算 ===
a = R_init / R0
Pext = (p * P0) / (a ** m)

def compute_equilibrium_radius(Pext):
    def f(R):
        term_pressure = (P0 / rho / (N - 2)) * (R0 / R)**m
        term_external = Pext / rho / (N - 2)
        term_surface = sigma * (N - 1) / rho / (N - 2) / R
        return term_pressure - term_external - term_surface
    sol = root_scalar(f, bracket=[1e-6, 1e-2], method='bisect')
    return sol.root

Req = compute_equilibrium_radius(Pext)

# 初期内部圧の計算（タイトル表示用）
p_internal_0 = P0 * (R0 / R_init)**m / 1000  # [kPa]

# === Rayleigh–Plesset 系の定義 ===
def rp(t, y):
    R, Rdot = y
    if R <= 0:
        return [0, 0]
    
    term_pressure = (P0 / rho / (N - 2)) * (R0 / R)**m
    term_external = Pext / rho / (N - 2)
    term_surface = sigma * (N - 1) / rho / (N - 2) / R
    term_viscous = mu / (2 * rho) * Rdot
    Rddot = (term_pressure - term_external - term_surface - term_viscous - (N / 2) * Rdot**2) / R

    return [Rdot, Rddot]

# === 数値解法 ===
sol = solve_ivp(
    rp, t_span, [R_init, Rdot_init],
    t_eval=t_eval, method='Radau', rtol=1e-8, atol=1e-10
)

# === 結果整形 ===
R_sol = sol.y[0] * 1e3        # mm
t_sol = sol.t * 1e3           # ms

# === グラフ描画（半径）===
fig, ax = plt.subplots()
ax.plot(t_sol, R_sol, label='Radius [mm]', color='tab:blue', linewidth=0.8)
ax.axhline(Req * 1e3, color='gray', linestyle='--', label='Equilibrium Radius', linewidth=0.8)
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Radius [mm]')
ax.grid(True)
ax.legend()
plt.title(f'Bubble Radius (p={p}, μ={mu:.1e}, p_internal_0={p_internal_0:.1f} kPa)')
plt.tight_layout()
plt.show()
plt.close()

# === フーリエ解析 ===
t_sec = t_sol / 1000               # [s]
dt = t_sec[1] - t_sec[0]           # サンプリング間隔
N = len(R_sol)                     # サンプル数
fs = 1 / dt                        # サンプリング周波数

R_detrended = R_sol - np.mean(R_sol)
R_fft = fft(R_detrended)
freq = fftfreq(N, d=dt)
amplitude = 2.0 / N * np.abs(R_fft)

# 正の周波数のみ抽出
idx = freq > 0
freq_plot = freq[idx]
amplitude_plot = amplitude[idx]

# === スペクトルプロット ===
plt.figure(figsize=(8, 4))
plt.plot(freq_plot, amplitude_plot, color='tab:purple', linewidth=1.0)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.title(f"FFT of Bubble Radius R(t) (p={p})")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close()

# === Minnaert理論周波数の拡張版（表面張力＋粘性補正あり）===
gamma = 1  # 空気の断熱指数
numerator = 3 * gamma * Pext + 2 * sigma / Req
visc_term = (4 * mu / Req)**2
f_minnaert = (1 / (2 * np.pi * Req)) * np.sqrt(numerator / rho - visc_term)

# 主ピーク周波数の抽出（最大振幅）
peak_idx = np.argmax(amplitude_plot)
f_peak = freq_plot[peak_idx]
A_peak = amplitude_plot[peak_idx]

# 表示
print(f"🧠 Minnaert frequency: {f_minnaert:.2f} Hz")
print(f"📈 FFT peak frequency: {f_peak:.2f} Hz")
print(f"📊 Peak amplitude: {A_peak:.4f}")

# === プロットにMinnaert線とFFTピーク線を追加（数値注記も） ===
plt.figure(figsize=(8, 4))
plt.plot(freq_plot, amplitude_plot, color='tab:purple', linewidth=1.0)

# Minnaert線
plt.axvline(f_minnaert, color='tab:orange', linestyle='--', linewidth=1.0, label=f'Minnaert f = {f_minnaert:.1f} Hz')

# FFTピーク線
plt.axvline(f_peak, color='tab:red', linestyle=':', linewidth=1.0, label=f'FFT peak f = {f_peak:.1f} Hz')

# 軸と凡例
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()

# 保存と表示
minnaert_filename = f"rp_fft_radius_minnaert_p{p}.png"
plt.savefig(minnaert_filename, dpi=300)
plt.show()
print(f"✅ Saved FFT + Minnaert comparison plot to: {minnaert_filename}")
