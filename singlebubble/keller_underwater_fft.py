import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq
import os

# ======================== パラメータ設定 ========================
rho = 1000         # 密度 (kg/m^3)
S = 0.072          # 表面張力 (N/m)
mu = 1.02e-3       # 動粘性係数 (Pa·s)
kappa = 5/3        # 多方指数
c = 1450           # 音速 (m/s)
p_inf = 1e5        # 周囲圧力 (Pa)
p_v = 2300         # 蒸気圧 (Pa)
R0 = 5e-3          # 初期半径 (m)
p_g0 = 1000        # 初期気体圧 (Pa)

p = 3.0            # スケーリング係数
Pext = p * p_inf   # スケーリングされた外圧

t_span = (0, 0.1)    # 時間範囲 [s]
t_eval = np.linspace(*t_span, 2000)

# ======================== 気泡内圧関数 ========================
def pl(R, R_dot):
    term1 = p_g0 * (R0 / R)**(3 * kappa)
    term2 = -4 * mu * R_dot / R
    term3 = -2 * S / R
    term4 = -(Pext - p_v)
    return term1 + term2 + term3 + term4

# ======================== KM方程式 ========================
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

    return [U, R_ddot]

# ======================== 数値解法 ========================
y0 = [R0, 0]
sol = solve_ivp(km_system, t_span, y0, t_eval=t_eval, method='RK45',
                rtol=1e-8, atol=1e-10)

R_t = sol.y[0] * 1e3         # mm
time_ms = sol.t * 1e3        # ms
time_s = sol.t               # s

# ======================== プロット1: 半径の時間変化 ========================
plt.figure(figsize=(8, 4))
plt.plot(time_ms, R_t, color='tab:blue', linewidth=0.8, label='Radius [mm]')
plt.xlabel('Time [ms]')
plt.ylabel('Radius [mm]')
plt.title(f'Bubble Radius (Keller–Miksis, p={p})')
plt.grid(True)
plt.tight_layout()
plt.show()

# ======================== FFT解析 ========================
R_detrended = R_t - np.mean(R_t)
N = len(R_detrended)
dt = time_s[1] - time_s[0]
freq = fftfreq(N, d=dt)
amplitude = 2.0 / N * np.abs(fft(R_detrended))

# 正の周波数のみ抽出
idx = freq > 0
freq_plot = freq[idx]
amplitude_plot = amplitude[idx]

# ピーク周波数検出
peak_idx = np.argmax(amplitude_plot)
f_peak = freq_plot[peak_idx]
A_peak = amplitude_plot[peak_idx]

# ======================== Minnaert理論周波数（拡張） ========================
gamma = 5/3
Req = np.mean(sol.y[0])  # 平均半径 [m]
numerator = 3 * gamma * Pext + 2 * S / Req
visc_term = (4 * mu / Req)**2
f_minnaert = (1 / (2 * np.pi * Req)) * np.sqrt(numerator / rho - visc_term)

# ======================== プロット2: FFT vs Minnaert ========================
plt.figure(figsize=(8, 4))
plt.plot(freq_plot, amplitude_plot, color='tab:purple', linewidth=1.0)
plt.axvline(f_minnaert, color='tab:orange', linestyle='--', linewidth=1.0,
            label=f'Minnaert f = {f_minnaert:.1f} Hz')
plt.axvline(f_peak, color='tab:red', linestyle=':', linewidth=1.0,
            label=f'FFT peak f = {f_peak:.1f} Hz')

plt.text(f_minnaert, max(amplitude_plot)*0.8, f'{f_minnaert:.1f} Hz',
         rotation=90, color='tab:orange', va='center', ha='right', fontsize=9)
plt.text(f_peak, max(amplitude_plot)*0.6, f'{f_peak:.1f} Hz',
         rotation=90, color='tab:red', va='center', ha='left', fontsize=9)

plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
fft_filename = f"keller_fft_minnaert_p{p}.png"
plt.savefig(fft_filename, dpi=300)
plt.show()

# ======================== 結果表示 ========================
print(f"🧠 Minnaert frequency (corrected): {f_minnaert:.2f} Hz")
print(f"📈 FFT peak frequency:             {f_peak:.2f} Hz")
print(f"📊 Peak amplitude:                 {A_peak:.4f}")
print(f"✅ Saved FFT + Minnaert comparison plot to: {fft_filename}")
