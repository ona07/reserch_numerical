import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# パラメータ
m = 1.0      # 質量
k = 20.0     # バネ定数
c = 1.5      # 減衰係数
p = 5.0    # 定数外力（負）

# 微分方程式定義
def forced_damped_oscillator(t, y):
    x, v = y
    dxdt = v
    dvdt = -(c/m)*v - (k/m)*x - (p/m)
    return [dxdt, dvdt]

# 初期条件
x0 = 0.0     # 初期変位
v0 = 0.0     # 初期速度
y0 = [x0, v0]

# 時間設定
t_span = (0, 10)
t_eval = np.linspace(*t_span, 1000)

# 数値解（RK45）
sol = solve_ivp(forced_damped_oscillator, t_span, y0, t_eval=t_eval, method='RK45')

# プロット
plt.figure(figsize=(10, 5))
plt.plot(sol.t, sol.y[0], label="Displacement x(t)")
plt.plot(sol.t, sol.y[1], '--', label="Velocity v(t)")
plt.axhline(-p/k, color='gray', linestyle=':', label="Static Equilibrium")
plt.title("Forced Damped Oscillator with Constant Force p")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("under_damping.png")
plt.show()