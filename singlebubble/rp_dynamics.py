import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- パラメータ設定 ---
N = 3
m = 3.0
p = 1.0
x_fixed = p**(-1/m)  # 固定点 r*
y_fixed = 0

# --- ベクトル場 ---
x_vals = np.linspace(0.3, 2.0, 40)
y_vals = np.linspace(-2.0, 2.0, 40)
X, Y = np.meshgrid(x_vals, y_vals)

# ベクトル場の定義
def f(x, y):
    return ((N - 2)*(x**(-m) - p) - (N / 2)*y**2) / x

U = Y
V = f(X, Y)
norm = np.sqrt(U**2 + V**2)
U, V = U / norm, V / norm  # 正規化

# --- 数値解（非線形系の軌道） ---
def rpe_system(t, z):
    x, y = z
    dxdt = y
    dydt = ((N - 2)*(x**(-m) - p) - (N / 2)*y**2) / x
    return [dxdt, dydt]

# 初期条件と時間範囲
z0 = [1.5, 0.0]  # 初期値（固定点からずらす）
t_span = (0, 30)
sol = solve_ivp(rpe_system, t_span, z0, t_eval=np.linspace(*t_span, 1000))

# --- 線形近似解（振動） ---
A = -m * (N - 2) * p**(1 + 2/m)
omega = np.sqrt(-A)
x_lin = x_fixed + 0.1 * np.cos(omega * sol.t)
y_lin = -0.1 * omega * np.sin(omega * sol.t)

# --- 描画 ---
plt.figure(figsize=(10, 6))
plt.quiver(X, Y, U, V, alpha=0.4, color='gray')
plt.plot(sol.y[0], sol.y[1], label='Numerical solution', color='blue')
plt.plot(x_lin, y_lin, '--', label='Linearized near fixed point', color='green')
plt.plot(x_fixed, y_fixed, 'ro', label='Fixed point')

plt.xlabel(r"$r$")
plt.ylabel(r"$\dot{r}$")
plt.title("Rayleigh–Plesset Dynamics (N=3, m=3) with Fixed Point and Linearized Orbit")
plt.legend()
plt.grid(True)
plt.xlim(0.3, 2.0)
plt.ylim(-2.0, 2.0)
plt.savefig("rp_dynamics.png")
plt.show()
