import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

# --- パラメータ設定 ---
N = 3
m = 3.0
p = 1.0   # 無次元外圧
S = 0.2   # 無次元表面張力 2σ / (P0 R0)

# --- 固定点の数値解 ---
def fixed_point_eq(x):
    return x**(-m) - p - S / x

res = root_scalar(fixed_point_eq, bracket=[0.1, 5.0], method='bisect')
x_fixed = res.root
y_fixed = 0

# --- ベクトル場 ---
x_vals = np.linspace(0.3, 2.0, 40)
y_vals = np.linspace(-2.0, 2.0, 40)
X, Y = np.meshgrid(x_vals, y_vals)

def f(x, y):
    return ((N - 2)*(x**(-m) - p - S / x) - (N / 2)*y**2) / x

U = Y
V = f(X, Y)
norm = np.sqrt(U**2 + V**2)
U, V = U / norm, V / norm

# --- 数値解（非線形系の軌道） ---
def rpe_system(t, z):
    x, y = z
    dxdt = y
    dydt = ((N - 2)*(x**(-m) - p - S / x) - (N / 2)*y**2) / x
    return [dxdt, dydt]

# 初期条件と時間範囲
z0 = [1.5, 0.0]  # 固定点から少しずらす
t_span = (0, 30)
sol = solve_ivp(rpe_system, t_span, z0, t_eval=np.linspace(*t_span, 1000))

# --- 線形近似解（振動） ---
# 固定点近傍でAを計算
A = (-(m * (N - 2)) * x_fixed**(-m - 1) + (N - 2) * S / x_fixed**2) / x_fixed
omega = np.sqrt(-A) if A < 0 else np.nan  # 安定中心のみ計算

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
plt.title("RPE Dynamics with Surface Tension (N=3, m=3, S=0.2)")
plt.legend()
plt.grid(True)
plt.xlim(0.3, 2.0)
plt.ylim(-2.0, 2.0)
plt.savefig("rp_dynamics_surface.png")
plt.show()
