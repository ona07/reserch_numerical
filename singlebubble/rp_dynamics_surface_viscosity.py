import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

# --- パラメータ設定 ---
N = 3
m = 3.0
p = 1.0    # 無次元外圧
S = 0.2    # 無次元表面張力: 2σ / (P0 R0)
V = 0.3    # 無次元粘性項: 4μ / (R0 sqrt(ρ P0))

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
    return ((N - 2)*(x**(-m) - p - S / x) - V * y - (N / 2)*y**2) / x

U = Y
V_field = f(X, Y)
norm = np.sqrt(U**2 + V_field**2)
U, V_norm = U / norm, V_field / norm

# --- 数値解（非線形系の軌道） ---
def rpe_system(t, z):
    x, y = z
    dxdt = y
    dydt = ((N - 2)*(x**(-m) - p - S / x) - V * y - (N / 2)*y**2) / x
    return [dxdt, dydt]

# 初期条件と時間範囲
z0 = [1.5, 0.0]
t_span = (0, 30)
sol = solve_ivp(rpe_system, t_span, z0, t_eval=np.linspace(*t_span, 1000))

# --- 線形近似解（減衰振動） ---
A = (-(m * (N - 2)) * x_fixed**(-m - 1) + (N - 2) * S / x_fixed**2) / x_fixed
B = -V / x_fixed
disc = B**2 + 4*A  # 判別式

if disc >= 0:
    # 実数解（ノード or 発散）
    lam1 = (B + np.sqrt(disc)) / 2
    lam2 = (B - np.sqrt(disc)) / 2
    x_lin = x_fixed + 0.1 * np.exp(lam1 * sol.t)
    y_lin = lam1 * (x_lin - x_fixed)
else:
    # 減衰振動
    omega = np.sqrt(-disc) / 2
    lam = B / 2
    x_lin = x_fixed + 0.1 * np.exp(lam * sol.t) * np.cos(omega * sol.t)
    y_lin = -0.1 * np.exp(lam * sol.t) * (lam * np.cos(omega * sol.t) + omega * np.sin(omega * sol.t))

# --- 描画 ---
plt.figure(figsize=(10, 6))
plt.quiver(X, Y, U, V_norm, alpha=0.4, color='gray')
plt.plot(sol.y[0], sol.y[1], label='Numerical solution', color='blue')
plt.plot(x_lin, y_lin, '--', label='Linearized with damping', color='green')
plt.plot(x_fixed, y_fixed, 'ro', label='Fixed point')

plt.xlabel(r"$r$")
plt.ylabel(r"$\dot{r}$")
plt.title("RPE with Surface Tension and Viscosity (N=3, m=3, S=0.2, V=0.3)")
plt.legend()
plt.grid(True)
plt.xlim(0.3, 2.0)
plt.ylim(-2.0, 2.0)
plt.savefig("rp_dynamics_surface_viscosity.png")
plt.show()
