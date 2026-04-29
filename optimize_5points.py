"""
求解优化问题:
最小化 ||p1p4||_2 + ||p2p4||_2 + ||p3p5||_2 - ||p1p3||_2 - ||p2p5||_2

约束:
  x1, x2 in [x_min, 400]   (x_min 作为参数扫描)
  x3 in [0, 200]
  x4 in [0, 200]
  x5 = 200 (固定)
  |y1| in [y_min, 200]     (y_min 作为参数扫描)
  |y2| in [y_min, 200]
  |y3| in [0, 200]
  y4 = 0, y5 = 0 (固定)

二维扫描 x_min 和 y_min，输出目标函数最小值矩阵
"""

import sys

import numpy as np
from scipy.optimize import minimize


def dist(ax, ay, bx, by):
    return np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def compute_obj(x1, y1, x2, y2, x3, y3, x4):
    d14 = dist(x1, y1, x4, 0)
    d24 = dist(x2, y2, x4, 0)
    d35 = dist(x3, y3, 200, 0)
    d13 = dist(x1, y1, x3, y3)
    d25 = dist(x2, y2, 200, 0)
    return d14 + d24 + d35 - d13 - d25


def solve(x_min, y_min):
    """给定 x_min, y_min，求最小目标函数值（多结构启发式 + 随机搜索）"""
    u_lo = float(y_min)
    u_hi = 200.0
    x_lo = float(x_min)
    u_mid = (u_lo + u_hi) / 2 if u_lo < u_hi else u_lo

    bounds = [
        (x_lo, 400),    # x1
        (x_lo, 400),    # x2
        (0, 200),       # x3
        (0, 200),       # x4
        (u_lo, u_hi),   # u1 = |y1|
        (u_lo, u_hi),   # u2 = |y2|
        (-200, 200),    # y3
    ]

    def objective(vars, s1, s2):
        x1, x2, x3, x4, u1, u2, y3 = vars
        y1 = s1 * u1
        y2 = s2 * u2
        return compute_obj(x1, y1, x2, y2, x3, y3, x4)

    # 多种启发式结构的初始点
    u_vals = list({u_lo, u_mid, u_hi})
    starts = []

    # 结构1: p1=p2=x_min, p3=p5, p4=0  (x_min小时最优)
    for uv in u_vals:
        for x4v in [0, 100, 200]:
            starts.append([x_lo, x_lo, 200, x4v, uv, uv, 0.0])
            starts.append([x_lo, x_lo, 0,   x4v, uv, uv, 0.0])

    # 结构2: p1=x_min, p2=400 (x1最小，p2最大距p5)
    for uv in u_vals:
        for x4v in [0, 100, 200]:
            for x3v in [0, 50, 200]:
                starts.append([x_lo, 400, x3v, x4v, uv, u_lo, 0.0])
                starts.append([x_lo, 400, x3v, x4v, u_lo, uv, 0.0])

    # 结构3: p2=400, p5=(200,0), p2p5方向 (p2尽量远离p5)
    for uv in u_vals:
        starts.append([400, 400, 200, 0,   uv, uv, 0.0])
        starts.append([400, 400, 0,   0,   uv, uv, 0.0])
        starts.append([x_lo, 400, 200, 0,  uv, uv, 0.0])

    # 结构4: p1,p2同侧 y 方向相反 (s1=-1, s2=+1 时有效)
    for uv in u_vals:
        for x4v in [0, 100, 200]:
            starts.append([x_lo, x_lo, 200, x4v, uv, uv, 50.0])

    # 随机点
    rng = np.random.default_rng(int(abs(x_min * 3 + y_min * 7)) % (2 ** 31))
    for _ in range(30):
        starts.append([
            rng.uniform(x_lo, 400), rng.uniform(x_lo, 400),
            rng.uniform(0, 200), rng.uniform(0, 200),
            rng.uniform(u_lo, u_hi), rng.uniform(u_lo, u_hi),
            rng.uniform(-200, 200),
        ])

    best_val = np.inf
    for s1 in [+1, -1]:
        for s2 in [+1, -1]:
            def obj(v, _s1=s1, _s2=s2):
                return objective(v, _s1, _s2)
            for x0_raw in starts:
                x0 = np.clip(np.array(x0_raw, dtype=float),
                             [lo for lo, hi in bounds],
                             [hi for lo, hi in bounds])
                try:
                    res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                                   options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12})
                    if res.fun < best_val:
                        best_val = res.fun
                except:
                    pass
    return best_val


# =============================================
# 二维扫描参数
# x_min: -200 到 400，步长 50（13个值）
# y_min:    0 到 200，步长 25（ 9个值）
# =============================================
x_min_list = list(range(-200, 401, 50))
y_min_list = list(range(0, 201, 25))

print("=" * 100)
print("二维扫描: min ||p1p4|| + ||p2p4|| + ||p3p5|| - ||p1p3|| - ||p2p5||")
print("行: x_min (x1,x2下界, -200~400步长50)    列: y_min (|y1|,|y2|下界, 0~200步长25)")
print("x3∈[0,200], x4∈[0,200], x5=200(固定), y4=y5=0(固定)")
print("=" * 100)

col_title = "x_min\\y_min"
header = f"{col_title:>12}" + "".join(f"{y:>10}" for y in y_min_list)
sep = "-" * len(header)
print(header)
print(sep)
sys.stdout.flush()

matrix = np.full((len(x_min_list), len(y_min_list)), np.nan)

for i, x_min in enumerate(x_min_list):
    for j, y_min in enumerate(y_min_list):
        matrix[i, j] = solve(x_min, y_min)
    row = f"{x_min:>12}" + "".join(f"{matrix[i,j]:>10.3f}" for j in range(len(y_min_list)))
    print(row, flush=True)

# 相对变化矩阵
print("\n" + "=" * 100)
print("【相对基准变化量矩阵】(基准: x_min=-200, y_min=0 处的值)")
base = matrix[0, 0]
print(f"基准值 = {base:.3f}")
print(header)
print(sep)
for i, x_min in enumerate(x_min_list):
    row = f"{x_min:>12}" + "".join(f"{matrix[i,j]-base:>+10.3f}" for j in range(len(y_min_list)))
    print(row)

print("\n规律: 正值越大 => 约束越紧 => 目标函数值越高（负收益越小）")

