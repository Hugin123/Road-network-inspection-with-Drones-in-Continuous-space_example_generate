import os
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ============================================================
# 无人机参数（定死）
# ============================================================
DRONE_BATTERY     = 30000     # 无人机电池容量 m
DRONE_SPEED       = 15        # 无人机速度 m/s
DRONE_ENERGY_COST = 1.5       # 无人机能源成本系数 元/kwh
DRONE_CALL_COST   = 10        # 无人机一次调用成本 元/次
DRONE_INSPECT_COEF  = 1       # 无人机巡检能耗系数 kwh/m
DRONE_TRANSFER_COEF = 1       # 无人机转移能耗系数 kwh/m
BIG_M = 10000                 # 足够大的常数 M

# ============================================================
# 路网空间约束：所有节点坐标需落在 5km×5km 范围内
# 生成时坐标为归一化单位，生成后缩放到实际米制
# ============================================================
AREA_SIZE_M  = 5000.0          # 路网覆盖区域边长（米）

# 基站偏移：路网包围盒跨度的比例，且至少 250m
DEPOT_OFFSET_RATIO = 0.15
DEPOT_OFFSET_MIN_M = 250.0


# ============================================================
# 碰撞检测工具函数
# ============================================================
def is_intersect(p1, p2, p3, p4):
    """判断线段(p1,p2)与(p3,p4)是否相交（含共线重叠）"""
    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # 共线情况：检测是否有重叠
    if cross_product(p3, p1, p2) == 0 and cross_product(p4, p1, p2) == 0:
        F1 = max(min(p1[0], p2[0]), min(p3[0], p4[0])) < min(max(p1[0], p2[0]), max(p3[0], p4[0]))
        F2 = max(min(p1[0], p2[0]), min(p3[0], p4[0])) == min(max(p1[0], p2[0]), max(p3[0], p4[0]))
        F3 = max(min(p1[1], p2[1]), min(p3[1], p4[1])) < min(max(p1[1], p2[1]), max(p3[1], p4[1]))
        F4 = max(min(p1[1], p2[1]), min(p3[1], p4[1])) == min(max(p1[1], p2[1]), max(p3[1], p4[1]))
        return (F1 and F3) or (F1 and F4) or (F2 and F3)

    # 普通相交：AABB 剪裁 + 叉积判断
    if (max(p1[0], p2[0]) < min(p3[0], p4[0]) or max(p3[0], p4[0]) < min(p1[0], p2[0]) or
            max(p1[1], p2[1]) < min(p3[1], p4[1]) or max(p3[1], p4[1]) < min(p1[1], p2[1])):
        return False
    cp1 = cross_product(p1, p2, p3)
    cp2 = cross_product(p1, p2, p4)
    cp3 = cross_product(p3, p4, p1)
    cp4 = cross_product(p3, p4, p2)
    return (cp1 * cp2 < 0) and (cp3 * cp4 < 0)


def can_add_edge(G, pos, u, v):
    """碰撞检测：新边(u,v)与已有边不相交才可添加"""
    if u == v or G.has_edge(u, v):
        return False
    p1, p2 = pos[u], pos[v]
    for edge in G.edges():
        p3, p4 = pos[edge[0]], pos[edge[1]]
        if is_intersect(p1, p2, p3, p4):
            return False
    return True


# ============================================================
# 各模式路网节点生成
# ============================================================
def _gen_nodes_grid_grow(target_nodes, grid_size):
    """
    用生长法在 grid_size×grid_size 背景网格上选取节点：
    - 从一个随机种子出发，每步以 80% 概率选取离当前节点集最近的候选格点，
      20% 概率在距离 ≤3 的候选格点中随机跳一步（增加路网稀疏感）。
    - 保证所选节点整体聚集、相邻，形成自然的网格团。
    返回 {node_id: np.array([col, row])} 格点坐标（整数值）。
    """
    all_points = set((col, row)
                     for row in range(grid_size)
                     for col in range(grid_size))

    # 随机选种子
    seed = random.choice(list(all_points))
    chosen = [seed]
    all_points.discard(seed)

    while len(chosen) < target_nodes and all_points:
        # 计算所有候选格点到当前节点集的最小曼哈顿距离
        min_dist = {}
        for (cc, cr) in all_points:
            d = min(abs(cc - sc) + abs(cr - sr) for (sc, sr) in chosen)
            min_dist[(cc, cr)] = d

        r = random.random()
        if r < 0.70:
            # 70%：选距离最近的候选（若有并列则随机选一个），确保紧密聚集
            min_d = min(min_dist.values())
            nearest = [p for p, d in min_dist.items() if d == min_d]
            pick = random.choice(nearest)
        elif r < 0.90:
            # 20%：在距离 ≤2 的候选中随机选（小跳跃，延伸到相邻格）
            close = [p for p, d in min_dist.items() if d <= 2]
            if close:
                pick = random.choice(close)
            else:
                min_d = min(min_dist.values())
                nearest = [p for p, d in min_dist.items() if d == min_d]
                pick = random.choice(nearest)
        else:
            # 10%：在距离 ≤4 的候选中随机选（较大跳跃，保持路网延伸感）
            medium = [p for p, d in min_dist.items() if d <= 4]
            if medium:
                pick = random.choice(medium)
            else:
                pick = random.choice(list(all_points))

        chosen.append(pick)
        all_points.discard(pick)

    return {i: np.array([float(c), float(r)]) for i, (c, r) in enumerate(chosen)}


def _build_grid_edges(pos, target_nodes, target_edges):
    """
    基于生长法生成的格点坐标，构建以横竖边为主、允许短斜边的网格路网：
    连边优先级：
      1. 正交相邻边（曼哈顿距离=1，水平/垂直）
      2. 正交延伸边（同行/同列，距离=2，跨一格）
      3. 短对角边（欧氏距离 ≤ √5 ≈ 2.24，对应 (1,1)/(1,2)/(2,1) 偏移）
      4. 保底：任意无交叉边（极端稀疏时兜底）
    削边：若超出目标，优先删除长边，保留短边。
    """
    G = nx.Graph()
    for i in range(target_nodes):
        G.add_node(i)

    coord_to_node = {(int(pos[i][0]), int(pos[i][1])): i for i in range(target_nodes)}

    # 预计算所有节点对的欧氏距离和格点曼哈顿距离
    def euclid(i, j):
        return np.linalg.norm(pos[i] - pos[j])

    # --- 阶段1：正交相邻边（距离=1，水平/垂直）---
    for i in range(target_nodes):
        col, row = int(pos[i][0]), int(pos[i][1])
        for dcol, drow in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nb = coord_to_node.get((col + dcol, row + drow))
            if nb is not None and nb > i:
                if can_add_edge(G, pos, i, nb):
                    G.add_edge(i, nb)

    # --- 阶段2：同行/同列正交延伸（曼哈顿距离2或3，跨一格或两格）---
    if G.number_of_edges() < target_edges:
        # 收集所有正交对（同行或同列），按距离升序添加
        ortho_ext = []
        for i in range(target_nodes):
            ci, ri = int(pos[i][0]), int(pos[i][1])
            for j in range(i + 1, target_nodes):
                cj, rj = int(pos[j][0]), int(pos[j][1])
                if ci == cj or ri == rj:
                    d = abs(ci - cj) + abs(ri - rj)
                    if d > 1:   # 相邻正交边已在阶段1添加
                        ortho_ext.append((d, i, j))
        ortho_ext.sort()
        for d, i, j in ortho_ext:
            if G.number_of_edges() >= target_edges:
                break
            if can_add_edge(G, pos, i, j):
                G.add_edge(i, j)

    # --- 阶段3：短对角边（欧氏距离 ≤ √5，偏移 (±1,±1)/(±1,±2)/(±2,±1)）---
    if G.number_of_edges() < target_edges:
        # 收集所有短对角候选对，按欧氏距离排序
        short_diag = []
        for i in range(target_nodes):
            for j in range(i + 1, target_nodes):
                d = euclid(i, j)
                if 1.01 < d <= 2.24:   # 排除正交边，保留对角短边
                    short_diag.append((d, i, j))
        short_diag.sort()
        for d, i, j in short_diag:
            if G.number_of_edges() >= target_edges:
                break
            if can_add_edge(G, pos, i, j):
                G.add_edge(i, j)

    # --- 阶段4：保底补边（仅限欧氏距离 ≤ 3 格，避免超长斜边）---
    if G.number_of_edges() < target_edges:
        medium_pairs = []
        for i in range(target_nodes):
            for j in range(i + 1, target_nodes):
                d = euclid(i, j)
                if 2.24 < d <= 3.0:
                    medium_pairs.append((d, i, j))
        medium_pairs.sort()
        for d, i, j in medium_pairs:
            if G.number_of_edges() >= target_edges:
                break
            if can_add_edge(G, pos, i, j):
                G.add_edge(i, j)

    # --- 阶段5：极端保底（所有剩余节点对按距离从小到大排序，允许最长4格）---
    if G.number_of_edges() < target_edges:
        all_pairs = []
        for i in range(target_nodes):
            for j in range(i + 1, target_nodes):
                d = euclid(i, j)
                if d <= 4.0 and not G.has_edge(i, j):
                    all_pairs.append((d, i, j))
        all_pairs.sort()
        for d, i, j in all_pairs:
            if G.number_of_edges() >= target_edges:
                break
            if can_add_edge(G, pos, i, j):
                G.add_edge(i, j)

    # --- 削边：若超出目标，优先删除长边 ---
    while G.number_of_edges() > target_edges:
        # 找出最长的边删除（保留短边）
        longest = max(G.edges(), key=lambda e: euclid(e[0], e[1]))
        G.remove_edge(*longest)

    return G


def _gen_nodes_radial(target_nodes):
    """
    放射状布局：以原点为中心，8条射线向外延伸。
    坐标范围约在 [-nodes_per_arm, nodes_per_arm]²
    """
    pos = {0: np.array([0.0, 0.0])}
    num_arms = 8
    nodes_per_arm = max(1, (target_nodes - 1) // num_arms)
    idx = 1
    for arm in range(num_arms):
        angle = np.radians(arm * (360.0 / num_arms))
        for r in range(1, nodes_per_arm + 2):
            if idx < target_nodes:
                pos[idx] = np.array([float(r * np.cos(angle)),
                                     float(r * np.sin(angle))])
                idx += 1
    # 剩余节点随机补充
    while idx < target_nodes:
        pos[idx] = np.random.uniform(-nodes_per_arm, nodes_per_arm, 2)
        idx += 1
    return pos


def _gen_nodes_linear(target_nodes):
    """
    鱼骨型布局：主干沿 x 轴，支干交替向上/向下挂载。
    主干长度约为总节点数的 1/3。
    """
    main_len = max(2, target_nodes // 3)
    pos = {i: np.array([float(i), 0.0]) for i in range(main_len)}
    idx = main_len
    while idx < target_nodes:
        col = random.randint(0, main_len - 1)
        sign = 1 if idx % 2 == 0 else -1
        h = sum(1 for n in range(idx) if pos[n][0] == col and pos[n][1] != 0) + 1
        pos[idx] = np.array([float(col), float(h * sign)])
        idx += 1
    return pos


def _gen_nodes_random(target_nodes, area=10.0, min_dist=0.5):
    """
    随机散点布局：在 [0, area]² 内生成节点，强制最小间距 min_dist。
    """
    pos = {}
    i = 0
    max_attempts = target_nodes * 500
    attempts = 0
    while i < target_nodes and attempts < max_attempts:
        candidate = np.random.rand(2) * area
        ok = all(np.linalg.norm(candidate - pos[k]) > min_dist for k in pos)
        if ok:
            pos[i] = candidate
            i += 1
        attempts += 1
    # 若强制间距下放不下，放宽间距补齐
    while i < target_nodes:
        pos[i] = np.random.rand(2) * area
        i += 1
    return pos


# ============================================================
# 路网生成主函数（含重试机制）
# ============================================================
def generate_road_network(mode, target_nodes, target_edges,
                          grid_size=None, max_retries=100):
    """
    生成满足 (target_nodes, target_edges) 的平面路网。
    - 节点数必须严格等于 target_nodes
    - 边数必须严格等于 target_edges
    - 不满足则重新生成，最多重试 max_retries 次
    - 所有模式生成后统一缩放到 AREA_SIZE_M×AREA_SIZE_M（米）范围内

    返回: (G, scaled_pos)，坐标单位已是"米"
    """
    for attempt in range(max_retries):
        G = nx.Graph()
        pos = {}

        # --- 1. 节点生成 ---
        if mode == "grid":
            gs = grid_size if grid_size is not None else max(8, int(np.ceil(np.sqrt(target_nodes * 2))))
            pos = _gen_nodes_grid_grow(target_nodes, gs)
        elif mode == "radial":
            pos = _gen_nodes_radial(target_nodes)
        elif mode == "linear":
            pos = _gen_nodes_linear(target_nodes)
        else:  # random
            pos = _gen_nodes_random(target_nodes)

        if len(pos) != target_nodes:
            continue

        # --- grid 模式：专用正交连边逻辑 ---
        if mode == "grid":
            G = _build_grid_edges(pos, target_nodes, target_edges)
        else:
            # --- 非 grid 模式：通用骨架 + 补边/削边逻辑 ---
            for i in range(target_nodes):
                G.add_node(i)

            nodes_list = list(G.nodes())
            base_thresh = 1.5

            for i in range(target_nodes):
                for j in range(i + 1, target_nodes):
                    dist = np.linalg.norm(pos[i] - pos[j])
                    if dist <= base_thresh:
                        if can_add_edge(G, pos, i, j):
                            G.add_edge(i, j)

            # 削边：若骨架边数已超过目标，随机删除多余边
            while G.number_of_edges() > target_edges:
                edge_to_remove = random.choice(list(G.edges()))
                G.remove_edge(*edge_to_remove)

            # 补边：若边数不足目标，随机添加无交叉边
            sub_attempts = 0
            while G.number_of_edges() < target_edges and sub_attempts < 10000:
                u, v = random.sample(nodes_list, 2)
                if can_add_edge(G, pos, u, v):
                    G.add_edge(u, v)
                sub_attempts += 1

        # --- 4. 校验约束 ---
        if G.number_of_nodes() == target_nodes and G.number_of_edges() == target_edges:
            # 缩放坐标到 AREA_SIZE_M×AREA_SIZE_M 范围内
            coords = np.array([pos[k] for k in range(target_nodes)])
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            x_span = x_max - x_min if x_max > x_min else 1.0
            y_span = y_max - y_min if y_max > y_min else 1.0
            scale = min(AREA_SIZE_M / x_span, AREA_SIZE_M / y_span) * 0.9  # 留 10% 边距
            scaled_pos = {}
            for k in range(target_nodes):
                sx = (pos[k][0] - x_min) * scale
                sy = (pos[k][1] - y_min) * scale
                scaled_pos[k] = np.array([sx, sy])
            print(f"    [OK] mode={mode}, nodes={target_nodes}, edges={target_edges} "
                  f"(第{attempt + 1}次尝试成功)")
            return G, scaled_pos

        if attempt % 10 == 9:
            print(f"    [重试] mode={mode}, nodes={target_nodes}, target_edges={target_edges}, "
                  f"当前edges={G.number_of_edges()}, 已重试{attempt + 1}次...")

    raise RuntimeError(
        f"无法在 {max_retries} 次内生成满足约束的路网: "
        f"mode={mode}, nodes={target_nodes}, edges={target_edges}"
    )


# ============================================================
# 基站坐标计算
# ============================================================
def compute_depot_pos(scaled_pos, direction):
    """
    计算基站坐标（位于路网包围盒外侧）。
    direction: 'up', 'down', 'left', 'right', 'center'
    坐标单位：米
    """
    coords = np.array(list(scaled_pos.values()))
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    x_span = max((x_max - x_min) * DEPOT_OFFSET_RATIO, DEPOT_OFFSET_MIN_M)
    y_span = max((y_max - y_min) * DEPOT_OFFSET_RATIO, DEPOT_OFFSET_MIN_M)

    if direction == 'up':
        return np.array([cx, y_max + y_span])
    elif direction == 'down':
        return np.array([cx, y_min - y_span])
    elif direction == 'left':
        return np.array([x_min - x_span, cy])
    elif direction == 'right':
        return np.array([x_max + x_span, cy])
    else:  # center
        return np.array([cx, cy])


# ============================================================
# 可视化绘图
# ============================================================
def save_network_figure(fig_path, G, scaled_pos, all_depot_positions, mode=None):
    """
    将路网与所有基站位置绘制成一张图并保存为 PNG。
    all_depot_positions: [(depot_id, direction_label, depot_pos), ...]
    """
    labels = {i: str(i + 1) for i in G.nodes()}
    pos_tuple = {k: (v[0], v[1]) for k, v in scaled_pos.items()}

    fig, ax = plt.subplots(figsize=(9, 9))

    nx.draw_networkx_edges(G, pos_tuple, ax=ax, edge_color='#444444', width=1.2)
    nx.draw_networkx_nodes(G, pos_tuple, ax=ax, node_color='#27AE60',
                           node_size=200, linewidths=0.8, edgecolors='white')
    nx.draw_networkx_labels(G, pos_tuple, labels=labels, ax=ax,
                            font_size=5, font_color='white', font_weight='bold')

    depot_colors = {1: '#E74C3C', 2: '#2ECC71', 3: '#F39C12', 4: '#9B59B6', 5: '#1ABC9C'}
    depot_label_map = {1: 'up', 2: 'down', 3: 'left', 4: 'right', 5: 'center'}

    # 收集所有坐标（路网节点 + 所有基站）用于动态计算显示范围
    all_x = [v[0] for v in scaled_pos.values()]
    all_y = [v[1] for v in scaled_pos.values()]

    for depot_id, _direction, depot_pos in all_depot_positions:
        dx, dy = float(depot_pos[0]), float(depot_pos[1])
        all_x.append(dx)
        all_y.append(dy)
        color = depot_colors[depot_id]
        ax.scatter(dx, dy, marker='*', s=400, c=color, zorder=5,
                   label=f'depot-{depot_label_map[depot_id]}',
                   edgecolors='white', linewidths=0.5)
        ax.annotate(f'D{depot_id}', xy=(dx, dy), xytext=(5, 5),
                    textcoords='offset points', fontsize=8, color=color, fontweight='bold')

    # 绘制 5km×5km 路网边界框（虚线，仅供参考）
    rect = plt.Rectangle((0, 0), AREA_SIZE_M, AREA_SIZE_M,
                          linewidth=1.5, edgecolor='#AAAAAA', facecolor='none',
                          linestyle='--', zorder=0)
    ax.add_patch(rect)

    # 动态设置坐标范围，保证所有路网节点和基站都在视图内，加 5% padding
    x_min_all, x_max_all = min(all_x), max(all_x)
    y_min_all, y_max_all = min(all_y), max(all_y)
    x_pad = (x_max_all - x_min_all) * 0.05 + 100
    y_pad = (y_max_all - y_min_all) * 0.05 + 100
    ax.set_xlim(x_min_all - x_pad, x_max_all + x_pad)
    ax.set_ylim(y_min_all - y_pad, y_max_all + y_pad)

    ax.legend(loc='upper right', fontsize=7, framealpha=0.8)
    ax.set_aspect('equal', adjustable='box')
    title = os.path.splitext(os.path.basename(fig_path))[0]
    ax.set_title(f"{title}  [mode={mode}]" if mode else title, fontsize=9)
    ax.axis('on')
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存图: {fig_path}")


# ============================================================
# 算例文件写入
# ============================================================
def save_instance(filepath, G, scaled_pos, depot_pos, num_drones):
    """
    将算例写入 txt 文件。
    节点编号：基站=0，路网节点 1..N（1-indexed）。
    """
    num_road_nodes = G.number_of_nodes()
    total_nodes = num_road_nodes + 1
    num_edges = G.number_of_edges()

    all_pos = {0: depot_pos}
    for k in range(num_road_nodes):
        all_pos[k + 1] = scaled_pos[k]

    x_list = [all_pos[0][0]] + [all_pos[i][0] for i in range(1, total_nodes)]
    y_list = [all_pos[0][1]] + [all_pos[i][1] for i in range(1, total_nodes)]
    edges = list(G.edges())

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"1\n")                      # 基站数量
        f.write(f"{num_road_nodes}\n")        # 路网节点数（不含基站）
        f.write(f"{total_nodes}\n")           # 总节点数（含基站）
        f.write(f"{num_edges}\n")             # 需求边数
        f.write(f"{num_drones}\n")            # 无人机数量
        f.write(f"\n")
        f.write(f"{DRONE_BATTERY}\n")
        f.write(f"{DRONE_SPEED}\n")
        f.write(f"{DRONE_ENERGY_COST}\n")
        f.write(f"{DRONE_CALL_COST}\n")
        f.write(f"{DRONE_INSPECT_COEF}\n")
        f.write(f"{DRONE_TRANSFER_COEF}\n")
        f.write(f"{BIG_M}\n")
        f.write(f"\n")
        f.write(", ".join(str(x) for x in x_list) + "\n")
        f.write(", ".join(str(y) for y in y_list) + "\n")
        f.write(f"\n")
        f.write(f"(0,0)\n")                  # 基站自环（固定首行，不计入边数）
        for u, v in edges:
            f.write(f"({u + 1},{v + 1})\n")

    print(f"  已保存: {filepath}")


# ============================================================
# 批量生成配置
# ============================================================
SAVE_TXT = True
SAVE_PNG = True

# 无人机数量
NUM_DRONES = 10

# 每种模式每组配置生成的算例数
NUM_INSTANCES_PER_MODE = 2

# 使用的模式列表
MODES = ["grid", "radial", "linear", "random"]

# grid 模式大规模使用 15×15 背景网格（225个候选格点）
GRID_SIZE_LARGE = 15

# 大规模算例配置：(路网节点数, 目标边数列表)
# 节点25: 30边、40边
# 节点30: 35边、45边
# 节点35: 50边、60边
# 节点40: 60边、70边
# 节点45: 70边、80边
# 节点50: 80边、100边
LARGE_CONFIGS = [
    (25, 30),
    (25, 40),
    (30, 35),
    (30, 45),
    (35, 50),
    (35, 60),
    (40, 60),
    (40, 70),
    (45, 70),
    (45, 80),
    (50, 80),
    (50, 100),
]

DEPOT_DIRECTIONS = [
    (1, 'up'),
    (2, 'down'),
    (3, 'left'),
    (4, 'right'),
    (5, 'center'),
]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "算例", "随机算例", "2-Large")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 主生成逻辑
# ============================================================
if __name__ == "__main__":
    total_saved = 0
    total_figs  = 0

    for mode in MODES:
        for (road_nodes, target_edges) in LARGE_CONFIGS:
            for inst_idx in range(NUM_INSTANCES_PER_MODE):
                print(f"\n[大规模] mode={mode}, nodes={road_nodes}, "
                      f"edges={target_edges}, 实例#{inst_idx}")

                # 生成满足约束的路网（不满足则内部自动重试）
                try:
                    G, scaled_pos = generate_road_network(
                        mode, road_nodes, target_edges,
                        grid_size=GRID_SIZE_LARGE,
                        max_retries=100
                    )
                except RuntimeError as e:
                    print(f"  [跳过] {e}")
                    continue

                actual_edges = G.number_of_edges()
                total_nodes  = road_nodes + 1

                # 收集5个基站位置
                all_depot_positions = []
                for depot_id, direction in DEPOT_DIRECTIONS:
                    depot_pos = compute_depot_pos(scaled_pos, direction)
                    all_depot_positions.append((depot_id, direction, depot_pos))

                    if SAVE_TXT:
                        filename = (f"{total_nodes}-{actual_edges}-{NUM_DRONES}"
                                    f"-{depot_id}-({mode}-{inst_idx}).txt")
                        filepath = os.path.join(OUTPUT_DIR, filename)
                        save_instance(filepath, G, scaled_pos, depot_pos, NUM_DRONES)
                        total_saved += 1

                if SAVE_PNG:
                    fig_name = (f"{total_nodes}-{actual_edges}-{NUM_DRONES}"
                                f"-({mode}-{inst_idx}).png")
                    fig_path = os.path.join(OUTPUT_DIR, fig_name)
                    save_network_figure(fig_path, G, scaled_pos,
                                        all_depot_positions, mode=mode)
                    total_figs += 1

    print(f"\n全部完成：共保存 {total_saved} 个 txt 文件、{total_figs} 张 png 图")
    print(f"  输出目录: {OUTPUT_DIR}")
