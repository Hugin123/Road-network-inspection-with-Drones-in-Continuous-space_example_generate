import os
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ============================================================
# 无人机参数（定死）
# ============================================================
DRONE_BATTERY    = 30000        # 无人机电池容量 30,000m
DRONE_SPEED      = 15           # 无人机速度 m/s
DRONE_ENERGY_COST = 1.5         # 无人机能源成本系数 元/kwh
DRONE_CALL_COST  = 10           # 无人机一次调用成本 元/次
DRONE_INSPECT_COEF = 1          # 无人机巡检能耗系数 kwh/m
DRONE_TRANSFER_COEF = 1         # 无人机转移能耗系数 kwh/m
DRONE_DRIVE_COST = 1000         # 无人机行驶成本系数 元/kwh
BIG_M = 10000                   # 足够大的常数 M

# ============================================================
# 坐标缩放参数：将归一化坐标映射到几十~几百米
# ============================================================
COORD_SCALE = 100             # 单位：米，路网节点坐标范围约 0~100*side

# ============================================================
# 基站位置偏移系数（相对路网包围盒）
# ============================================================
DEPOT_OFFSET = 0.15            # 基站距路网边界的偏移比例


def is_intersect(p1, p2, p3, p4):
    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    """一种情况为，(p1,p2) 是 (p3,p4)的一部分，二者重合"""
    if cross_product(p3, p1, p2) == 0 and cross_product(p4, p1, p2) == 0:
        F1 = max(min(p1[0], p2[0]), min(p3[0], p4[0])) < min(max(p1[0], p2[0]), max(p3[0], p4[0]))
        F2 = max(min(p1[0], p2[0]), min(p3[0], p4[0])) == min(max(p1[0], p2[0]), max(p3[0], p4[0]))
        F3 = max(min(p1[1], p2[1]), min(p3[1], p4[1])) < min(max(p1[1], p2[1]), max(p3[1], p4[1]))
        F4 = max(min(p1[1], p2[1]), min(p3[1], p4[1])) == min(max(p1[1], p2[1]), max(p3[1], p4[1]))
        if (F1 and F3) or (F1 and F4) or (F2 and F3):
            return True
        else:
            return False

    """另一种情况为： (p1,p2) 与 (p3,p4) 是否在非端点处相交"""
    if (max(p1[0], p2[0]) < min(p3[0], p4[0]) or max(p3[0], p4[0]) < min(p1[0], p2[0]) or
            max(p1[1], p2[1]) < min(p3[1], p4[1]) or max(p3[1], p4[1]) < min(p1[1], p2[1])):
        return False

    cp1 = cross_product(p1, p2, p3)
    cp2 = cross_product(p1, p2, p4)
    cp3 = cross_product(p3, p4, p1)
    cp4 = cross_product(p3, p4, p2)
    return (cp1 * cp2 < 0) and (cp3 * cp4 < 0)


def can_add_edge(G, pos, u, v):
    """碰撞检测：检查新边是否与现有边交叉"""
    if u == v or G.has_edge(u, v): return False
    p1, p2 = pos[u], pos[v]
    for edge in G.edges():
        p3, p4 = pos[edge[0]], pos[edge[1]]
        if is_intersect(p1, p2, p3, p4): return False
    return True


def generate_strict_road_network(mode, target_nodes, target_edges):
    """生成平面路网，返回图 G 和坐标字典 pos（原始坐标，不归一化）"""
    G = nx.Graph()
    pos = {}

    # --- 1. 节点生成阶段 ---
    if mode == "grid":
        side = int(np.ceil(np.sqrt(target_nodes)))
        for i in range(target_nodes):
            pos[i] = np.array([float(i % side), float(i // side)])
    elif mode == "radial":
        pos[0] = np.array([0.0, 0.0])
        num_arms = 8
        nodes_per_arm = (target_nodes - 1) // num_arms
        idx = 1
        for arm in range(num_arms):
            angle = np.radians(arm * (360 / num_arms))
            for r in range(1, nodes_per_arm + 2):
                if idx < target_nodes:
                    pos[idx] = np.array([float(r * np.cos(angle)), float(r * np.sin(angle))])
                    idx += 1
        while idx < target_nodes:
            pos[idx] = np.random.uniform(-nodes_per_arm, nodes_per_arm, 2)
            idx += 1
    elif mode == "linear":
        main_len = target_nodes // 3
        for i in range(main_len): pos[i] = np.array([float(i), 0.0])
        idx = main_len
        while idx < target_nodes:
            col = random.randint(0, main_len - 1)
            side = 1 if idx % 2 == 0 else -1
            h = sum(1 for n in range(idx) if pos[n][0] == col and pos[n][1] != 0) + 1
            pos[idx] = np.array([float(col), float(h * side)])
            idx += 1
    else:  # random
        for i in range(target_nodes):
            pos[i] = np.random.rand(2) * 10

    for i in range(target_nodes): G.add_node(i)

    # --- 2. 基础骨架连接 ---
    nodes_list = list(G.nodes())
    for i in range(target_nodes):
        for j in range(i + 1, target_nodes):
            dist = np.linalg.norm(pos[i] - pos[j])
            if dist <= 1.1:
                if can_add_edge(G, pos, i, j):
                    G.add_edge(i, j)

    # --- 3. 动态补边 ---
    attempts = 0
    while G.number_of_edges() < target_edges and attempts < 2000:
        u, v = random.sample(nodes_list, 2)
        if np.linalg.norm(pos[u] - pos[v]) < 4.0:
            if can_add_edge(G, pos, u, v):
                G.add_edge(u, v)
        attempts += 1

    return G, pos


def compute_depot_pos(pos, direction):
    """
    根据方向计算基站坐标（位于路网包围盒外侧）。
    direction: 'up', 'down', 'left', 'right', 'center'
    返回 np.array([x, y])（与原始坐标同单位）
    """
    coords = np.array(list(pos.values()))
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    # 偏移量：路网跨度的 DEPOT_OFFSET 倍，至少 0.5 个单位
    x_span = max((x_max - x_min) * DEPOT_OFFSET, 250)
    y_span = max((y_max - y_min) * DEPOT_OFFSET, 250)

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


def save_network_figure(fig_path, G, pos, all_depot_positions):
    """
    将路网与所有基站位置绘制成一张图并保存为 PNG。
    all_depot_positions: [(depot_id, direction_label, depot_pos), ...]
    """
    # 构建 1-indexed 节点标签
    labels = {i: str(i + 1) for i in G.nodes()}
    # pos 字典转为 tuple（matplotlib 兼容）
    pos_tuple = {k: (v[0], v[1]) for k, v in pos.items()}

    fig, ax = plt.subplots(figsize=(7, 7))

    # 画路网
    nx.draw_networkx_edges(G, pos_tuple, ax=ax, edge_color='#444444', width=1.5)
    nx.draw_networkx_nodes(G, pos_tuple, ax=ax, node_color='#5B9BD5',
                           node_size=350, linewidths=0.8, edgecolors='white')
    nx.draw_networkx_labels(G, pos_tuple, labels=labels, ax=ax,
                            font_size=8, font_color='white', font_weight='bold')

    # 各基站方向的颜色与标签
    depot_colors = {1: '#E74C3C', 2: '#2ECC71', 3: '#F39C12', 4: '#9B59B6', 5: '#1ABC9C'}
    depot_labels = {1: 'up', 2: 'down', 3: 'left', 4: 'right', 5: 'center'}

    for depot_id, _direction, depot_pos in all_depot_positions:
        dx, dy = float(depot_pos[0]), float(depot_pos[1])
        color = depot_colors[depot_id]
        ax.scatter(dx, dy, marker='*', s=300, c=color, zorder=5,
                   label=f'depot-{depot_labels[depot_id]}', edgecolors='white', linewidths=0.5)
        ax.annotate(f'D{depot_id}', xy=(dx, dy), xytext=(4, 4),
                    textcoords='offset points', fontsize=7, color=color, fontweight='bold')

    ax.legend(loc='best', fontsize=7, framealpha=0.8)
    ax.set_aspect('equal', adjustable='datalim')
    ax.margins(0.2)
    ax.set_title(os.path.splitext(os.path.basename(fig_path))[0], fontsize=9)
    ax.axis('on')
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存图: {fig_path}")


def save_instance(filepath, G, pos, depot_pos, num_drones):
    """
    将算例写入 txt 文件。
    节点编号规则：基站=0，路网节点 1..N（1-indexed）。
    坐标列表顺序：先列路网节点（1-indexed顺序），最后是基站（0）。
    需求弧：所有路网边，端点编号 1-indexed。
    """
    num_road_nodes = G.number_of_nodes()     # 路网节点数（不含基站）
    total_nodes = num_road_nodes + 1          # 含基站
    num_edges = G.number_of_edges()           # 需求边数 = 总边数

    # 构建含基站的完整坐标字典（基站索引=0）
    all_pos = {0: depot_pos}
    for road_node in range(num_road_nodes):
        all_pos[road_node + 1] = pos[road_node]   # 1-indexed，直接使用原始坐标

    # 坐标列表：先路网节点（1..N），最后基站（0）
    x_list = [all_pos[i][0] for i in range(1, total_nodes)] + [all_pos[0][0]]
    y_list = [all_pos[i][1] for i in range(1, total_nodes)] + [all_pos[0][1]]

    # 需求弧列表（所有路网边，端点 1-indexed）
    edges = list(G.edges())

    with open(filepath, 'w', encoding='utf-8') as f:
        # 基本信息
        f.write(f"1\n")                        # 基站点数量
        f.write(f"{num_road_nodes}\n")         # 端点数量（路网节点数，不含基站）
        f.write(f"{total_nodes}\n")            # 节点数量（含基站）
        f.write(f"{num_edges}\n")              # 边数量（需求边数）
        f.write(f"{num_drones}\n")             # 无人机数量
        f.write(f"\n")
        # 无人机参数
        f.write(f"{DRONE_BATTERY}\n")
        f.write(f"{DRONE_SPEED}\n")
        f.write(f"{DRONE_ENERGY_COST}\n")
        f.write(f"{DRONE_CALL_COST}\n")
        f.write(f"{DRONE_INSPECT_COEF}\n")
        f.write(f"{DRONE_TRANSFER_COEF}\n")
        f.write(f"{DRONE_DRIVE_COST}\n")
        f.write(f"{BIG_M}\n")
        f.write(f"\n")
        # 坐标
        f.write(", ".join(str(x) for x in x_list) + "\n")
        f.write(", ".join(str(y) for y in y_list) + "\n")
        f.write(f"\n")
        # 需求弧（首行固定为基站自环 (0,0)，不计入边数量；其余 1-indexed）
        f.write(f"(0,0)\n")
        for u, v in edges:
            f.write(f"({u + 1},{v + 1})\n")

    print(f"  已保存: {filepath}")


# ============================================================
# 批量生成配置
# ============================================================
# --- 输出开关（测试时可关闭，不生成任何文件）---
SAVE_TXT = True    # 是否生成 .txt 算例文件
SAVE_PNG = True    # 是否生成 .png 可视化图

# 坐标缩放倍数（路网原始坐标以1为边长，乘以此值转换为米）
COORD_MULTIPLIER = 250

# 无人机数量固定
NUM_DRONES = 2

# 每组 (路网节点数, 目标边数)
SMALL_MEDIUM_CONFIGS = [
    # (road_nodes, edges)
    (5,  4),
    (6,  5),
    (7,  6),
    (7,  7),
    (7,  8),
]

MODES = ["grid"]   # 小中规模使用 grid 模式

# 基站方向标签
DEPOT_DIRECTIONS = [
    (1, 'up'),
    (2, 'down'),
    (3, 'left'),
    (4, 'right'),
    (5, 'center'),
]

# 每组配置生成的随机实例数
NUM_INSTANCES_PER_CONFIG = 3

# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "算例", "小中规模算例")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 主生成逻辑
# ============================================================
if __name__ == "__main__":
    total_saved = 0
    total_figs  = 0
    for mode in MODES:
        for (road_nodes, edges) in SMALL_MEDIUM_CONFIGS:
            for inst_idx in range(NUM_INSTANCES_PER_CONFIG):
                print(f"\n生成路网: mode={mode}, nodes={road_nodes}, edges={edges}, 实例#{inst_idx}")
                G, pos = generate_strict_road_network(mode, road_nodes, edges)
                actual_edges = G.number_of_edges()
                if actual_edges == 0:
                    print(f"  警告：路网边数为0，跳过。")
                    continue

                # 坐标乘以缩放倍数（路网生成后再缩放，不影响码冲检测）
                scaled_pos = {k: v * COORD_MULTIPLIER for k, v in pos.items()}

                total_nodes = road_nodes + 1

                # 收集5种基站位置，用于统一绘图
                all_depot_positions = []
                for depot_id, direction in DEPOT_DIRECTIONS:
                    depot_pos = compute_depot_pos(scaled_pos, direction)
                    all_depot_positions.append((depot_id, direction, depot_pos))

                    if SAVE_TXT:
                        filename = (f"{total_nodes}-{actual_edges}-{NUM_DRONES}"
                                    f"-{depot_id}-({inst_idx}).txt")
                        filepath = os.path.join(OUTPUT_DIR, filename)
                        save_instance(filepath, G, scaled_pos, depot_pos, NUM_DRONES)
                        total_saved += 1

                if SAVE_PNG:
                    fig_name = f"{total_nodes}-{actual_edges}-{NUM_DRONES}-({inst_idx}).png"
                    fig_path = os.path.join(OUTPUT_DIR, fig_name)
                    save_network_figure(fig_path, G, scaled_pos, all_depot_positions)
                    total_figs += 1

    print(f"\n全部完成，共保存 {total_saved} 个算例文件、{total_figs} 张路网图到: {OUTPUT_DIR}")
