import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random


def is_intersect(p1, p2, p3, p4):
    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    """一种情况为，(p1,p2) 是 (p3,p4)的一部分，二者重合"""
    # 先判断是否共线
    if cross_product(p3, p1, p2) == 0 and cross_product(p4, p1, p2) == 0:
        # 再判断是否有重合区域
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
        i = 0
        while i < target_nodes:
            pos[i] = np.random.rand(2) * 10
            add_flag = True
            for ind in range(i):
                if np.linalg.norm(pos[i] - pos[ind]) <= 0.8:
                    add_flag = False
                    break
            if add_flag:
                i += 1

    for i in range(target_nodes): G.add_node(i)

    # --- 2. 基础骨架连接 ---
    nodes_list = list(G.nodes())
    for i in range(target_nodes):
        for j in range(i + 1, target_nodes):
            dist = np.linalg.norm(pos[i] - pos[j])
            if 0.5 <= dist <= 1.1:
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


# --- 执行、输出与可视化 ---
# modes = ["grid", "radial", "linear", "random"]
modes = ["random"]
N, E = 30, 60  # 为了展示清晰，这里调小了数值，您可以自行改回 50, 100

for m in modes:
    print(f"\n{'=' * 20} MODE: {m.upper()} {'=' * 20}")
    G_temp, pos_temp = generate_strict_road_network(m, N, E)

    # print("--- 节点坐标 (编号: [x, y]) ---")
    # for node, coord in pos_temp.items():
    #     print(f"{node}: [{coord[0]:.2f}, {coord[1]:.2f}]")
    #
    # print("\n--- 边列表 (u, v) ---")
    # edges_list = list(G_temp.edges())
    # for edge in edges_list:
    #     print(f"({edge[0]}, {edge[1]})")
    # for edge in edges_list:
    #     G_tmp = G_temp.copy()
    #     G_tmp.remove_edge(*edge)
    #     print(f"({edge[0]}, {edge[1]}) : {can_add_edge(G_tmp, pos_temp, edge[0], edge[1])}")


    fig, axes = plt.subplots(1, 1, figsize=(8, 8))
    nx.draw(G_temp, pos_temp, ax=axes, with_labels=True, node_size=300, node_color='skyblue', font_size=8)
    axes.set_title(f"Mode: {m.upper()}")
    plt.show()

# 可视化部分（可选）
# fig, axes = plt.subplots(1, 4, figsize=(20, 5))
# for i, m in enumerate(modes):
#     G_temp, pos_temp = generate_strict_road_network(m, N, E)
#     nx.draw(G_temp, pos_temp, ax=axes[i], with_labels=True, node_size=300, node_color='skyblue', font_size=8)
#     axes[i].set_title(f"Mode: {m.upper()}")
# plt.show()