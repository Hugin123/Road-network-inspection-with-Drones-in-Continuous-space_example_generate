"""
Practical_Instance_Extract.py
============================
从 OpenStreetMap 提取真实城市/农村道路路网，并生成与随机算例相同格式的 .txt 算例文件。

依赖安装（首次使用）：
    pip install osmnx networkx numpy matplotlib pyproj

使用方式：
    1. 直接运行本脚本，修改底部 __main__ 块中的参数即可。
    2. 也可作为模块导入，调用 extract_and_save() 函数。

主要流程：
    Step 1  用 OSMnx 按地名或中心点+半径查询路网
    Step 2  根据道路类型筛选（城市/农村/混合）
    Step 3  简化路网：合并度为2的中间节点，保留交叉口和端点
    Step 4  （可选）按目标节点数裁剪：取连通最大子图后随机子图采样
    Step 5  投影为 UTM 坐标（单位：米），保持真实地理距离
    Step 6  生成5种基站方向，逐一保存 .txt 算例 + .png 可视化图
"""

import io
import math
import os
import random
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ============================================================
# 无人机参数（与随机算例保持一致，可按需修改）
# ============================================================
DRONE_BATTERY       = 30000   # 无人机电池容量（此处以距离计）m
DRONE_SPEED         = 15      # 无人机速度 m/s
DRONE_ENERGY_COST   = 1.5     # 无人机能源成本系数 元/kwh
DRONE_CALL_COST     = 10      # 无人机一次调用成本 元/次
DRONE_INSPECT_COEF  = 1       # 无人机巡检能耗系数 kwh/m
DRONE_TRANSFER_COEF = 1       # 无人机转移能耗系数 kwh/m
BIG_M               = 10000   # 足够大的常数 M

# 基站偏移量相对路网包围盒跨度的比例
DEPOT_OFFSET = 0.15

# ============================================================
# 道路类型配置
# ============================================================
# 城市路网推荐类型：主次干道
URBAN_HIGHWAY_FILTER = {
    "highway": [
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
        "residential", "unclassified"
    ]
}

# 农村路网推荐类型：乡村/未分类道路
RURAL_HIGHWAY_FILTER = {
    "highway": [
        "primary", "secondary", "tertiary",
        "residential", "unclassified", "track", "service"
    ]
}

# 全类型（不过滤）
ALL_HIGHWAY_FILTER = None  # 传入 None 时让 OSMnx 使用默认 drive 过滤

# 路网场景预设（方便用户选择）
NETWORK_PRESETS = {
    "urban":   URBAN_HIGHWAY_FILTER,    # 城市主次干道
    "rural":   RURAL_HIGHWAY_FILTER,    # 农村乡道
    "drive":   None,                    # OSMnx 默认 drive 网络
    "all":     None,                    # 所有可通行道路
}

# 基站方向列表（与随机算例一致）
DEPOT_DIRECTIONS = [
    (1, "up"),
    (2, "down"),
    (3, "left"),
    (4, "right"),
    (5, "center"),
]


# ============================================================
# Step 1-2：从 OSM 拉取路网
# ============================================================

def fetch_road_network_by_place(place_name: str,
                                network_type: str = "drive",
                                custom_filter: dict = None):
    """
    按地名查询路网。

    Parameters
    ----------
    place_name : str
        地名，例如 "Shenzhen, Guangdong, China" 或 "顺义区, 北京"
    network_type : str
        OSMnx network_type：'drive' | 'walk' | 'bike' | 'all'
        当 custom_filter 不为 None 时，network_type 被忽略。
    custom_filter : dict or None
        自定义道路类型过滤器，例如 URBAN_HIGHWAY_FILTER。
        为 None 则使用 network_type 默认规则。

    Returns
    -------
    G_osm : networkx.MultiDiGraph  OSMnx 原始有向图
    """
    try:
        import osmnx as ox
    except ImportError:
        raise ImportError("请先安装 osmnx：pip install osmnx")

    ox.settings.use_cache = True
    ox.settings.log_console = False

    print(f"[OSMnx] 按地名查询路网: '{place_name}'")
    if custom_filter:
        G_osm = ox.graph_from_place(place_name, custom_filter=_build_osmnx_filter(custom_filter))
    else:
        G_osm = ox.graph_from_place(place_name, network_type=network_type)
    print(f"  原始路网: {G_osm.number_of_nodes()} 节点, {G_osm.number_of_edges()} 边")
    return G_osm


def fetch_road_network_by_point(lat: float, lon: float,
                                dist: float = 1000,
                                network_type: str = "drive",
                                custom_filter: dict = None):
    """
    按中心点 + 半径查询路网。

    Parameters
    ----------
    lat, lon : float   中心点纬度/经度（WGS84）
    dist     : float   半径，单位：米
    network_type / custom_filter：同 fetch_road_network_by_place
    """
    try:
        import osmnx as ox
    except ImportError:
        raise ImportError("请先安装 osmnx：pip install osmnx")

    ox.settings.use_cache = True
    ox.settings.log_console = False

    print(f"[OSMnx] 按中心点查询路网: lat={lat}, lon={lon}, dist={dist}m")
    if custom_filter:
        G_osm = ox.graph_from_point((lat, lon), dist=dist,
                                    custom_filter=_build_osmnx_filter(custom_filter))
    else:
        G_osm = ox.graph_from_point((lat, lon), dist=dist, network_type=network_type)
    print(f"  原始路网: {G_osm.number_of_nodes()} 节点, {G_osm.number_of_edges()} 边")
    return G_osm


def _build_osmnx_filter(filter_dict: dict) -> str:
    """将 {'highway': [...]} 转换为 OSMnx custom_filter 字符串。"""
    if "highway" in filter_dict:
        values = "|".join(filter_dict["highway"])
        return f'["highway"~"{values}"]'
    raise ValueError(f"不支持的 filter_dict 格式: {filter_dict}")


# ============================================================
# Step 3：路网简化 & 投影
# ============================================================

def simplify_and_project(G_osm):
    """
    简化路网：合并度为2的中间节点，投影为 UTM 坐标（单位：米）。
    兼容 osmnx 1.x 和 2.x（2.x 版本下载时已自动简化）。

    Returns
    -------
    G_proj : networkx.MultiDiGraph  投影后的路网（坐标单位：米）
    """
    try:
        import osmnx as ox
    except ImportError:
        raise ImportError("请先安装 osmnx：pip install osmnx")

    # 尝试简化（osmnx 2.x 下载的图已是简化图，重复简化会报错，忽略即可）
    try:
        G_simplified = ox.simplify_graph(G_osm)
        print(f"  简化后: {G_simplified.number_of_nodes()} 节点, {G_simplified.number_of_edges()} 边")
    except Exception:
        G_simplified = G_osm
        print(f"  已是简化图（osmnx 2.x）: {G_simplified.number_of_nodes()} 节点, "
              f"{G_simplified.number_of_edges()} 边")

    # 投影为 UTM（自动选择最近 UTM 带）
    # osmnx 下载的原始图坐标始终是 WGS84 经纬度，必须投影
    G_proj = ox.project_graph(G_simplified)
    crs = G_proj.graph.get("crs", "unknown")
    print(f"  已投影为 UTM 坐标（单位：米），CRS={crs}")
    return G_proj


def osmnx_to_networkx(G_proj):
    """
    将 OSMnx MultiDiGraph 转换为无向简单图（networkx.Graph），
    提取节点坐标字典 pos（单位：米，已投影）。

    Returns
    -------
    G : networkx.Graph   无向简单图
    pos : dict           {node_id: np.array([x, y])}（UTM 坐标，单位：米）
    node_list : list     节点 ID 列表（原始 OSM node id）
    """
    # 转无向图并去除多重边（只保留一条最短边），同时跳过自环（u==v）
    G_undirected = nx.Graph()
    for u, v, data in G_proj.edges(data=True):
        if u == v:          # 跳过自环
            continue
        length = data.get("length", 1.0)
        if G_undirected.has_edge(u, v):
            if length < G_undirected[u][v].get("length", float("inf")):
                G_undirected[u][v]["length"] = length
        else:
            G_undirected.add_edge(u, v, length=length)

    # 补充孤立节点
    for n in G_proj.nodes():
        if n not in G_undirected:
            G_undirected.add_node(n)

    # 提取坐标（使用 x/y 属性，即 UTM 东坐标/北坐标）
    pos = {}
    for n, data in G_proj.nodes(data=True):
        x = data.get("x", 0.0)
        y = data.get("y", 0.0)
        pos[n] = np.array([x, y])

    node_list = list(G_undirected.nodes())
    return G_undirected, pos, node_list


# ============================================================
# Step 4：按目标规模裁剪路网
# ============================================================

def crop_to_target_size(G: nx.Graph, pos: dict,
                        target_nodes: int = None,
                        target_edges: int = None,
                        max_span_m: float = 10000.0,
                        min_node_dist_m: float = 100.0,
                        seed: int = 42) -> tuple:
    """
    将路网裁剪到目标规模，并执行以下质量控制：
    1. 取最大连通子图
    2. 若跨度超过 max_span_m，按中心空间窗口收缩
    3. 【近邻过滤】删除与其他节点距离 < min_node_dist_m 的节点（保留度高的）
    4. 按目标节点数 BFS 裁剪（BFS 内部同步跳过近邻节点）
    5. 按目标边数修剪叶边
    6. 坐标中心化 + 重新索引

    Parameters
    ----------
    G                : 无向简单图
    pos              : 节点坐标字典（UTM，单位：米）
    target_nodes     : 目标节点数上限（None 表示不裁剪节点）
    target_edges     : 目标边数上限（None 表示不裁剪边）
    max_span_m       : 路网坐标跨度上限（米），默认 10000m（即 10km）
    min_node_dist_m  : 节点间最小距离（米），默认 100m，过近的节点会被过滤掉
    seed             : 随机种子

    Returns
    -------
    G_crop : nx.Graph    裁剪后的图
    pos_crop : dict      对应坐标字典（坐标中心化：减去均值，单位仍为米）
    """
    random.seed(seed)
    np.random.seed(seed)

    # 1. 取最大连通子图
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    G_main = G.subgraph(components[0]).copy()
    print(f"  最大连通子图: {G_main.number_of_nodes()} 节点, {G_main.number_of_edges()} 边")

    # 2. 若坐标跨度超过 max_span_m，先按空间窗口过滤节点（保留中心区域）
    coords_all = np.array([pos[n] for n in G_main.nodes() if n in pos])
    if len(coords_all) > 0:
        center = coords_all.mean(axis=0)
        x_span = coords_all[:, 0].max() - coords_all[:, 0].min()
        y_span = coords_all[:, 1].max() - coords_all[:, 1].min()
        if x_span > max_span_m or y_span > max_span_m:
            half = max_span_m / 2.0
            nodes_in_window = [
                n for n in G_main.nodes()
                if n in pos and
                   abs(pos[n][0] - center[0]) <= half and
                   abs(pos[n][1] - center[1]) <= half
            ]
            if len(nodes_in_window) >= 2:
                G_sub = G_main.subgraph(nodes_in_window).copy()
                comps2 = sorted(nx.connected_components(G_sub), key=len, reverse=True)
                G_main = G_sub.subgraph(comps2[0]).copy()
                print(f"  10km×10km 空间裁剪后: {G_main.number_of_nodes()} 节点, "
                      f"{G_main.number_of_edges()} 边")

    # 3. 近邻节点过滤：删除节点间距 < min_node_dist_m 的重复节点
    #    （处理双向车道平行节点、立交桥平行路等 OSM 特有问题）
    if min_node_dist_m > 0 and G_main.number_of_nodes() > 1:
        G_filtered, pos = _filter_close_nodes(G_main, pos, min_node_dist_m)
        # 过滤后取最大连通子图
        if G_filtered.number_of_nodes() >= 2:
            comps_f = sorted(nx.connected_components(G_filtered), key=len, reverse=True)
            G_main = G_filtered.subgraph(comps_f[0]).copy()
            print(f"  近邻过滤后（≥{min_node_dist_m:.0f}m）: "
                  f"{G_main.number_of_nodes()} 节点, {G_main.number_of_edges()} 边")

    # 4. 按目标节点数 BFS 裁剪（BFS 内部同步保持间距约束）
    if target_nodes is not None and G_main.number_of_nodes() > target_nodes:
        G_main, pos = _crop_by_bfs(G_main, pos, target_nodes, seed,
                                   min_dist_m=min_node_dist_m)
        print(f"  节点裁剪后: {G_main.number_of_nodes()} 节点, {G_main.number_of_edges()} 边")

    # 5. 按目标边数裁剪（移除叶节点，直到满足边数上限）
    if target_edges is not None and G_main.number_of_edges() > target_edges:
        G_main = _trim_edges(G_main, target_edges, seed)
        print(f"  边裁剪后: {G_main.number_of_nodes()} 节点, {G_main.number_of_edges()} 边")

    # 6. 重建 pos 子集并中心化坐标
    nodes_kept = list(G_main.nodes())
    pos_crop = {n: pos[n] for n in nodes_kept if n in pos}

    # 坐标中心化（减去包围盒中心，方便可视化和数值稳定）
    coords = np.array(list(pos_crop.values()))
    centroid = coords.mean(axis=0)          # UTM 绝对坐标均值（用于底图还原）
    pos_crop = {n: v - centroid for n, v in pos_crop.items()}

    # 重新索引：将 OSM node id 映射到 0..N-1
    G_reindexed, pos_reindexed = _reindex(G_main, pos_crop)

    # 返回值额外包含 utm_centroid，供底图坐标还原使用
    return G_reindexed, pos_reindexed, centroid


def _filter_close_nodes(G: nx.Graph, pos: dict, min_dist_m: float) -> tuple:
    """
    过滤掉间距过近（< min_dist_m）的节点对，保留度更高的那个。
    使用贪心策略：按节点度降序排列，逐个加入"保留集"，若候选节点与已保留任一节点
    距离 < min_dist_m 则标记为"被合并"。

    关键改进：被删除节点的所有邻居之间，会补充直连边（节点收缩/路径压缩），
    确保删除节点后其两侧邻居仍保持连通，从而不丢失路段。
    补充边的长度取被删节点到两侧邻居距离之和（折算直线近似）。

    Parameters
    ----------
    G           : 无向图
    pos         : 节点坐标字典（UTM，单位：米）
    min_dist_m  : 节点间最小距离（米）

    Returns
    -------
    G_filtered : nx.Graph   过滤后包含补充边的图
    pos        : dict       原始坐标字典（未改变）
    """
    nodes = list(G.nodes())
    # 按度降序排列：优先保留度高（交叉口）的节点
    nodes_sorted = sorted(nodes, key=lambda n: G.degree(n), reverse=True)

    kept = set()
    kept_coords = []
    # node -> 对应的"保留集代表节点"（被合并节点映射到最近保留节点）
    merge_to = {}   # {removed_node: kept_representative}

    for n in nodes_sorted:
        if n not in pos:
            continue
        c = pos[n]
        # 检查与已保留节点的最近距离
        closest_kept = None
        closest_dist = float("inf")
        for i, kc in enumerate(kept_coords):
            d = np.linalg.norm(c - kc)
            if d < closest_dist:
                closest_dist = d
                closest_kept = list(kept)[i] if i < len(kept) else None
        # 重新找对应 kept 节点（因为 kept 是 set，顺序不固定）
        closest_kept = None
        for k in kept:
            if k in pos:
                d = np.linalg.norm(c - pos[k])
                if d < min_dist_m:
                    if closest_kept is None or d < np.linalg.norm(c - pos[closest_kept]):
                        closest_kept = k

        if closest_kept is None:
            # 离所有已保留节点都足够远 → 保留该节点
            kept.add(n)
            kept_coords.append(c)
        else:
            # 太近 → 标记合并到 closest_kept
            merge_to[n] = closest_kept

    # 在保留节点集合上构建新图，同时做节点收缩：
    # 被删节点的边转移到其代表节点上
    G_new = nx.Graph()
    G_new.add_nodes_from(kept)

    def resolve(n):
        """递归找最终代表节点（处理链式合并）"""
        visited = set()
        while n in merge_to and n not in visited:
            visited.add(n)
            n = merge_to[n]
        return n

    for u, v in G.edges():
        ru = resolve(u)
        rv = resolve(v)
        if ru == rv:        # 收缩后形成自环，跳过
            continue
        if ru not in kept or rv not in kept:
            continue
        if not G_new.has_edge(ru, rv):
            # 估算边长：使用实际节点坐标的欧氏距离
            length = float(np.linalg.norm(pos[ru] - pos[rv])) if (ru in pos and rv in pos) else 1.0
            G_new.add_edge(ru, rv, length=length)

    return G_new, pos


def _crop_by_bfs(G: nx.Graph, pos: dict, target_nodes: int, seed: int,
                 min_dist_m: float = 100.0) -> tuple:
    """
    从图的地理中心最近节点开始 BFS，采集节点，同时跳过与已收集节点
    距离 < min_dist_m 的过近节点，直到收集够 target_nodes 个。

    关键改进：跳过的过近节点不直接丢弃，而是做节点收缩——
    将其所有边转移到最近的已保留代表节点上，确保路段连通性不被破坏。

    Parameters
    ----------
    min_dist_m : 节点间最小距离（米），默认 100m
    """
    nodes = list(G.nodes())
    coords = np.array([pos[n] for n in nodes])
    center = coords.mean(axis=0)

    # 找到距离中心最近的节点作为 BFS 起点
    dists = np.linalg.norm(coords - center, axis=1)
    start_node = nodes[int(np.argmin(dists))]

    # BFS 收集节点，同时做最小间距过滤
    visited = []          # 最终保留的节点（满足间距约束）
    visited_coords = []   # 对应坐标（加速距离计算）
    merge_to = {}         # {跳过节点: 最近的已保留节点}（用于边转移）
    queue = [start_node]
    seen = {start_node}

    while queue and len(visited) < target_nodes:
        node = queue.pop(0)
        if node not in pos:
            continue
        c = pos[node]
        # 找最近的已保留节点
        closest_kept = None
        for i, vc in enumerate(visited_coords):
            if np.linalg.norm(c - vc) < min_dist_m:
                closest_kept = visited[i]
                break
        if closest_kept is None:
            # 离所有已保留节点都足够远 → 保留该节点
            visited.append(node)
            visited_coords.append(c)
        else:
            # 太近 → 合并到最近保留节点，记录用于边转移
            merge_to[node] = closest_kept

        # 按距离中心排序邻居，优先访问近中心的节点
        neighbors = sorted(
            [nb for nb in G.neighbors(node) if nb not in seen],
            key=lambda nb: np.linalg.norm(pos[nb] - center) if nb in pos else 1e9
        )
        for nb in neighbors:
            seen.add(nb)
            queue.append(nb)

    # 若 BFS 过滤后节点不足（路网过密），放宽限制取最多节点
    if len(visited) < 2:
        visited2 = []
        queue2 = [start_node]
        seen2 = {start_node}
        merge_to = {}
        while queue2 and len(visited2) < target_nodes:
            node = queue2.pop(0)
            visited2.append(node)
            for nb in sorted(G.neighbors(node),
                             key=lambda nb: np.linalg.norm(pos[nb] - center) if nb in pos else 1e9):
                if nb not in seen2:
                    seen2.add(nb)
                    queue2.append(nb)
        visited = visited2

    # 构建裁剪后的子图（含节点收缩后的边转移）
    visited_set = set(visited)

    def resolve(n):
        """递归找最终代表节点（处理链式合并）"""
        v_set = set()
        while n in merge_to and n not in v_set:
            v_set.add(n)
            n = merge_to[n]
        return n

    sub = nx.Graph()
    sub.add_nodes_from(visited)

    # 遍历 BFS 扫描过的所有节点的边，做收缩后加入子图
    for node in seen:
        for nb in G.neighbors(node):
            if nb not in seen:
                continue
            ru = resolve(node)
            rv = resolve(nb)
            if ru == rv:            # 收缩后自环，跳过
                continue
            if ru not in visited_set or rv not in visited_set:
                continue
            if not sub.has_edge(ru, rv):
                length = float(np.linalg.norm(pos[ru] - pos[rv])) if (ru in pos and rv in pos) else 1.0
                sub.add_edge(ru, rv, length=length)

    return sub, pos


def _trim_edges(G: nx.Graph, target_edges: int, seed: int) -> nx.Graph:
    """
    通过逐步移除叶边（连接度为1的节点的边）来减少边数，
    同时保持图的连通性。
    """
    G = G.copy()
    random.seed(seed)
    max_iter = G.number_of_edges() * 2
    it = 0
    while G.number_of_edges() > target_edges and it < max_iter:
        it += 1
        # 找度为1的叶节点
        leaves = [n for n in G.nodes() if G.degree(n) == 1]
        if not leaves:
            break
        leaf = random.choice(leaves)
        neighbor = list(G.neighbors(leaf))[0]
        G.remove_node(leaf)
    return G


def _reindex(G: nx.Graph, pos: dict) -> tuple:
    """
    将节点 ID 重新映射为 0..N-1 的整数索引。
    """
    node_list = list(G.nodes())
    mapping = {old: new for new, old in enumerate(node_list)}
    G_new = nx.relabel_nodes(G, mapping)
    pos_new = {mapping[n]: pos[n] for n in node_list}
    return G_new, pos_new


# ============================================================
# Step 5：基站位置计算
# ============================================================

def compute_depot_pos(pos: dict, direction: str) -> np.ndarray:
    """
    根据方向在路网包围盒外侧生成基站坐标（单位：米）。

    Parameters
    ----------
    pos       : {node_id: np.array([x, y])}
    direction : 'up' | 'down' | 'left' | 'right' | 'center'
    """
    coords = np.array(list(pos.values()))
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    x_span = max((x_max - x_min) * DEPOT_OFFSET, 250.0)  # 至少 250m
    y_span = max((y_max - y_min) * DEPOT_OFFSET, 250.0)

    if direction == "up":
        return np.array([cx, y_max + y_span])
    elif direction == "down":
        return np.array([cx, y_min - y_span])
    elif direction == "left":
        return np.array([x_min - x_span, cy])
    elif direction == "right":
        return np.array([x_max + x_span, cy])
    else:  # center
        return np.array([cx, cy])


# ============================================================
# Step 6：保存算例 txt
# ============================================================

def save_instance(filepath: str, G: nx.Graph, pos: dict,
                  depot_pos: np.ndarray, num_drones: int):
    """
    将路网算例保存为 txt 文件，格式与随机算例完全一致。

    节点编号：基站=0，路网节点 1..N（1-indexed）
    坐标：基站放第一位，其后依次为路网节点
    需求弧：首行固定 (0,0)（基站自环），其余为路网边（1-indexed）
    """
    num_road_nodes = G.number_of_nodes()
    total_nodes    = num_road_nodes + 1
    # 过滤自环边（u==v），双重保险
    edges          = [(u, v) for u, v in G.edges() if u != v]
    num_edges      = len(edges)

    # 构建含基站的完整坐标字典（基站索引=0）
    all_pos = {0: depot_pos}
    for road_node in range(num_road_nodes):
        all_pos[road_node + 1] = pos[road_node]

    x_list = [all_pos[0][0]] + [all_pos[i][0] for i in range(1, total_nodes)]
    y_list = [all_pos[0][1]] + [all_pos[i][1] for i in range(1, total_nodes)]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"1\n")                     # 基站点数量
        f.write(f"{num_road_nodes}\n")      # 端点数量（路网节点数，不含基站）
        f.write(f"{total_nodes}\n")         # 节点数量（含基站）
        f.write(f"{num_edges}\n")           # 边数量（需求边数）
        f.write(f"{num_drones}\n")          # 无人机数量
        f.write(f"\n")
        f.write(f"{DRONE_BATTERY}\n")
        f.write(f"{DRONE_SPEED}\n")
        f.write(f"{DRONE_ENERGY_COST}\n")
        f.write(f"{DRONE_CALL_COST}\n")
        f.write(f"{DRONE_INSPECT_COEF}\n")
        f.write(f"{DRONE_TRANSFER_COEF}\n")
        f.write(f"{BIG_M}\n")
        f.write(f"\n")
        # 坐标（保留2位小数）
        f.write(", ".join(f"{x:.2f}" for x in x_list) + "\n")
        f.write(", ".join(f"{y:.2f}" for y in y_list) + "\n")
        f.write(f"\n")
        # 需求弧
        f.write("(0,0)\n")
        for u, v in edges:
            f.write(f"({u + 1},{v + 1})\n")

    print(f"  已保存算例: {filepath}")


# ============================================================
# Step 6：OSM 瓦片底图工具
# ============================================================

def _deg2tile(lat_deg: float, lon_deg: float, zoom: int):
    """经纬度 → OSM 瓦片 (x, y) 坐标（整数）。"""
    lat_r = math.radians(lat_deg)
    n = 2 ** zoom
    x = int((lon_deg + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n)
    return x, y


def _tile2deg(x: int, y: int, zoom: int):
    """瓦片左上角 → 经纬度。"""
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_r = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_r)
    return lat, lon


def _fetch_osm_basemap(lon_min: float, lat_min: float,
                       lon_max: float, lat_max: float,
                       zoom: int = None):
    """
    拉取覆盖给定经纬度范围的 OSM 瓦片，拼接为单张 PIL Image。
    返回 (img, (lon_left, lon_right, lat_bottom, lat_top))，
    失败时返回 None。

    zoom=None 时自动根据范围选择缩放级别（使瓦片数 ≤ 4×4=16）。
    """
    try:
        import requests
        from PIL import Image
    except ImportError:
        return None

    # 自动选择 zoom（使拼接瓦片数合理）
    if zoom is None:
        for z in range(16, 10, -1):
            x0, y0 = _deg2tile(lat_max, lon_min, z)
            x1, y1 = _deg2tile(lat_min, lon_max, z)
            if (abs(x1 - x0) + 1) * (abs(y1 - y0) + 1) <= 25:
                zoom = z
                break
        else:
            zoom = 12

    x0, y0 = _deg2tile(lat_max, lon_min, zoom)
    x1, y1 = _deg2tile(lat_min, lon_max, zoom)
    # 保证 x0<=x1, y0<=y1
    if x0 > x1: x0, x1 = x1, x0
    if y0 > y1: y0, y1 = y1, y0

    # OSM 瓦片服务（多个镜像轮流使用，提高可靠性）
    tile_servers = [
        "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "https://b.tile.openstreetmap.org/{z}/{x}/{y}.png",
    ]
    HEADERS = {"User-Agent": "RoadInspectionResearch/1.0 (academic use)"}

    tile_size = 256
    cols = x1 - x0 + 1
    rows = y1 - y0 + 1
    mosaic = Image.new("RGB", (cols * tile_size, rows * tile_size), (240, 240, 240))

    any_ok = False
    for row, ty in enumerate(range(y0, y1 + 1)):
        for col, tx in enumerate(range(x0, x1 + 1)):
            fetched = False
            for server in tile_servers:
                url = server.format(z=zoom, x=tx, y=ty)
                try:
                    resp = requests.get(url, headers=HEADERS, timeout=8)
                    if resp.status_code == 200:
                        tile_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                        mosaic.paste(tile_img, (col * tile_size, row * tile_size))
                        fetched = True
                        any_ok = True
                        break
                except Exception:
                    continue
            if not fetched:
                # 占位灰色
                pass

    if not any_ok:
        return None

    # 计算拼接图对应的地理范围
    lat_top,  lon_left  = _tile2deg(x0,     y0,     zoom)
    lat_bot,  lon_right = _tile2deg(x1 + 1, y1 + 1, zoom)
    return mosaic, (lon_left, lon_right, lat_bot, lat_top)


def _utm_pos_to_wgs84(pos: dict, crs_epsg: int):
    """
    将 UTM 坐标字典（已中心化）还原为 WGS84 经纬度。
    pos 中的坐标是相对均值中心化后的值（单位：米），
    需要先加上中心 UTM 坐标后再反投影。

    注意：pos 在 crop_to_target_size 中已减去均值，
    所以这里需要传入原始均值偏移（utm_origin）才能正确还原。
    如果 utm_origin 未知则无法还原——因此本函数接收原始 UTM pos（未中心化）。
    """
    try:
        from pyproj import Transformer
    except ImportError:
        return None, None

    transformer = Transformer.from_crs(f"EPSG:{crs_epsg}", "EPSG:4326", always_xy=True)
    lons, lats = [], []
    for coord in pos.values():
        lon, lat = transformer.transform(float(coord[0]), float(coord[1]))
        lons.append(lon)
        lats.append(lat)
    return lats, lons


# ============================================================
# Step 6：保存可视化图
# ============================================================

def save_network_figure(fig_path: str, G: nx.Graph, pos: dict,
                        all_depot_positions: list, title: str = "",
                        crs_epsg: int = None, utm_centroid: np.ndarray = None):
    """
    绘制路网与所有基站位置，保存为 PNG。
    若提供 crs_epsg 和 utm_centroid，则自动叠加 OSM 底图。

    Parameters
    ----------
    all_depot_positions : [(depot_id, direction_label, depot_pos), ...]
    crs_epsg            : UTM 投影 EPSG 代码（如 32649），用于还原地理坐标
    utm_centroid        : 坐标中心化时减去的 UTM 均值偏移 (shape: [2,])
    """
    pos_tuple = {k: (float(v[0]), float(v[1])) for k, v in pos.items()}
    labels    = {i: str(i + 1) for i in G.nodes()}

    # 收集所有点坐标（节点 + 基站），用于确定绘图范围
    all_xy = list(pos_tuple.values())
    for _, _, dp in all_depot_positions:
        all_xy.append((float(dp[0]), float(dp[1])))
    xs = [p[0] for p in all_xy]
    ys = [p[1] for p in all_xy]
    margin = max((max(xs) - min(xs)) * 0.15, (max(ys) - min(ys)) * 0.15, 300.0)
    x_lo, x_hi = min(xs) - margin, max(xs) + margin
    y_lo, y_hi = min(ys) - margin, max(ys) + margin

    fig, ax = plt.subplots(figsize=(10, 10))

    # ---- 叠加 OSM 底图 ----
    basemap_ok = False
    if crs_epsg is not None and utm_centroid is not None:
        try:
            from pyproj import Transformer
            # 把绘图范围（中心化 UTM）还原回真实 UTM，再转 WGS84
            tf = Transformer.from_crs(f"EPSG:{crs_epsg}", "EPSG:4326", always_xy=True)
            cx, cy = float(utm_centroid[0]), float(utm_centroid[1])
            # 四角还原
            corners_utm = [
                (x_lo + cx, y_lo + cy),
                (x_hi + cx, y_hi + cy),
            ]
            wgs84_corners = [tf.transform(u, v) for u, v in corners_utm]
            lon_min = min(c[0] for c in wgs84_corners)
            lon_max = max(c[0] for c in wgs84_corners)
            lat_min = min(c[1] for c in wgs84_corners)
            lat_max = max(c[1] for c in wgs84_corners)

            result = _fetch_osm_basemap(lon_min, lat_min, lon_max, lat_max)
            if result is not None:
                mosaic, (tlon_l, tlon_r, tlat_b, tlat_t) = result
                # 将底图地理范围转回中心化 UTM 范围，用于 imshow extent
                def wgs84_to_centered_utm(lon, lat):
                    tf2 = Transformer.from_crs("EPSG:4326", f"EPSG:{crs_epsg}", always_xy=True)
                    ux, uy = tf2.transform(lon, lat)
                    return ux - cx, uy - cy
                ul_x, ul_y = wgs84_to_centered_utm(tlon_l, tlat_t)
                lr_x, lr_y = wgs84_to_centered_utm(tlon_r, tlat_b)
                ax.imshow(mosaic, extent=[ul_x, lr_x, lr_y, ul_y],
                          aspect="auto", zorder=0, alpha=0.75)
                basemap_ok = True
        except Exception as e:
            warnings.warn(f"底图加载失败: {e}")

    if not basemap_ok:
        # 无底图时用淡灰色背景
        ax.set_facecolor("#F5F5F5")

    # ---- 绘制路网 ----
    edge_color = "#1A3A6B" if basemap_ok else "#444444"
    node_color = "#4FC3F7" if basemap_ok else "#5B9BD5"
    nx.draw_networkx_edges(G, pos_tuple, ax=ax,
                           edge_color=edge_color, width=2.2, alpha=0.9)
    nx.draw_networkx_nodes(G, pos_tuple, ax=ax,
                           node_color=node_color, node_size=320,
                           linewidths=1.2, edgecolors="white")
    nx.draw_networkx_labels(G, pos_tuple, labels=labels, ax=ax,
                            font_size=6.5, font_color="white",
                            font_weight="bold")

    # ---- 绘制基站 ----
    depot_colors = {1: "#E74C3C", 2: "#2ECC71", 3: "#F39C12", 4: "#9B59B6", 5: "#1ABC9C"}
    depot_labels = {1: "up",      2: "down",    3: "left",    4: "right",   5: "center"}

    for depot_id, _direction, depot_pos in all_depot_positions:
        dx, dy = float(depot_pos[0]), float(depot_pos[1])
        color  = depot_colors[depot_id]
        ax.scatter(dx, dy, marker="*", s=420, c=color,
                   label=f"depot-{depot_labels[depot_id]}",
                   edgecolors="white", linewidths=0.8, zorder=5)
        ax.annotate(f"D{depot_id}", xy=(dx, dy), xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=7.5, color=color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, ec="none"))

    # ---- 坐标轴与样式 ----
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.85,
              facecolor="white", edgecolor="#CCCCCC")
    ax.set_title(title or os.path.splitext(os.path.basename(fig_path))[0],
                 fontsize=9.5, pad=10)
    ax.axis("on")
    ax.grid(True, linestyle="--", alpha=0.25, color="gray")
    ax.set_xlabel("UTM Easting offset (m)", fontsize=8)
    ax.set_ylabel("UTM Northing offset (m)", fontsize=8)
    if basemap_ok:
        ax.text(0.01, 0.01, "© OpenStreetMap contributors",
                transform=ax.transAxes, fontsize=6, color="gray",
                va="bottom", ha="left")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  已保存图: {fig_path}")


# ============================================================
# 主接口：extract_and_save
# ============================================================

def extract_and_save(
    query,
    query_type: str = "place",
    dist: float = 2000,
    network_scene: str = "drive",
    custom_filter: dict = None,
    target_nodes: int = None,
    target_edges: int = None,
    max_span_m: float = 10000.0,
    min_node_dist_m: float = 100.0,
    num_drones: int = 2,
    instance_name: str = None,
    output_dir: str = None,
    save_txt: bool = True,
    save_png: bool = True,
    seed: int = 42,
):
    """
    一键提取真实路网并生成算例文件。

    Parameters
    ----------
    query : str or (float, float)
        地名字符串（query_type='place'）或 (lat, lon) 元组（query_type='point'）
    query_type : str
        'place'（按地名）或 'point'（按中心点+半径）
    dist : float
        中心点查询时的半径（米），query_type='place' 时忽略
    network_scene : str
        路网场景预设：'urban'（城市主次干道）| 'rural'（乡村道路）|
                     'drive'（OSMnx 默认可驾驶路网）| 'all'（所有道路）
    custom_filter : dict or None
        自定义道路类型过滤器，不为 None 时覆盖 network_scene
    target_nodes : int or None
        期望保留的路网节点数上限（None=不裁剪）
    target_edges : int or None
        期望保留的路网边数上限（None=不裁剪）
    max_span_m : float
        路网坐标跨度上限（米），默认 10000（10km×10km 限制）
    min_node_dist_m : float
        节点间最小距离（米），默认 100m，过近的节点会被过滤
    num_drones : int
        无人机数量
    instance_name : str or None
        算例基础名称（None 时自动生成）
    output_dir : str or None
        输出目录（None 时使用 算例/实际算例/）
    save_txt : bool
        是否保存 txt 算例文件
    save_png : bool
        是否保存 png 可视化图
    seed : int
        随机裁剪时的随机种子

    Returns
    -------
    G : nx.Graph         最终路网无向图
    pos : dict           节点坐标字典（UTM，已中心化，单位：米）
    saved_files : list   保存的文件路径列表
    """
    # ----- 确定输出目录 -----
    if output_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, "算例", "实际算例")
    os.makedirs(output_dir, exist_ok=True)

    # ----- 确定道路过滤器 -----
    if custom_filter is not None:
        cf = custom_filter
        nt = "all"
    else:
        cf = NETWORK_PRESETS.get(network_scene)
        nt = "drive" if network_scene in ("drive", "all") else "all"

    # ----- Step 1-2：拉取路网 -----
    if query_type == "place":
        G_osm = fetch_road_network_by_place(query, network_type=nt, custom_filter=cf)
    elif query_type == "point":
        lat, lon = query
        G_osm = fetch_road_network_by_point(lat, lon, dist=dist,
                                            network_type=nt, custom_filter=cf)
    else:
        raise ValueError(f"query_type 必须为 'place' 或 'point'，当前值: {query_type}")

    # ----- Step 3：简化 + 投影 -----
    G_proj = simplify_and_project(G_osm)
    G_undi, pos_raw, _ = osmnx_to_networkx(G_proj)

    # ----- Step 4：按目标规模裁剪 -----
    # 提取 EPSG 代码用于底图定位
    crs_str = str(G_proj.graph.get("crs", ""))
    crs_epsg = None
    if "EPSG:" in crs_str.upper():
        try:
            crs_epsg = int(crs_str.upper().split("EPSG:")[1].split()[0].strip("),:;"))
        except Exception:
            pass

    utm_centroid = None
    if target_nodes is not None or target_edges is not None:
        G_final, pos_final, utm_centroid = crop_to_target_size(
            G_undi, pos_raw,
            target_nodes=target_nodes,
            target_edges=target_edges,
            max_span_m=max_span_m,
            min_node_dist_m=min_node_dist_m,
            seed=seed
        )
    else:
        # 不裁剪：仍取最大连通子图并重新索引
        comps = sorted(nx.connected_components(G_undi), key=len, reverse=True)
        G_main = G_undi.subgraph(comps[0]).copy()
        coords = np.array([pos_raw[n] for n in G_main.nodes() if n in pos_raw])
        utm_centroid = coords.mean(axis=0)
        pos_centered = {n: pos_raw[n] - utm_centroid
                        for n in G_main.nodes() if n in pos_raw}
        G_final, pos_final = _reindex(G_main, pos_centered)

    road_nodes = G_final.number_of_nodes()
    actual_edges = G_final.number_of_edges()
    total_nodes  = road_nodes + 1

    if actual_edges == 0:
        warnings.warn("路网边数为0，请检查查询范围或道路类型设置！")
        return G_final, pos_final, []

    print(f"\n最终路网: {road_nodes} 路网节点, {actual_edges} 需求边, "
          f"含基站共 {total_nodes} 节点")

    # ----- Step 5-6：生成5种基站，保存文件 -----
    # 自动生成算例基础名称
    if instance_name is None:
        if query_type == "place":
            safe_name = str(query).replace(" ", "_").replace(",", "").replace("/", "_")[:20]
        else:
            lat, lon = query
            safe_name = f"{lat:.4f}_{lon:.4f}"
        instance_name = f"{safe_name}"

    saved_files = []

    # 收集5种基站位置，用于统一绘图
    all_depot_positions = []
    for depot_id, direction in DEPOT_DIRECTIONS:
        depot_pos = compute_depot_pos(pos_final, direction)
        all_depot_positions.append((depot_id, direction, depot_pos))

        if save_txt:
            fname = f"{total_nodes}-{actual_edges}-{num_drones}-{depot_id}-({instance_name}).txt"
            fpath = os.path.join(output_dir, fname)
            save_instance(fpath, G_final, pos_final, depot_pos, num_drones)
            saved_files.append(fpath)

    if save_png:
        fig_name = f"{total_nodes}-{actual_edges}-{num_drones}-({instance_name}).png"
        fig_path = os.path.join(output_dir, fig_name)
        title    = f"{instance_name} | nodes={road_nodes}, edges={actual_edges}"
        save_network_figure(fig_path, G_final, pos_final, all_depot_positions,
                            title=title, crs_epsg=crs_epsg, utm_centroid=utm_centroid)
        saved_files.append(fig_path)

    print(f"\n全部完成，输出目录: {output_dir}")
    print(f"  共生成 {len([f for f in saved_files if f.endswith('.txt')])} 个算例文件"
          f" + {len([f for f in saved_files if f.endswith('.png')])} 张可视化图")

    return G_final, pos_final, saved_files


# ============================================================
# 按规模批量生成实际算例（类比随机算例的批量逻辑）
# ============================================================

def batch_extract(
    queries: list,
    scale_configs: list = None,
    num_drones: int = 2,
    network_scene: str = "drive",
    max_span_m: float = 10000.0,
    output_base_dir: str = None,
    save_txt: bool = True,
    save_png: bool = True,
):
    """
    批量从多个地点提取实际路网算例。

    Parameters
    ----------
    queries : list of dict
        每个元素为一个查询配置，格式：
        {
            "query": "深圳市南山区" 或 (lat, lon),
            "query_type": "place" 或 "point",   # 默认 "place"
            "dist": 2000,                         # 仅 point 时有效
            "name": "ShenZhen_NanShan",           # 可选，算例名称前缀
        }
    scale_configs : list of dict or None
        每个元素为一个规模配置，格式：
        {
            "target_nodes": 20,
            "target_edges": 30,
            "scale_label": "1-Medium",   # 输出子目录名称
        }
        None 时不裁剪，直接用原始简化后的路网。
    num_drones : int
        无人机数量
    network_scene : str
        路网场景：'urban' | 'rural' | 'drive' | 'all'
    output_base_dir : str or None
        输出根目录（None 时使用 算例/实际算例/）
    save_txt / save_png : bool
        是否保存 txt/png

    Returns
    -------
    all_results : list   所有生成文件路径
    """
    if output_base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_base_dir = os.path.join(base_dir, "算例", "实际算例")

    all_results = []

    for q_cfg in queries:
        query      = q_cfg["query"]
        query_type = q_cfg.get("query_type", "place")
        dist       = q_cfg.get("dist", 2000)
        name       = q_cfg.get("name", None)

        if scale_configs is None:
            # 不按规模裁剪
            out_dir = output_base_dir
            _, _, files = extract_and_save(
                query=query, query_type=query_type, dist=dist,
                network_scene=network_scene,
                max_span_m=max_span_m,
                num_drones=num_drones, instance_name=name,
                output_dir=out_dir,
                save_txt=save_txt, save_png=save_png,
            )
            all_results.extend(files)
        else:
            for sc in scale_configs:
                target_nodes  = sc.get("target_nodes")
                target_edges  = sc.get("target_edges")
                scale_label   = sc.get("scale_label", "0-Small")
                out_dir = os.path.join(output_base_dir, scale_label)
                os.makedirs(out_dir, exist_ok=True)

                print(f"\n{'='*60}")
                print(f"查询: {query}  |  规模: nodes<={target_nodes}, edges<={target_edges}")
                print(f"{'='*60}")

                _, _, files = extract_and_save(
                    query=query, query_type=query_type, dist=dist,
                    network_scene=network_scene,
                    target_nodes=target_nodes, target_edges=target_edges,
                    max_span_m=max_span_m,
                    num_drones=num_drones, instance_name=name,
                    output_dir=out_dir,
                    save_txt=save_txt, save_png=save_png,
                )
                all_results.extend(files)

    return all_results


# ============================================================
# 大规模实际算例批量生成
# ============================================================

# ----------------------------------------------------------
# 20 个查询点配置表：广州(GZ)、成都(CD)、上海(SH)、深圳(SZ) 各 5 个
# 查询半径 dist 设为 4000~5000m，保证路网覆盖范围在 10km×10km 内
# (lat, lon) 均为各区域代表性路口/中心点坐标
# ----------------------------------------------------------
LARGE_PRACTICAL_QUERIES = [
    # -------- 广州 Guangzhou (5个) --------
    {
        "query": (23.1291, 113.2644),   # 广州天河体育中心
        "query_type": "point", "dist": 4500,
        "name": "GZ_Tianhe",
    },
    {
        "query": (23.1001, 113.3240),   # 广州番禺广场
        "query_type": "point", "dist": 4500,
        "name": "GZ_Panyu",
    },
    {
        "query": (23.1570, 113.2210),   # 广州白云区白云大道
        "query_type": "point", "dist": 4500,
        "name": "GZ_Baiyun",
    },
    {
        "query": (23.0785, 113.1608),   # 广州荔湾区
        "query_type": "point", "dist": 4500,
        "name": "GZ_Liwan",
    },
    {
        "query": (23.0200, 113.4112),   # 广州黄埔区
        "query_type": "point", "dist": 4500,
        "name": "GZ_Huangpu",
    },

    # -------- 成都 Chengdu (5个) --------
    {
        "query": (30.6598, 104.0658),   # 成都天府广场
        "query_type": "point", "dist": 4500,
        "name": "CD_Tianfu",
    },
    {
        "query": (30.5728, 104.0668),   # 成都高新区
        "query_type": "point", "dist": 4500,
        "name": "CD_Gaoxin",
    },
    {
        "query": (30.7420, 104.0427),   # 成都金牛区
        "query_type": "point", "dist": 4500,
        "name": "CD_Jinniu",
    },
    {
        "query": (30.6290, 104.1460),   # 成都龙泉驿区
        "query_type": "point", "dist": 4500,
        "name": "CD_Longquanyi",
    },
    {
        "query": (30.6924, 103.9380),   # 成都温江区
        "query_type": "point", "dist": 4500,
        "name": "CD_Wenjiang",
    },

    # -------- 上海 Shanghai (5个) --------
    {
        "query": (31.2304, 121.4737),   # 上海人民广场
        "query_type": "point", "dist": 4500,
        "name": "SH_Renmin",
    },
    {
        "query": (31.1983, 121.5440),   # 上海浦东陆家嘴
        "query_type": "point", "dist": 4500,
        "name": "SH_Lujiazui",
    },
    {
        "query": (31.2990, 121.4580),   # 上海宝山区
        "query_type": "point", "dist": 4500,
        "name": "SH_Baoshan",
    },
    {
        "query": (31.1490, 121.3870),   # 上海闵行区
        "query_type": "point", "dist": 4500,
        "name": "SH_Minhang",
    },
    {
        "query": (31.3680, 121.3620),   # 上海嘉定区
        "query_type": "point", "dist": 4500,
        "name": "SH_Jiading",
    },

    # -------- 深圳 Shenzhen (5个) --------
    {
        "query": (22.5431, 114.0579),   # 深圳南山科技园
        "query_type": "point", "dist": 4500,
        "name": "SZ_Nanshan",
    },
    {
        "query": (22.5729, 114.1050),   # 深圳福田中心区
        "query_type": "point", "dist": 4500,
        "name": "SZ_Futian",
    },
    {
        "query": (22.6470, 114.0650),   # 深圳龙华区
        "query_type": "point", "dist": 4500,
        "name": "SZ_Longhua",
    },
    {
        "query": (22.7236, 114.2340),   # 深圳龙岗区
        "query_type": "point", "dist": 4500,
        "name": "SZ_Longgang",
    },
    {
        "query": (22.5105, 113.9280),   # 深圳宝安区
        "query_type": "point", "dist": 4500,
        "name": "SZ_Baoan",
    },
]

# ----------------------------------------------------------
# 20 个 (target_nodes, target_edges) 配置，节点 20-50 均匀，边 30-100 均匀
# 一个查询点对应一个配置，20×5方向 = 100个算例
# ----------------------------------------------------------
LARGE_PRACTICAL_SCALE_CONFIGS = [
    # 广州 5 个：节点 20/25/30/38/45，边 30/40/52/63/75
    {"target_nodes": 20, "target_edges":  30},
    {"target_nodes": 25, "target_edges":  40},
    {"target_nodes": 30, "target_edges":  52},
    {"target_nodes": 38, "target_edges":  63},
    {"target_nodes": 45, "target_edges":  75},
    # 成都 5 个：节点 22/27/33/40/48，边 35/45/55/68/80
    {"target_nodes": 22, "target_edges":  35},
    {"target_nodes": 27, "target_edges":  45},
    {"target_nodes": 33, "target_edges":  55},
    {"target_nodes": 40, "target_edges":  68},
    {"target_nodes": 48, "target_edges":  80},
    # 上海 5 个：节点 21/29/35/42/50，边 32/48/58/72/88
    {"target_nodes": 21, "target_edges":  32},
    {"target_nodes": 29, "target_edges":  48},
    {"target_nodes": 35, "target_edges":  58},
    {"target_nodes": 42, "target_edges":  72},
    {"target_nodes": 50, "target_edges":  88},
    # 深圳 5 个：节点 23/31/37/44/50，边 38/52/65/80/100
    {"target_nodes": 23, "target_edges":  38},
    {"target_nodes": 31, "target_edges":  52},
    {"target_nodes": 37, "target_edges":  65},
    {"target_nodes": 44, "target_edges":  80},
    {"target_nodes": 50, "target_edges": 100},
]


def generate_large_practical_instances(
    output_dir: str = None,
    num_drones: int = 2,
    network_scene: str = "drive",
    save_txt: bool = True,
    save_png: bool = True,
    max_span_m: float = 10000.0,
    min_node_dist_m: float = 100.0,
):
    """
    批量生成 100 个大规模实际算例（20个路网 × 5个基站方向）。

    规模范围：路网节点数 20-50，需求边数 30-100，路网地理跨度 ≤ 10km×10km。
    地区来源：广州、成都、上海、深圳各 5 个查询点。
    输出目录：算例/实际算例/2-Large/

    Parameters
    ----------
    output_dir      : 输出目录（None 时自动定位到 算例/实际算例/2-Large/）
    num_drones      : 无人机数量
    network_scene   : 路网场景（'drive' | 'urban' | 'rural' | 'all'）
    save_txt        : 是否保存 txt 算例文件
    save_png        : 是否保存 png 可视化图
    max_span_m      : 路网坐标跨度上限（米），默认 10000（10km×10km）
    min_node_dist_m : 节点间最小距离（米），默认 100m，用于过滤重合节点

    Returns
    -------
    all_files : list  所有已生成的文件路径
    """
    if output_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, "算例", "实际算例", "2-Large")
    os.makedirs(output_dir, exist_ok=True)

    assert len(LARGE_PRACTICAL_QUERIES) == len(LARGE_PRACTICAL_SCALE_CONFIGS), (
        "查询点数量与规模配置数量不匹配！"
    )

    all_files = []
    total = len(LARGE_PRACTICAL_QUERIES)

    for idx, (q_cfg, sc) in enumerate(
            zip(LARGE_PRACTICAL_QUERIES, LARGE_PRACTICAL_SCALE_CONFIGS)):

        query      = q_cfg["query"]
        query_type = q_cfg.get("query_type", "point")
        dist       = q_cfg.get("dist", 4500)
        name       = q_cfg["name"]
        target_nodes = sc["target_nodes"]
        target_edges = sc["target_edges"]

        print(f"\n{'='*65}")
        print(f"[{idx+1:02d}/{total}] {name}  "
              f"目标: nodes≤{target_nodes}, edges≤{target_edges}, span≤{max_span_m/1000:.0f}km")
        print(f"{'='*65}")

        try:
            # 确定道路过滤器
            cf = NETWORK_PRESETS.get(network_scene)
            nt = "drive" if network_scene in ("drive", "all") else "all"

            # 拉取路网
            lat, lon = query
            G_osm = fetch_road_network_by_point(lat, lon, dist=dist,
                                                network_type=nt, custom_filter=cf)

            # 简化 + 投影
            G_proj = simplify_and_project(G_osm)
            G_undi, pos_raw, _ = osmnx_to_networkx(G_proj)

            # 提取 EPSG 代码用于底图定位
            crs_str = str(G_proj.graph.get("crs", ""))
            crs_epsg = None
            if "EPSG:" in crs_str.upper():
                try:
                    crs_epsg = int(crs_str.upper().split("EPSG:")[1].split()[0].strip("),:;"))
                except Exception:
                    pass

            # 裁剪（含 10km×10km 空间约束 + 近邻节点过滤）
            G_final, pos_final, utm_centroid = crop_to_target_size(
                G_undi, pos_raw,
                target_nodes=target_nodes,
                target_edges=target_edges,
                max_span_m=max_span_m,
                min_node_dist_m=min_node_dist_m,
                seed=idx * 7 + 42,   # 每个算例不同种子，增加多样性
            )

            road_nodes   = G_final.number_of_nodes()
            actual_edges = G_final.number_of_edges()
            total_nodes  = road_nodes + 1

            if actual_edges == 0 or road_nodes < 2:
                print(f"  ⚠ 路网节点/边不足，跳过 {name}")
                continue

            # 校验空间跨度
            coords_chk = np.array(list(pos_final.values()))
            x_sp = coords_chk[:, 0].max() - coords_chk[:, 0].min()
            y_sp = coords_chk[:, 1].max() - coords_chk[:, 1].min()
            print(f"  路网跨度: X={x_sp:.0f}m, Y={y_sp:.0f}m"
                  f"  {'✓ 满足10km限制' if x_sp <= max_span_m and y_sp <= max_span_m else '⚠ 超出限制（已尽力裁剪）'}")

            print(f"  最终: {road_nodes} 路网节点, {actual_edges} 需求边, "
                  f"含基站共 {total_nodes} 节点")

            # 生成5种基站，保存文件
            all_depot_positions = []
            for depot_id, direction in DEPOT_DIRECTIONS:
                depot_pos = compute_depot_pos(pos_final, direction)
                all_depot_positions.append((depot_id, direction, depot_pos))

                if save_txt:
                    fname = (f"{total_nodes}-{actual_edges}-{num_drones}"
                             f"-{depot_id}-({name}).txt")
                    fpath = os.path.join(output_dir, fname)
                    save_instance(fpath, G_final, pos_final, depot_pos, num_drones)
                    all_files.append(fpath)

            if save_png:
                fig_name = f"{total_nodes}-{actual_edges}-{num_drones}-({name}).png"
                fig_path = os.path.join(output_dir, fig_name)
                title = (f"{name} | nodes={road_nodes}, edges={actual_edges} | "
                         f"X={x_sp/1000:.1f}km × Y={y_sp/1000:.1f}km")
                save_network_figure(fig_path, G_final, pos_final,
                                    all_depot_positions, title=title,
                                    crs_epsg=crs_epsg, utm_centroid=utm_centroid)
                all_files.append(fig_path)

        except Exception as e:
            print(f"  ✗ 生成失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    txt_count = sum(1 for f in all_files if f.endswith(".txt"))
    png_count = sum(1 for f in all_files if f.endswith(".png"))
    print(f"\n{'='*65}")
    print(f"全部完成！共生成 {txt_count} 个算例文件（目标100）+ {png_count} 张图")
    print(f"输出目录: {output_dir}")
    print(f"{'='*65}")
    return all_files


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    """
    运行前请确认已安装依赖：
        pip install osmnx networkx numpy matplotlib pyproj

    直接运行本脚本，将自动从广州、成都、上海、深圳共20个地点提取路网，
    生成100个大规模实际算例（20路网 × 5基站方向），保存到：
        算例/实际算例/2-Large/

    如需自定义使用，也可调用以下函数：
        extract_and_save(...)   单个算例
        batch_extract(...)      自定义批量
    """

    # ============================================================
    # 生成100个大规模实际算例（默认入口）
    # ============================================================
    generate_large_practical_instances(
        num_drones=2,
        network_scene="drive",   # 使用 OSMnx 默认可驾驶路网
        save_txt=True,
        save_png=True,
        max_span_m=10000.0,      # 路网跨度限制：10km × 10km
        min_node_dist_m=100.0,   # 节点间最小距离：100m，过滤双向车道重合节点
    )

    # ============================================================
    # 以下为其他示例，按需取消注释
    # ============================================================

    # ----------------------------------------------------------
    # 示例 A：单地点提取，不裁剪，适合先探索规模
    # ----------------------------------------------------------
    # extract_and_save(
    #     query="Futian District, Shenzhen, China",
    #     query_type="place",
    #     network_scene="urban",
    #     num_drones=2,
    #     instance_name="Shenzhen_Futian_full",
    # )

    # ----------------------------------------------------------
    # 示例 B：单地点提取并裁剪到指定规模
    # ----------------------------------------------------------
    # extract_and_save(
    #     query=(23.1291, 113.2644),
    #     query_type="point",
    #     dist=4500,
    #     network_scene="drive",
    #     target_nodes=30,
    #     target_edges=50,
    #     max_span_m=10000.0,
    #     num_drones=2,
    #     instance_name="GZ_Tianhe_30n50e",
    # )

    # ----------------------------------------------------------
    # 示例 C：自定义批量提取
    # ----------------------------------------------------------
    # results = batch_extract(
    #     queries=[
    #         {"query": (22.5431, 114.0579), "query_type": "point",
    #          "dist": 4500, "name": "SZ_Nanshan"},
    #     ],
    #     scale_configs=[
    #         {"target_nodes": 30, "target_edges": 50, "scale_label": "2-Large"},
    #     ],
    #     num_drones=2,
    #     network_scene="drive",
    # )

