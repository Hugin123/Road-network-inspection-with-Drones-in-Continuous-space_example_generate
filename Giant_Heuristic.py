"""
Giant Route + Split 元启发式算法
解决无人机弧路径问题（连续空间断点决策）

编码方式：Giant Route + Split
- Giant Route：所有子边的全局有序排列（不含方向，方向由贪婪确定）
- Split：DP 将 Giant Route 最优切割为多段，每段分给一架无人机
- ALNS：操作 Giant Route 排列（删除/插入/2-opt/or-opt）
- PSO：固定 Giant Route 顺序和断点存在性，只优化断点位置 λ
"""

import math
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ============================================================
# 1. 数据读取与问题表示
# ============================================================

@dataclass
class Instance:
    """算例数据"""
    num_depots: int
    num_road_nodes: int
    total_nodes: int
    num_edges: int
    num_drones: int
    battery: float
    speed: float
    energy_cost: float
    call_cost: float
    inspect_coef: float
    transfer_coef: float
    big_m: float
    x: np.ndarray
    y: np.ndarray
    edges: List[Tuple[int, int]] = field(default_factory=list)
    depot_idx: int = 0

    def edge_length(self, edge_idx: int) -> float:
        u, v = self.edges[edge_idx]
        ux, uy = self.node_coord(u)
        vx, vy = self.node_coord(v)
        return math.hypot(ux - vx, uy - vy)

    def point_on_edge(self, edge_idx: int, lam: float) -> Tuple[float, float]:
        u, v = self.edges[edge_idx]
        ux, uy = self.node_coord(u)
        vx, vy = self.node_coord(v)
        return ux + lam * (vx - ux), uy + lam * (vy - uy)

    def euclidean(self, ax: float, ay: float, bx: float, by: float) -> float:
        return math.hypot(ax - bx, ay - by)

    def node_coord(self, node_idx: int) -> Tuple[float, float]:
        return float(self.x[node_idx]), float(self.y[node_idx])


def parse_instance(filepath: str) -> Instance:
    """从txt文件解析算例"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines()]
    non_empty = [l for l in lines if l]
    idx = 0
    num_depots     = int(non_empty[idx]);   idx += 1
    num_road_nodes = int(non_empty[idx]);   idx += 1
    total_nodes    = int(non_empty[idx]);   idx += 1
    num_edges      = int(non_empty[idx]);   idx += 1
    num_drones     = int(non_empty[idx]);   idx += 1
    battery        = float(non_empty[idx]); idx += 1
    speed          = float(non_empty[idx]); idx += 1
    energy_cost    = float(non_empty[idx]); idx += 1
    call_cost      = float(non_empty[idx]); idx += 1
    inspect_coef   = float(non_empty[idx]); idx += 1
    transfer_coef  = float(non_empty[idx]); idx += 1

    big_m = 10000.0
    extra_params = []
    while idx < len(non_empty):
        line = non_empty[idx]
        if ',' in line or '(' in line:
            break
        try:
            extra_params.append(float(line))
            idx += 1
        except ValueError:
            break
    if len(extra_params) == 1:
        big_m = extra_params[0]
    elif len(extra_params) >= 2:
        big_m = extra_params[-1]

    x_list = [float(v.strip()) for v in non_empty[idx].split(',')]; idx += 1
    y_list = [float(v.strip()) for v in non_empty[idx].split(',')]; idx += 1
    x_arr = np.array(x_list)
    y_arr = np.array(y_list)

    edges = []
    while idx < len(non_empty):
        token = non_empty[idx].strip(); idx += 1
        token = token.replace('(', '').replace(')', '')
        parts = token.split(',')
        u, v = int(parts[0]), int(parts[1])
        if u == 0 and v == 0:
            continue
        edges.append((u, v))

    return Instance(
        num_depots=num_depots, num_road_nodes=num_road_nodes,
        total_nodes=total_nodes, num_edges=len(edges),
        num_drones=num_drones, battery=battery, speed=speed,
        energy_cost=energy_cost, call_cost=call_cost,
        inspect_coef=inspect_coef, transfer_coef=transfer_coef,
        big_m=big_m, x=x_arr, y=y_arr, edges=edges, depot_idx=0,
    )


# ============================================================
# 2. 基础数据结构
# ============================================================

@dataclass
class SubEdge:
    """
    子边：原始边的一段
    seg=0: 整条边(无断点)  seg=1: 第一段(u→bp)  seg=2: 第二段(bp→v)
    ax,ay / bx,by 为子边两端坐标（飞行可正反两方向）
    """
    origin_edge_idx: int
    seg: int
    ax: float
    ay: float
    bx: float
    by: float

    @property
    def length(self) -> float:
        return math.hypot(self.ax - self.bx, self.ay - self.by)


@dataclass
class DroneRoute:
    """单架无人机路径（用于最终 Solution 输出）"""
    sub_edges: List[SubEdge] = field(default_factory=list)
    directions: List[bool] = field(default_factory=list)  # True: a->b

    def start_point(self, i: int) -> Tuple[float, float]:
        se = self.sub_edges[i]
        return (se.ax, se.ay) if self.directions[i] else (se.bx, se.by)

    def end_point(self, i: int) -> Tuple[float, float]:
        se = self.sub_edges[i]
        return (se.bx, se.by) if self.directions[i] else (se.ax, se.ay)


class Solution:
    """
    完整解（用于费用计算和输出）
    breakpoints[i]: 第i条原始边的断点位置（None=无断点）
    routes: 每架无人机的路径
    """
    def __init__(self, num_drones: int, num_edges: int):
        self.num_drones = num_drones
        self.num_edges = num_edges
        self.breakpoints: List[Optional[float]] = [None] * num_edges
        self.routes: List[DroneRoute] = [DroneRoute() for _ in range(num_drones)]
        self._cost: Optional[float] = None

    def invalidate_cache(self):
        self._cost = None

    def copy(self) -> 'Solution':
        sol = Solution(self.num_drones, self.num_edges)
        sol.breakpoints = self.breakpoints.copy()
        sol.routes = [DroneRoute(
            sub_edges=list(r.sub_edges),
            directions=list(r.directions)
        ) for r in self.routes]
        sol._cost = self._cost
        return sol


class GiantRouteSolution:
    """
    Giant Route 编码的解（ALNS 操作的核心对象）：
    - giant_route: 有序子边列表（不含方向），ALNS 直接操作该排列
    - breakpoints: 每条原始边的断点位置（None=无断点，λ∈(0,1)=有断点）
    """
    def __init__(self, num_edges: int):
        self.num_edges = num_edges
        self.giant_route: List[SubEdge] = []
        self.breakpoints: List[Optional[float]] = [None] * num_edges
        self._cost: Optional[float] = None

    def invalidate_cache(self):
        self._cost = None

    def copy(self) -> 'GiantRouteSolution':
        gs = GiantRouteSolution(self.num_edges)
        gs.giant_route = list(self.giant_route)
        gs.breakpoints = list(self.breakpoints)
        gs._cost = self._cost
        return gs


# ============================================================
# 3. 子边构建工具
# ============================================================

def build_sub_edges(inst: Instance,
                    breakpoints: List[Optional[float]]) -> List[SubEdge]:
    """
    根据断点配置构建所有子边列表。
    无断点→1条子边(seg=0)；有断点→2条子边(seg=1, seg=2)。
    """
    sub_edges = []
    for ei, (u, v) in enumerate(inst.edges):
        ux, uy = inst.node_coord(u)
        vx, vy = inst.node_coord(v)
        lam = breakpoints[ei]
        if lam is None:
            sub_edges.append(SubEdge(origin_edge_idx=ei, seg=0,
                                     ax=ux, ay=uy, bx=vx, by=vy))
        else:
            bpx = ux + lam * (vx - ux)
            bpy = uy + lam * (vy - uy)
            sub_edges.append(SubEdge(origin_edge_idx=ei, seg=1,
                                     ax=ux, ay=uy, bx=bpx, by=bpy))
            sub_edges.append(SubEdge(origin_edge_idx=ei, seg=2,
                                     ax=bpx, ay=bpy, bx=vx, by=vy))
    return sub_edges


def rebuild_sub_edges_in_giant_route(
    gs: GiantRouteSolution, inst: Instance,
    new_breakpoints: List[Optional[float]]
) -> GiantRouteSolution:
    """
    断点位置更新后，重建 Giant Route 中所有子边的坐标。
    Giant Route 的顺序和断点存在性保持不变，只更新子边端点坐标。
    """
    new_gs = GiantRouteSolution(gs.num_edges)
    new_gs.breakpoints = list(new_breakpoints)
    for se in gs.giant_route:
        ei = se.origin_edge_idx
        u, v = inst.edges[ei]
        ux, uy = inst.node_coord(u)
        vx, vy = inst.node_coord(v)
        lam = new_breakpoints[ei]
        if se.seg == 0:
            new_se = SubEdge(origin_edge_idx=ei, seg=0,
                             ax=ux, ay=uy, bx=vx, by=vy)
        else:
            # 断点存在性未变，只更新断点坐标
            if lam is not None:
                bpx = ux + lam * (vx - ux)
                bpy = uy + lam * (vy - uy)
            else:
                # 理论上不应发生（PSO不改变断点存在性）
                bpx, bpy = (ux + vx) / 2, (uy + vy) / 2
            if se.seg == 1:
                new_se = SubEdge(origin_edge_idx=ei, seg=1,
                                 ax=ux, ay=uy, bx=bpx, by=bpy)
            else:
                new_se = SubEdge(origin_edge_idx=ei, seg=2,
                                 ax=bpx, ay=bpy, bx=vx, by=vy)
        new_gs.giant_route.append(new_se)
    return new_gs


# ============================================================
# 4. 方向贪婪确定
# ============================================================

def assign_directions_greedy(
    sub_edges: List[SubEdge],
    depot_x: float, depot_y: float
) -> List[bool]:
    """
    对一段有序子边序列，从 depot 出发，贪婪地为每条子边选择方向：
    选择离当前位置最近的端点作为入端。
    返回方向列表（True: a→b, False: b→a）。
    """
    directions = []
    cx, cy = depot_x, depot_y
    for se in sub_edges:
        dist_a = math.hypot(cx - se.ax, cy - se.ay)
        dist_b = math.hypot(cx - se.bx, cy - se.by)
        if dist_a <= dist_b:
            directions.append(True)
            cx, cy = se.bx, se.by
        else:
            directions.append(False)
            cx, cy = se.ax, se.ay
    return directions


# ============================================================
# 5. 费用计算
# ============================================================

def compute_route_distance(route: DroneRoute, inst: Instance) -> float:
    """计算单条路径的能量消耗（用于电池约束检查）"""
    if not route.sub_edges:
        return 0.0
    depot_x, depot_y = inst.node_coord(inst.depot_idx)
    n = len(route.sub_edges)
    inspect_dist = sum(se.length for se in route.sub_edges)
    sx, sy = route.start_point(0)
    transfer_dist = math.hypot(depot_x - sx, depot_y - sy)
    for i in range(n - 1):
        ex, ey = route.end_point(i)
        nsx, nsy = route.start_point(i + 1)
        transfer_dist += math.hypot(ex - nsx, ey - nsy)
    ex, ey = route.end_point(n - 1)
    transfer_dist += math.hypot(ex - depot_x, ey - depot_y)
    return inspect_dist * inst.inspect_coef + transfer_dist * inst.transfer_coef


def compute_route_raw_distance(route: DroneRoute, inst: Instance) -> Tuple[float, float]:
    """返回 (inspect_dist, transfer_dist)"""
    if not route.sub_edges:
        return 0.0, 0.0
    depot_x, depot_y = inst.node_coord(inst.depot_idx)
    n = len(route.sub_edges)
    inspect_dist = sum(se.length for se in route.sub_edges)
    sx, sy = route.start_point(0)
    transfer_dist = math.hypot(depot_x - sx, depot_y - sy)
    for i in range(n - 1):
        ex, ey = route.end_point(i)
        nsx, nsy = route.start_point(i + 1)
        transfer_dist += math.hypot(ex - nsx, ey - nsy)
    ex, ey = route.end_point(n - 1)
    transfer_dist += math.hypot(ex - depot_x, ey - depot_y)
    return inspect_dist, transfer_dist


def compute_route_cost(route: DroneRoute, inst: Instance) -> float:
    """计算单条路径费用（含 call_cost）"""
    if not route.sub_edges:
        return 0.0
    depot_x, depot_y = inst.node_coord(inst.depot_idx)
    n = len(route.sub_edges)
    cost = inst.call_cost
    sx, sy = route.start_point(0)
    cost += math.hypot(depot_x - sx, depot_y - sy) * inst.transfer_coef * inst.energy_cost
    for i in range(n):
        cost += route.sub_edges[i].length * inst.inspect_coef * inst.energy_cost
        if i + 1 < n:
            ex, ey = route.end_point(i)
            nsx, nsy = route.start_point(i + 1)
            cost += math.hypot(ex - nsx, ey - nsy) * inst.transfer_coef * inst.energy_cost
    ex, ey = route.end_point(n - 1)
    cost += math.hypot(ex - depot_x, ey - depot_y) * inst.transfer_coef * inst.energy_cost
    return cost


def compute_cost(sol: Solution, inst: Instance) -> float:
    """计算 Solution 的总费用"""
    if sol._cost is not None:
        return sol._cost
    total = sum(compute_route_cost(r, inst) for r in sol.routes if r.sub_edges)
    sol._cost = total
    return total


def _compute_segment_cost(
    sub_edges: List[SubEdge],
    inst: Instance,
    depot_x: float, depot_y: float
) -> Tuple[float, float]:
    """
    计算一段子边（从 depot 出发、方向贪婪确定）的费用和能量消耗。
    返回 (cost_with_call, energy)
    """
    if not sub_edges:
        return 0.0, 0.0
    n = len(sub_edges)
    dirs = assign_directions_greedy(sub_edges, depot_x, depot_y)

    def start(i):
        se = sub_edges[i]
        return (se.ax, se.ay) if dirs[i] else (se.bx, se.by)

    def end(i):
        se = sub_edges[i]
        return (se.bx, se.by) if dirs[i] else (se.ax, se.ay)

    inspect_dist = sum(se.length for se in sub_edges)
    sx, sy = start(0)
    transfer_dist = math.hypot(depot_x - sx, depot_y - sy)
    for i in range(n - 1):
        ex, ey = end(i)
        nsx, nsy = start(i + 1)
        transfer_dist += math.hypot(ex - nsx, ey - nsy)
    ex, ey = end(n - 1)
    transfer_dist += math.hypot(ex - depot_x, ey - depot_y)

    energy = inspect_dist * inst.inspect_coef + transfer_dist * inst.transfer_coef
    cost = (inst.call_cost
            + inspect_dist * inst.inspect_coef * inst.energy_cost
            + transfer_dist * inst.transfer_coef * inst.energy_cost)
    return cost, energy


# ============================================================
# 6. Split DP 算法
# ============================================================

def split_dp(
    giant_route: List[SubEdge],
    inst: Instance,
    max_drones: int
) -> Solution:
    """
    Split DP：将 Giant Route 最优切割为最多 max_drones 段，
    每段分给一架无人机，最小化总费用（含 call_cost）。

    DP 状态：dp[i] = 将前i条子边最优分配后的最小总费用
    转移：dp[i] = min_{0<=j<i} { dp[j] + cost(giant_route[j..i-1]) }
          其中 cost() 含 call_cost，且能量消耗 <= battery

    若某单条子边无法被单架无人机完成则抛出 ValueError。
    若所有切割方案都超出 max_drones 架无人机则抛出 ValueError。
    """
    n = len(giant_route)
    if n == 0:
        return Solution(inst.num_drones, inst.num_edges)

    depot_x, depot_y = inst.node_coord(inst.depot_idx)
    INF = float('inf')
    dp = [INF] * (n + 1)
    prev_cut = [-1] * (n + 1)  # prev_cut[i] = j 表示段 [j, i-1] 是一段
    dp[0] = 0.0

    # 为加速，预先检查每条单独子边是否可行
    for k, se in enumerate(giant_route):
        dirs_single = assign_directions_greedy([se], depot_x, depot_y)
        if dirs_single[0]:
            sx, sy = se.ax, se.ay
            ex, ey = se.bx, se.by
        else:
            sx, sy = se.bx, se.by
            ex, ey = se.ax, se.ay
        energy_single = (se.length * inst.inspect_coef
                         + (math.hypot(depot_x - sx, depot_y - sy)
                            + math.hypot(ex - depot_x, ey - depot_y))
                         * inst.transfer_coef)
        if energy_single > inst.battery:
            raise ValueError(
                f"子边 origin_edge={se.origin_edge_idx}(seg={se.seg}) "
                f"单独飞行能量 {energy_single:.4f} > 电池容量 {inst.battery:.4f}，"
                f"请检查断点配置或电池参数。"
            )

    for i in range(n):
        if dp[i] == INF:
            continue

        # 增量构建段 [i, k]，贪婪方向从 depot 出发
        cx, cy = depot_x, depot_y
        inspect_dist = 0.0
        transfer_dist_to_first = 0.0  # depot 到第一条子边起点
        transfer_dist_between = 0.0   # 子边之间转移
        first_sx, first_sy = None, None

        for k in range(i, n):
            se = giant_route[k]
            # 贪婪确定方向
            dist_a = math.hypot(cx - se.ax, cy - se.ay)
            dist_b = math.hypot(cx - se.bx, cy - se.by)
            if dist_a <= dist_b:
                sx, sy = se.ax, se.ay
                ex, ey = se.bx, se.by
            else:
                sx, sy = se.bx, se.by
                ex, ey = se.ax, se.ay

            if k == i:
                # 第一条子边：depot → 子边起点
                first_sx, first_sy = sx, sy
                transfer_dist_to_first = math.hypot(depot_x - sx, depot_y - sy)
                transfer_dist_between = 0.0
            else:
                # 后续子边：前一条子边终点 → 当前子边起点
                transfer_dist_between += math.hypot(cx - sx, cy - sy)

            inspect_dist += se.length
            cx, cy = ex, ey

            # 到基站的距离
            to_depot = math.hypot(cx - depot_x, cy - depot_y)

            total_transfer = transfer_dist_to_first + transfer_dist_between + to_depot
            energy = inspect_dist * inst.inspect_coef + total_transfer * inst.transfer_coef

            # 能量超限则停止扩展（后续更长的段也不可行）
            if energy > inst.battery:
                break

            seg_cost = (inst.call_cost
                        + inspect_dist * inst.inspect_coef * inst.energy_cost
                        + total_transfer * inst.transfer_coef * inst.energy_cost)

            new_cost = dp[i] + seg_cost
            if new_cost < dp[k + 1]:
                dp[k + 1] = new_cost
                prev_cut[k + 1] = i

    if dp[n] == INF:
        raise ValueError(
            f"Split DP 无法在 {max_drones} 架无人机内完成所有子边的覆盖，"
            f"请增加无人机数量或检查问题参数。"
        )

    # 回溯切割点
    segments: List[Tuple[int, int]] = []
    cur = n
    while cur > 0:
        j = prev_cut[cur]
        segments.append((j, cur - 1))
        cur = j
    segments.reverse()

    if len(segments) > max_drones:
        raise ValueError(
            f"Split DP 最优切割需要 {len(segments)} 架无人机，超过上限 {max_drones}。"
        )

    # 构建 Solution
    sol = Solution(inst.num_drones, inst.num_edges)
    for drone_idx, (si, ei_) in enumerate(segments):
        seg_ses = giant_route[si: ei_ + 1]
        dirs = assign_directions_greedy(seg_ses, depot_x, depot_y)
        sol.routes[drone_idx] = DroneRoute(
            sub_edges=list(seg_ses),
            directions=dirs
        )
    sol.invalidate_cache()
    return sol


def gs_to_solution(gs: GiantRouteSolution, inst: Instance) -> Solution:
    """将 GiantRouteSolution 解码为 Solution（调用 Split DP）"""
    sol = split_dp(gs.giant_route, inst, inst.num_drones)
    sol.breakpoints = list(gs.breakpoints)
    return sol


def evaluate_gs(gs: GiantRouteSolution, inst: Instance) -> float:
    """评估 GiantRouteSolution 的总费用（解码 + 计算费用）"""
    if gs._cost is not None:
        return gs._cost
    sol = gs_to_solution(gs, inst)
    cost = compute_cost(sol, inst)
    gs._cost = cost
    return cost


# ============================================================
# 7. 初始 Giant Route 生成
# ============================================================

def _greedy_giant_route_nn(sub_edges: List[SubEdge],
                            inst: Instance) -> List[SubEdge]:
    """方案B：全局最近邻贪婪构建 Giant Route"""
    depot_x, depot_y = inst.node_coord(inst.depot_idx)
    unvisited = list(sub_edges)
    giant = []
    cx, cy = depot_x, depot_y
    while unvisited:
        best_idx = -1
        best_dist = float('inf')
        for i, se in enumerate(unvisited):
            d = min(math.hypot(cx - se.ax, cy - se.ay),
                    math.hypot(cx - se.bx, cy - se.by))
            if d < best_dist:
                best_dist = d
                best_idx = i
        se = unvisited.pop(best_idx)
        giant.append(se)
        da = math.hypot(cx - se.ax, cy - se.ay)
        db = math.hypot(cx - se.bx, cy - se.by)
        cx, cy = (se.bx, se.by) if da <= db else (se.ax, se.ay)
    return giant


def _greedy_multi_route_solution(sub_edges: List[SubEdge],
                                  inst: Instance) -> 'Solution':
    """贪心最近邻多路径解（用于方案A拼接）"""
    depot_x, depot_y = inst.node_coord(inst.depot_idx)
    n_drones = inst.num_drones
    sol = Solution(n_drones, inst.num_edges)
    unassigned = list(sub_edges)
    curr_pos = [(depot_x, depot_y)] * n_drones
    curr_energy = [0.0] * n_drones

    max_outer = len(sub_edges) * n_drones * 2
    attempt = 0
    while unassigned and attempt < max_outer:
        attempt += 1
        placed = False
        drone_order = sorted(range(n_drones), key=lambda d: curr_energy[d])
        for di in drone_order:
            cx, cy = curr_pos[di]
            best_uid = -1
            best_metric = float('inf')
            best_dir = True
            for uid, se in enumerate(unassigned):
                da = math.hypot(cx - se.ax, cy - se.ay)
                db = math.hypot(cx - se.bx, cy - se.by)
                direction = da <= db
                sx, sy = (se.ax, se.ay) if direction else (se.bx, se.by)
                ex, ey = (se.bx, se.by) if direction else (se.ax, se.ay)
                to_start = min(da, db)
                to_depot = math.hypot(ex - depot_x, ey - depot_y)
                energy_add = to_start * inst.transfer_coef + se.length * inst.inspect_coef
                to_depot_e = to_depot * inst.transfer_coef
                if curr_energy[di] + energy_add + to_depot_e <= inst.battery:
                    if to_start < best_metric:
                        best_metric = to_start
                        best_uid = uid
                        best_dir = direction
            if best_uid >= 0:
                se = unassigned.pop(best_uid)
                sol.routes[di].sub_edges.append(se)
                sol.routes[di].directions.append(best_dir)
                sx, sy = (se.ax, se.ay) if best_dir else (se.bx, se.by)
                ex, ey = (se.bx, se.by) if best_dir else (se.ax, se.ay)
                curr_energy[di] += (math.hypot(curr_pos[di][0] - sx, curr_pos[di][1] - sy)
                                    * inst.transfer_coef
                                    + se.length * inst.inspect_coef)
                curr_pos[di] = (ex, ey)
                placed = True
                break
        if not placed:
            di = min(range(n_drones), key=lambda d: curr_energy[d])
            se = unassigned.pop(0)
            da = math.hypot(curr_pos[di][0] - se.ax, curr_pos[di][1] - se.ay)
            db = math.hypot(curr_pos[di][0] - se.bx, curr_pos[di][1] - se.by)
            direction = da <= db
            ex, ey = (se.bx, se.by) if direction else (se.ax, se.ay)
            sol.routes[di].sub_edges.append(se)
            sol.routes[di].directions.append(direction)
            curr_energy[di] += min(da, db) * inst.transfer_coef + se.length * inst.inspect_coef
            curr_pos[di] = (ex, ey)
    sol.invalidate_cache()
    return sol


def generate_initial_gs(
    inst: Instance,
    breakpoints: List[Optional[float]],
    strategy: str = 'nearest_neighbor'
) -> GiantRouteSolution:
    """
    生成初始 GiantRouteSolution。
    strategy: 'nearest_neighbor' | 'multi_route' | 'random'
    """
    sub_edges = build_sub_edges(inst, breakpoints)
    gs = GiantRouteSolution(inst.num_edges)
    gs.breakpoints = list(breakpoints)
    if strategy == 'nearest_neighbor':
        gs.giant_route = _greedy_giant_route_nn(sub_edges, inst)
    elif strategy == 'multi_route':
        sol = _greedy_multi_route_solution(sub_edges, inst)
        gs.giant_route = []
        for route in sol.routes:
            gs.giant_route.extend(route.sub_edges)
        # 补充未进入任何路径的子边（强制分配情况）
        assigned = {id(se) for r in sol.routes for se in r.sub_edges}
        for se in sub_edges:
            if id(se) not in assigned:
                gs.giant_route.append(se)
    elif strategy == 'random':
        gs.giant_route = list(sub_edges)
        random.shuffle(gs.giant_route)
    else:
        gs.giant_route = _greedy_giant_route_nn(sub_edges, inst)
    return gs


def multi_start_initial_gs(inst: Instance, n_starts: int = 6) -> GiantRouteSolution:
    """
    多启动初始解生成：尝试多种断点配置和构建策略，返回费用最低的解。
    """
    candidates: List[Tuple[float, GiantRouteSolution]] = []

    def _try(bps: List[Optional[float]], strategy: str):
        try:
            gs = generate_initial_gs(inst, bps, strategy)
            cost = evaluate_gs(gs, inst)
            candidates.append((cost, gs))
        except Exception:
            pass

    # 无断点
    bps_none = [None] * inst.num_edges
    _try(bps_none, 'nearest_neighbor')
    _try(bps_none, 'multi_route')

    # 全部断点在 0.5
    bps_all = [0.5] * inst.num_edges
    _try(bps_all, 'nearest_neighbor')
    _try(bps_all, 'multi_route')

    # 随机断点配置
    for _ in range(max(0, n_starts - 4)):
        bps_rand = [
            random.uniform(0.15, 0.85) if random.random() < 0.4 else None
            for _ in range(inst.num_edges)
        ]
        _try(bps_rand, 'nearest_neighbor')

    if not candidates:
        raise ValueError("多启动初始解生成失败，请检查算例参数。")

    _, best_gs = min(candidates, key=lambda x: x[0])
    return best_gs


# ============================================================
# 8. ALNS 破坏算子（操作 Giant Route 排列）
# ============================================================

def destroy_random_removal(
    gs: GiantRouteSolution, inst: Instance,
    removal_fraction: float = 0.3
) -> Tuple[GiantRouteSolution, List[SubEdge]]:
    """随机移除一定比例的子边"""
    new_gs = gs.copy()
    n = len(new_gs.giant_route)
    if n == 0:
        return new_gs, []
    num_remove = max(1, int(n * removal_fraction))
    num_remove = min(num_remove, n)
    remove_indices = sorted(
        random.sample(range(n), num_remove), reverse=True
    )
    removed = []
    for idx in remove_indices:
        removed.append(new_gs.giant_route.pop(idx))
    new_gs.invalidate_cache()
    return new_gs, removed


def destroy_worst_removal(
    gs: GiantRouteSolution, inst: Instance,
    removal_fraction: float = 0.3
) -> Tuple[GiantRouteSolution, List[SubEdge]]:
    """
    移除对总费用贡献最大的子边（近似：移除转移代价最高的子边）。
    在 Giant Route 中，用贪婪方向估算每条子边的"转移代价"：
    = 到该子边的转移 + 该子边巡检 + 该子边到下一条的转移
      - （跳过该子边时前后直接转移）
    """
    new_gs = gs.copy()
    n = len(new_gs.giant_route)
    if n == 0:
        return new_gs, []

    depot_x, depot_y = inst.node_coord(inst.depot_idx)
    # 贪婪确定整条 Giant Route 的方向
    dirs = assign_directions_greedy(new_gs.giant_route, depot_x, depot_y)

    def get_start(i):
        se = new_gs.giant_route[i]
        return (se.ax, se.ay) if dirs[i] else (se.bx, se.by)

    def get_end(i):
        se = new_gs.giant_route[i]
        return (se.bx, se.by) if dirs[i] else (se.ax, se.ay)

    scores = []
    for i in range(n):
        se = new_gs.giant_route[i]
        sx, sy = get_start(i)
        ex, ey = get_end(i)
        prev_x = depot_x if i == 0 else get_end(i - 1)[0]
        prev_y = depot_y if i == 0 else get_end(i - 1)[1]
        next_x = depot_x if i == n - 1 else get_start(i + 1)[0]
        next_y = depot_y if i == n - 1 else get_start(i + 1)[1]

        current_cost = (math.hypot(prev_x - sx, prev_y - sy)
                        + se.length
                        + math.hypot(ex - next_x, ey - next_y))
        bypass_cost = math.hypot(prev_x - next_x, prev_y - next_y)
        score = (current_cost - bypass_cost) * inst.transfer_coef + \
                se.length * inst.inspect_coef
        scores.append((score, i))

    scores.sort(key=lambda x: -x[0])
    num_remove = max(1, int(n * removal_fraction))
    remove_indices = sorted([idx for _, idx in scores[:num_remove]], reverse=True)

    removed = []
    for idx in remove_indices:
        removed.append(new_gs.giant_route.pop(idx))
    new_gs.invalidate_cache()
    return new_gs, removed


def destroy_segment_removal(
    gs: GiantRouteSolution, inst: Instance,
    removal_fraction: float = 0.3
) -> Tuple[GiantRouteSolution, List[SubEdge]]:
    """
    连续片段移除：从 Giant Route 中移除一段连续子边。
    对应原算法的 route_removal（移除一架无人机的整段路径）。
    """
    new_gs = gs.copy()
    n = len(new_gs.giant_route)
    if n == 0:
        return new_gs, []
    num_remove = max(1, int(n * removal_fraction))
    num_remove = min(num_remove, n)
    start_i = random.randint(0, n - num_remove)
    removed = new_gs.giant_route[start_i: start_i + num_remove]
    new_gs.giant_route = (new_gs.giant_route[:start_i]
                          + new_gs.giant_route[start_i + num_remove:])
    new_gs.invalidate_cache()
    return new_gs, removed


# ============================================================
# 9. ALNS 修复算子（将子边插回 Giant Route）
# ============================================================

def repair_greedy_insert(
    gs: GiantRouteSolution, inst: Instance,
    removed: List[SubEdge]
) -> GiantRouteSolution:
    """
    贪婪插入：对每条被移除的子边，找到插入 Giant Route 后
    使局部转移费用增量最小的位置。
    插入后不立即调用 Split，保持 Giant Route 编码。
    """
    new_gs = gs.copy()
    depot_x, depot_y = inst.node_coord(inst.depot_idx)
    random.shuffle(removed)

    for se in removed:
        n = len(new_gs.giant_route)
        best_delta = float('inf')
        best_pos = 0

        for pos in range(n + 1):
            # 前驱端点（贪婪方向下的出端）
            if pos == 0:
                prev_x, prev_y = depot_x, depot_y
            else:
                prev_se = new_gs.giant_route[pos - 1]
                # 估计前驱子边的出端（用最近端点到后继的方式估算）
                # 简化：直接用前驱子边两端中离当前 se 更近的端作为出端
                # 采用贪婪方向估算
                if pos == 1:
                    px0, py0 = depot_x, depot_y
                else:
                    ppse = new_gs.giant_route[pos - 2]
                    da = math.hypot(depot_x - ppse.ax, depot_y - ppse.ay)
                    db = math.hypot(depot_x - ppse.bx, depot_y - ppse.by)
                    px0 = ppse.bx if da <= db else ppse.ax
                    py0 = ppse.by if da <= db else ppse.ay
                da = math.hypot(px0 - prev_se.ax, py0 - prev_se.ay)
                db = math.hypot(px0 - prev_se.bx, py0 - prev_se.by)
                prev_x = prev_se.bx if da <= db else prev_se.ax
                prev_y = prev_se.by if da <= db else prev_se.ay

            # 后继端点
            if pos == n:
                next_x, next_y = depot_x, depot_y
            else:
                next_se = new_gs.giant_route[pos]
                da = math.hypot(prev_x - next_se.ax, prev_y - next_se.ay)
                db = math.hypot(prev_x - next_se.bx, prev_y - next_se.by)
                next_x = next_se.ax if da <= db else next_se.bx
                next_y = next_se.ay if da <= db else next_se.by

            # se 的最近端作为入端
            da = math.hypot(prev_x - se.ax, prev_y - se.ay)
            db = math.hypot(prev_x - se.bx, prev_y - se.by)
            if da <= db:
                sx, sy = se.ax, se.ay
                ex, ey = se.bx, se.by
            else:
                sx, sy = se.bx, se.by
                ex, ey = se.ax, se.ay

            old_transfer = math.hypot(prev_x - next_x, prev_y - next_y)
            new_transfer = (math.hypot(prev_x - sx, prev_y - sy)
                            + se.length
                            + math.hypot(ex - next_x, ey - next_y))
            delta = ((new_transfer - old_transfer) * inst.transfer_coef
                     + se.length * inst.inspect_coef) * inst.energy_cost

            if delta < best_delta:
                best_delta = delta
                best_pos = pos

        new_gs.giant_route.insert(best_pos, se)

    new_gs.invalidate_cache()
    return new_gs


def repair_random_insert(
    gs: GiantRouteSolution, inst: Instance,
    removed: List[SubEdge]
) -> GiantRouteSolution:
    """随机插入：将被移除的子边插入 Giant Route 的随机位置"""
    new_gs = gs.copy()
    random.shuffle(removed)
    for se in removed:
        pos = random.randint(0, len(new_gs.giant_route))
        new_gs.giant_route.insert(pos, se)
    new_gs.invalidate_cache()
    return new_gs


def repair_regret_insert(
    gs: GiantRouteSolution, inst: Instance,
    removed: List[SubEdge]
) -> GiantRouteSolution:
    """
    Regret-2 插入：每次选择"最优插入位置费用"与"次优插入位置费用"之差最大的子边优先插入，
    以减少未来插入代价（后悔值启发）。
    """
    new_gs = gs.copy()
    depot_x, depot_y = inst.node_coord(inst.depot_idx)
    remaining = list(removed)

    def _insert_delta(gs_: GiantRouteSolution, se: SubEdge, pos: int) -> float:
        """计算在 pos 处插入 se 的转移费用增量"""
        n = len(gs_.giant_route)
        if pos == 0:
            prev_x, prev_y = depot_x, depot_y
        else:
            prev_se = gs_.giant_route[pos - 1]
            if pos == 1:
                px0, py0 = depot_x, depot_y
            else:
                ppse = gs_.giant_route[pos - 2]
                da = math.hypot(depot_x - ppse.ax, depot_y - ppse.ay)
                db = math.hypot(depot_x - ppse.bx, depot_y - ppse.by)
                px0 = ppse.bx if da <= db else ppse.ax
                py0 = ppse.by if da <= db else ppse.ay
            da = math.hypot(px0 - prev_se.ax, py0 - prev_se.ay)
            db = math.hypot(px0 - prev_se.bx, py0 - prev_se.by)
            prev_x = prev_se.bx if da <= db else prev_se.ax
            prev_y = prev_se.by if da <= db else prev_se.ay

        if pos == n:
            next_x, next_y = depot_x, depot_y
        else:
            next_se = gs_.giant_route[pos]
            da = math.hypot(prev_x - next_se.ax, prev_y - next_se.ay)
            db = math.hypot(prev_x - next_se.bx, prev_y - next_se.by)
            next_x = next_se.ax if da <= db else next_se.bx
            next_y = next_se.ay if da <= db else next_se.by

        da = math.hypot(prev_x - se.ax, prev_y - se.ay)
        db = math.hypot(prev_x - se.bx, prev_y - se.by)
        sx, sy = (se.ax, se.ay) if da <= db else (se.bx, se.by)
        ex, ey = (se.bx, se.by) if da <= db else (se.ax, se.ay)

        old_t = math.hypot(prev_x - next_x, prev_y - next_y)
        new_t = (math.hypot(prev_x - sx, prev_y - sy)
                 + se.length
                 + math.hypot(ex - next_x, ey - next_y))
        return ((new_t - old_t) * inst.transfer_coef
                + se.length * inst.inspect_coef) * inst.energy_cost

    while remaining:
        best_se_idx = -1
        best_pos = 0
        best_regret = -float('inf')

        for sei, se in enumerate(remaining):
            n = len(new_gs.giant_route)
            deltas = sorted([_insert_delta(new_gs, se, p) for p in range(n + 1)])
            best_d = deltas[0]
            second_d = deltas[1] if len(deltas) > 1 else deltas[0]
            regret = second_d - best_d
            if regret > best_regret:
                best_regret = regret
                best_se_idx = sei
                # 找最优插入位置
                best_pos = min(range(n + 1),
                               key=lambda p: _insert_delta(new_gs, se, p))

        se = remaining.pop(best_se_idx)
        new_gs.giant_route.insert(best_pos, se)

    new_gs.invalidate_cache()
    return new_gs


# ============================================================
# 10. 局部搜索算子（2-opt 和 or-opt）
# ============================================================

def local_search_2opt(
    gs: GiantRouteSolution, inst: Instance,
    max_no_improve: int = None
) -> GiantRouteSolution:
    """
    Giant Route 上的 2-opt 局部搜索：
    翻转 Giant Route 中一段 [i, j] 的顺序（子序列反转），
    翻转后重新由贪婪确定方向（不保留原方向）。
    接受改进的移动（严格改进）。

    评估方式：直接用 evaluate_gs()（调用 Split DP + 计算费用）。
    """
    best_gs = gs.copy()
    best_cost = evaluate_gs(best_gs, inst)
    n = len(best_gs.giant_route)
    if n < 4:
        return best_gs

    no_improve = 0
    max_ni = max_no_improve if max_no_improve is not None else n * n

    improved = True
    while improved and no_improve < max_ni:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                # 翻转 [i, j]
                new_route = (best_gs.giant_route[:i]
                             + list(reversed(best_gs.giant_route[i: j + 1]))
                             + best_gs.giant_route[j + 1:])
                trial_gs = best_gs.copy()
                trial_gs.giant_route = new_route
                trial_gs.invalidate_cache()
                try:
                    trial_cost = evaluate_gs(trial_gs, inst)
                except Exception:
                    continue
                if trial_cost < best_cost - 1e-9:
                    best_cost = trial_cost
                    best_gs = trial_gs.copy()
                    improved = True
                    no_improve = 0
                    break
            if improved:
                break
        if not improved:
            no_improve += 1

    best_gs._cost = best_cost
    return best_gs


def local_search_or_opt(
    gs: GiantRouteSolution, inst: Instance,
    segment_sizes: List[int] = None,
    max_no_improve: int = None
) -> GiantRouteSolution:
    """
    Giant Route 上的 or-opt 局部搜索：
    将一段连续子边（长度为 k = 1, 2, 3）从当前位置移动到另一个位置，
    翻转后方向由贪婪重新确定。
    接受严格改进的移动。
    """
    if segment_sizes is None:
        segment_sizes = [1, 2, 3]

    best_gs = gs.copy()
    best_cost = evaluate_gs(best_gs, inst)
    n = len(best_gs.giant_route)
    if n < 3:
        return best_gs

    max_ni = max_no_improve if max_no_improve is not None else n * 2

    improved = True
    no_improve = 0
    while improved and no_improve < max_ni:
        improved = False
        for k in segment_sizes:
            if k >= n:
                continue
            for i in range(n - k + 1):
                segment = best_gs.giant_route[i: i + k]
                rest = best_gs.giant_route[:i] + best_gs.giant_route[i + k:]
                for j in range(len(rest) + 1):
                    if j == i:
                        continue  # 原位
                    new_route = rest[:j] + segment + rest[j:]
                    trial_gs = best_gs.copy()
                    trial_gs.giant_route = new_route
                    trial_gs.invalidate_cache()
                    try:
                        trial_cost = evaluate_gs(trial_gs, inst)
                    except Exception:
                        continue
                    if trial_cost < best_cost - 1e-9:
                        best_cost = trial_cost
                        best_gs = trial_gs.copy()
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            no_improve += 1

    best_gs._cost = best_cost
    return best_gs


# ============================================================
# 11. 断点邻域搜索
# ============================================================

def try_add_remove_breakpoints(
    gs: GiantRouteSolution, inst: Instance,
    fast_mode: bool = False
) -> GiantRouteSolution:
    """
    断点邻域搜索：对每条原始边尝试添加/删除/移动断点。
    断点改变后，对应子边的坐标更新（Giant Route 顺序不变），
    然后重新评估费用。

    fast_mode=True：随机选少量边快速尝试
    fast_mode=False：枚举所有边，并尝试多个候选位置
    """
    best_gs = gs.copy()
    best_cost = evaluate_gs(best_gs, inst)

    if fast_mode:
        n_try = min(inst.num_edges, max(2, inst.num_edges // 3))
        edge_order = random.sample(range(inst.num_edges), n_try)
        candidates_for_add = [0.5, random.uniform(0.2, 0.8)]
    else:
        edge_order = list(range(inst.num_edges))
        random.shuffle(edge_order)
        candidates_for_add = [0.25, 0.33, 0.5, 0.67, 0.75,
                               random.uniform(0.1, 0.45),
                               random.uniform(0.55, 0.9)]

    for ei in edge_order:
        current_bps = list(best_gs.breakpoints)

        if current_bps[ei] is None:
            # 尝试添加断点：将 seg=0 的子边替换为 seg=1 + seg=2
            # 找到该边在 giant_route 中的位置
            se_idx = None
            for gi, se in enumerate(best_gs.giant_route):
                if se.origin_edge_idx == ei and se.seg == 0:
                    se_idx = gi
                    break
            if se_idx is None:
                continue

            for lam in candidates_for_add:
                trial_bps = list(current_bps)
                trial_bps[ei] = lam
                # 构建新的 giant_route：将该位置的 seg=0 替换为 seg=1, seg=2
                u, v = inst.edges[ei]
                ux, uy = inst.node_coord(u)
                vx, vy = inst.node_coord(v)
                bpx = ux + lam * (vx - ux)
                bpy = uy + lam * (vy - uy)
                se1 = SubEdge(origin_edge_idx=ei, seg=1,
                              ax=ux, ay=uy, bx=bpx, by=bpy)
                se2 = SubEdge(origin_edge_idx=ei, seg=2,
                              ax=bpx, ay=bpy, bx=vx, by=vy)
                new_route = (best_gs.giant_route[:se_idx]
                             + [se1, se2]
                             + best_gs.giant_route[se_idx + 1:])
                trial_gs = GiantRouteSolution(best_gs.num_edges)
                trial_gs.breakpoints = trial_bps
                trial_gs.giant_route = new_route
                try:
                    trial_cost = evaluate_gs(trial_gs, inst)
                except Exception:
                    continue
                if trial_cost < best_cost - 1e-9:
                    best_cost = trial_cost
                    best_gs = trial_gs.copy()
                    best_gs._cost = trial_cost
                    current_bps = trial_bps
        else:
            # 尝试删除断点：将 seg=1 和 seg=2 合并回 seg=0
            # 找到该边的两段在 giant_route 中的位置
            idx1, idx2 = None, None
            for gi, se in enumerate(best_gs.giant_route):
                if se.origin_edge_idx == ei:
                    if se.seg == 1:
                        idx1 = gi
                    elif se.seg == 2:
                        idx2 = gi
            if idx1 is None or idx2 is None:
                continue

            trial_bps = list(current_bps)
            trial_bps[ei] = None
            u, v = inst.edges[ei]
            ux, uy = inst.node_coord(u)
            vx, vy = inst.node_coord(v)
            se_whole = SubEdge(origin_edge_idx=ei, seg=0,
                               ax=ux, ay=uy, bx=vx, by=vy)
            # 合并：保留 idx1 位置放整边，删除 idx2
            # 两段可能不相邻，保留先出现的位置
            keep_idx = min(idx1, idx2)
            del_idx = max(idx1, idx2)
            new_route = list(best_gs.giant_route)
            new_route[keep_idx] = se_whole
            new_route.pop(del_idx)
            trial_gs = GiantRouteSolution(best_gs.num_edges)
            trial_gs.breakpoints = trial_bps
            trial_gs.giant_route = new_route
            try:
                trial_cost = evaluate_gs(trial_gs, inst)
            except Exception:
                trial_cost = float('inf')

            if trial_cost < best_cost - 1e-9:
                best_cost = trial_cost
                best_gs = trial_gs.copy()
                best_gs._cost = trial_cost
                current_bps = trial_bps
            elif not fast_mode:
                # 不删除，尝试移动断点位置
                for lam in [0.2, 0.35, 0.5, 0.65, 0.8,
                            random.uniform(0.1, 0.45),
                            random.uniform(0.55, 0.9)]:
                    trial_bps2 = list(current_bps)
                    trial_bps2[ei] = lam
                    trial_gs2 = rebuild_sub_edges_in_giant_route(
                        best_gs, inst, trial_bps2)
                    try:
                        trial_cost2 = evaluate_gs(trial_gs2, inst)
                    except Exception:
                        continue
                    if trial_cost2 < best_cost - 1e-9:
                        best_cost = trial_cost2
                        best_gs = trial_gs2.copy()
                        best_gs._cost = trial_cost2
                        current_bps = trial_bps2
                        break

    return best_gs


# ============================================================
# 12. PSO 优化断点位置
# ============================================================

class PSOBreakpointOptimizer:
    """
    粒子群优化器：在固定 Giant Route 顺序和断点存在性的前提下，
    只优化各有断点的边的断点位置 λ。

    决策变量：对每条有断点的边，λ_i ∈ [0.05, 0.95]
    评估函数：重建子边坐标 → 调用 Split DP → 计算总费用
    """
    def __init__(self, num_particles: int = 20, max_iter: int = 30,
                 w: float = 0.7, c1: float = 1.5, c2: float = 1.5):
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def optimize(
        self, gs: GiantRouteSolution, inst: Instance
    ) -> GiantRouteSolution:
        """
        对当前 GiantRouteSolution 中有断点的边优化断点位置。
        返回优化后的 GiantRouteSolution（若改进则更新，否则返回原解）。
        """
        # 收集有断点的边索引
        bp_edges = [ei for ei, lam in enumerate(gs.breakpoints) if lam is not None]
        if not bp_edges:
            return gs

        dim = len(bp_edges)
        current_cost = evaluate_gs(gs, inst)

        # 初始化粒子
        init_pos = np.array([gs.breakpoints[ei] for ei in bp_edges])
        particles = np.tile(init_pos, (self.num_particles, 1))
        noise = np.random.uniform(-0.2, 0.2, (self.num_particles, dim))
        particles = np.clip(particles + noise, 0.05, 0.95)
        particles[0] = init_pos.copy()
        # 部分粒子随机初始化
        n_rand = max(2, self.num_particles // 5)
        particles[-n_rand:] = np.random.uniform(0.05, 0.95, (n_rand, dim))

        velocities = np.random.uniform(-0.15, 0.15, (self.num_particles, dim))

        def evaluate(pos: np.ndarray) -> float:
            new_bps = list(gs.breakpoints)
            for i, ei in enumerate(bp_edges):
                new_bps[ei] = float(np.clip(pos[i], 0.05, 0.95))
            trial_gs = rebuild_sub_edges_in_giant_route(gs, inst, new_bps)
            try:
                return evaluate_gs(trial_gs, inst)
            except Exception:
                return float('inf')

        pbest_pos = particles.copy()
        pbest_cost = np.array([evaluate(p) for p in particles])
        gbest_idx = int(np.argmin(pbest_cost))
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_cost = float(pbest_cost[gbest_idx])

        w = self.w
        for _ in range(self.max_iter):
            for i in range(self.num_particles):
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)
                velocities[i] = (w * velocities[i]
                                 + self.c1 * r1 * (pbest_pos[i] - particles[i])
                                 + self.c2 * r2 * (gbest_pos - particles[i]))
                velocities[i] = np.clip(velocities[i], -0.3, 0.3)
                particles[i] = np.clip(particles[i] + velocities[i], 0.05, 0.95)
                cost = evaluate(particles[i])
                if cost < pbest_cost[i]:
                    pbest_cost[i] = cost
                    pbest_pos[i] = particles[i].copy()
                    if cost < gbest_cost:
                        gbest_cost = cost
                        gbest_pos = particles[i].copy()
            w = max(0.35, w * 0.97)

        if gbest_cost < current_cost - 1e-9:
            new_bps = list(gs.breakpoints)
            for i, ei in enumerate(bp_edges):
                new_bps[ei] = float(np.clip(gbest_pos[i], 0.05, 0.95))
            result_gs = rebuild_sub_edges_in_giant_route(gs, inst, new_bps)
            result_gs._cost = gbest_cost
            return result_gs
        return gs


# ============================================================
# 13. ALNS 主框架
# ============================================================

class GiantRouteALNSSolver:
    """
    Giant Route + Split 的 ALNS 求解器。

    搜索空间：Giant Route 排列（子边的访问顺序）+ 断点配置
    解码器：Split DP（将排列最优切割为多段路径）
    优化器：
      - ALNS：操作 Giant Route 排列（删除/插入/断点搜索）
      - 局部搜索：2-opt + or-opt（Giant Route 上的排列优化）
      - PSO：优化断点位置（固定排列和断点存在性）
    """
    def __init__(self, inst: Instance,
                 max_iter: int = 500,
                 segment_size: int = 50,
                 removal_min: float = 0.1,
                 removal_max: float = 0.4,
                 sigma1: float = 33.0,
                 sigma2: float = 9.0,
                 sigma3: float = 3.0,
                 decay: float = 0.8,
                 sa_temp_init: float = None,
                 sa_cooling: float = 0.998,
                 pso_freq: int = 50,
                 pso_particles: int = 20,
                 pso_iter: int = 30,
                 ls_freq: int = 25,
                 ):
        self.inst = inst
        self.max_iter = max_iter
        self.segment_size = segment_size
        self.removal_min = removal_min
        self.removal_max = removal_max
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma3 = sigma3
        self.decay = decay
        self.sa_cooling = sa_cooling
        self.pso_freq = pso_freq
        self.ls_freq = ls_freq
        self.sa_temp_init = sa_temp_init
        self.sa_temp = sa_temp_init

        self.pso = PSOBreakpointOptimizer(pso_particles, pso_iter)

        # 破坏算子
        self.destroy_ops = [
            destroy_random_removal,
            destroy_worst_removal,
            destroy_segment_removal,
        ]
        self.destroy_names = ["random", "worst", "segment"]
        self.destroy_weights = [1.0] * len(self.destroy_ops)
        self.destroy_scores = [0.0] * len(self.destroy_ops)
        self.destroy_counts = [0] * len(self.destroy_ops)

        # 修复算子
        self.repair_ops = [
            repair_greedy_insert,
            repair_random_insert,
            repair_regret_insert,
        ]
        self.repair_names = ["greedy", "random", "regret"]
        self.repair_weights = [1.0] * len(self.repair_ops)
        self.repair_scores = [0.0] * len(self.repair_ops)
        self.repair_counts = [0] * len(self.repair_ops)

        self.cost_history = []
        self.best_cost_history = []

    def _roulette_select(self, weights: List[float]) -> int:
        total = sum(weights)
        if total <= 0:
            return random.randint(0, len(weights) - 1)
        r = random.uniform(0, total)
        cumsum = 0.0
        for i, w in enumerate(weights):
            cumsum += w
            if r <= cumsum:
                return i
        return len(weights) - 1

    def _update_weights(self, di: int, ri: int, score: float):
        self.destroy_scores[di] += score
        self.destroy_counts[di] += 1
        self.repair_scores[ri] += score
        self.repair_counts[ri] += 1

    def _normalize_weights(self):
        for i in range(len(self.destroy_weights)):
            if self.destroy_counts[i] > 0:
                new_score = self.destroy_scores[i] / self.destroy_counts[i]
                self.destroy_weights[i] = (
                    (1 - self.decay) * self.destroy_weights[i] + self.decay * new_score
                )
            self.destroy_weights[i] = max(0.1, self.destroy_weights[i])
            self.destroy_scores[i] = 0.0
            self.destroy_counts[i] = 0
        for i in range(len(self.repair_weights)):
            if self.repair_counts[i] > 0:
                new_score = self.repair_scores[i] / self.repair_counts[i]
                self.repair_weights[i] = (
                    (1 - self.decay) * self.repair_weights[i] + self.decay * new_score
                )
            self.repair_weights[i] = max(0.1, self.repair_weights[i])
            self.repair_scores[i] = 0.0
            self.repair_counts[i] = 0

    def _sa_accept(self, current_cost: float, new_cost: float) -> bool:
        if new_cost <= current_cost:
            return True
        if self.sa_temp is None or self.sa_temp <= 0:
            return False
        delta = new_cost - current_cost
        return random.random() < math.exp(-delta / self.sa_temp)

    def solve(self, initial_gs: Optional[GiantRouteSolution] = None,
              verbose: bool = True) -> Tuple[Solution, List[float]]:
        """
        主求解函数，返回 (best_Solution, best_cost_history)。
        """
        inst = self.inst

        # ---- 初始解 ----
        if initial_gs is None:
            if verbose:
                print("多启动策略生成初始 Giant Route 解...")
            current_gs = multi_start_initial_gs(inst, n_starts=6)
        else:
            current_gs = initial_gs.copy()

        # 初始断点邻域搜索
        if verbose:
            print("对初始解执行断点邻域搜索...")
        current_gs = try_add_remove_breakpoints(current_gs, inst, fast_mode=False)

        current_cost = evaluate_gs(current_gs, inst)
        best_gs = current_gs.copy()
        best_cost = current_cost

        if self.sa_temp is None:
            self.sa_temp = current_cost * 0.05
            if verbose:
                print(f"SA 初始温度自动设置为: {self.sa_temp:.4f}")

        if verbose:
            print(f"初始解费用: {current_cost:.4f}")

        self.cost_history = [current_cost]
        self.best_cost_history = [best_cost]
        start_time = time.time()

        for iteration in range(self.max_iter):
            removal_fraction = random.uniform(self.removal_min, self.removal_max)

            # 破坏
            di_op = self._roulette_select(self.destroy_weights)
            ri_op = self._roulette_select(self.repair_weights)
            destroyed_gs, removed = self.destroy_ops[di_op](
                current_gs, inst, removal_fraction)

            # 修复
            new_gs = self.repair_ops[ri_op](destroyed_gs, inst, removed)

            # 局部搜索（or-opt，每 ls_freq 次）
            if (iteration + 1) % self.ls_freq == 0:
                new_gs = local_search_or_opt(new_gs, inst,
                                             segment_sizes=[1, 2],
                                             max_no_improve=3)

            # PSO 优化断点位置（每 pso_freq 次）
            if (iteration + 1) % self.pso_freq == 0:
                new_gs = self.pso.optimize(new_gs, inst)
                new_gs = try_add_remove_breakpoints(new_gs, inst, fast_mode=False)

            try:
                new_cost = evaluate_gs(new_gs, inst)
            except Exception as e:
                if verbose:
                    print(f"  Iter {iteration+1}: 评估失败 ({e})，跳过")
                self.cost_history.append(current_cost)
                self.best_cost_history.append(best_cost)
                continue

            # 接受准则
            score = 0.0
            if new_cost < best_cost - 1e-9:
                best_gs = new_gs.copy()
                best_cost = new_cost
                current_gs = new_gs.copy()
                current_cost = new_cost
                score = self.sigma1
            elif new_cost < current_cost - 1e-9:
                current_gs = new_gs.copy()
                current_cost = new_cost
                score = self.sigma2
            elif self._sa_accept(current_cost, new_cost):
                current_gs = new_gs.copy()
                current_cost = new_cost
                score = self.sigma3

            self._update_weights(di_op, ri_op, score)
            self.sa_temp *= self.sa_cooling

            if (iteration + 1) % self.segment_size == 0:
                self._normalize_weights()
                if verbose:
                    elapsed = time.time() - start_time
                    print(f"  Iter {iteration+1:4d}/{self.max_iter} | "
                          f"current: {current_cost:.4f} | best: {best_cost:.4f} | "
                          f"T: {self.sa_temp:.4f} | time: {elapsed:.1f}s")

            self.cost_history.append(current_cost)
            self.best_cost_history.append(best_cost)

        # ---- 最终精化 ----
        if verbose:
            print("最终精化：2-opt + or-opt + 断点邻域搜索 + PSO...")

        # 第一轮：2-opt + or-opt
        final_gs = local_search_2opt(best_gs, inst, max_no_improve=2)
        final_gs = local_search_or_opt(final_gs, inst,
                                       segment_sizes=[1, 2, 3], max_no_improve=3)
        try:
            final_cost = evaluate_gs(final_gs, inst)
            if final_cost < best_cost - 1e-9:
                best_gs = final_gs.copy()
                best_cost = final_cost
        except Exception:
            pass

        # 第二轮：断点邻域搜索
        final_gs2 = try_add_remove_breakpoints(best_gs, inst, fast_mode=False)
        try:
            final_cost2 = evaluate_gs(final_gs2, inst)
            if final_cost2 < best_cost - 1e-9:
                best_gs = final_gs2.copy()
                best_cost = final_cost2
        except Exception:
            pass

        # 第三轮：PSO 精化
        final_gs3 = self.pso.optimize(best_gs, inst)
        try:
            final_cost3 = evaluate_gs(final_gs3, inst)
            if final_cost3 < best_cost - 1e-9:
                best_gs = final_gs3.copy()
                best_cost = final_cost3
        except Exception:
            pass

        # 第四轮：再一次断点邻域搜索
        final_gs4 = try_add_remove_breakpoints(best_gs, inst, fast_mode=False)
        try:
            final_cost4 = evaluate_gs(final_gs4, inst)
            if final_cost4 < best_cost - 1e-9:
                best_gs = final_gs4.copy()
                best_cost = final_cost4
        except Exception:
            pass

        if verbose:
            print(f"\n求解完成！最优费用: {best_cost:.4f}")

        best_sol = gs_to_solution(best_gs, inst)
        best_sol.breakpoints = list(best_gs.breakpoints)
        return best_sol, self.best_cost_history

    def solve_parallel_worker(
        self,
        pool: 'SharedPool',
        thread_id: int,
        push_freq: int = 50,
        stagnation_limit: int = 100,
        initial_gs: Optional[GiantRouteSolution] = None,
        verbose: bool = False,
    ) -> Tuple[Solution, List[float]]:
        """
        并行工作线程的求解主体。
        与公共池交互：
          - 每 push_freq 次迭代推送本线程最优解（以 Giant Route 形式存储）
          - 连续 stagnation_limit 次未改进时从公共池拉取多样解重启
        """
        inst = self.inst

        if initial_gs is None:
            current_gs = multi_start_initial_gs(inst, n_starts=6)
        else:
            current_gs = initial_gs.copy()

        current_gs = try_add_remove_breakpoints(current_gs, inst, fast_mode=False)
        current_cost = evaluate_gs(current_gs, inst)
        best_gs = current_gs.copy()
        best_cost = current_cost

        if self.sa_temp is None:
            self.sa_temp = current_cost * 0.05

        self.cost_history = [current_cost]
        self.best_cost_history = [best_cost]
        stagnation_count = 0

        for iteration in range(self.max_iter):
            removal_fraction = random.uniform(self.removal_min, self.removal_max)

            di_op = self._roulette_select(self.destroy_weights)
            ri_op = self._roulette_select(self.repair_weights)
            destroyed_gs, removed = self.destroy_ops[di_op](
                current_gs, inst, removal_fraction)
            new_gs = self.repair_ops[ri_op](destroyed_gs, inst, removed)

            if (iteration + 1) % self.ls_freq == 0:
                new_gs = local_search_or_opt(new_gs, inst,
                                             segment_sizes=[1, 2],
                                             max_no_improve=3)

            if (iteration + 1) % self.pso_freq == 0:
                new_gs = self.pso.optimize(new_gs, inst)
                new_gs = try_add_remove_breakpoints(new_gs, inst, fast_mode=False)

            try:
                new_cost = evaluate_gs(new_gs, inst)
            except Exception:
                self.cost_history.append(current_cost)
                self.best_cost_history.append(best_cost)
                stagnation_count += 1
                continue

            score = 0.0
            if new_cost < best_cost - 1e-9:
                best_gs = new_gs.copy()
                best_cost = new_cost
                current_gs = new_gs.copy()
                current_cost = new_cost
                score = self.sigma1
                stagnation_count = 0
            elif new_cost < current_cost - 1e-9:
                current_gs = new_gs.copy()
                current_cost = new_cost
                score = self.sigma2
                stagnation_count += 1
            elif self._sa_accept(current_cost, new_cost):
                current_gs = new_gs.copy()
                current_cost = new_cost
                score = self.sigma3
                stagnation_count += 1
            else:
                stagnation_count += 1

            self._update_weights(di_op, ri_op, score)
            self.sa_temp *= self.sa_cooling

            if (iteration + 1) % self.segment_size == 0:
                self._normalize_weights()

            self.cost_history.append(current_cost)
            self.best_cost_history.append(best_cost)

            # 推送
            if (iteration + 1) % push_freq == 0:
                best_sol_tmp = gs_to_solution(best_gs, inst)
                best_sol_tmp.breakpoints = list(best_gs.breakpoints)
                pool.push(best_sol_tmp, best_cost, thread_id, best_gs)
                if verbose:
                    print(f"  [T{thread_id}] Iter {iteration+1:4d} push cost={best_cost:.4f}")

            # 拉取（停滞重启）
            if stagnation_count >= stagnation_limit:
                pull_gs, pull_cost = pool.pull_diverse_gs(best_gs)
                if pull_gs is not None:
                    restarted_gs = try_add_remove_breakpoints(
                        pull_gs.copy(), inst, fast_mode=False)
                    try:
                        restarted_cost = evaluate_gs(restarted_gs, inst)
                        current_gs = restarted_gs.copy()
                        current_cost = restarted_cost
                        if restarted_cost < best_cost - 1e-9:
                            best_gs = restarted_gs.copy()
                            best_cost = restarted_cost
                    except Exception:
                        pass
                    if verbose:
                        print(f"  [T{thread_id}] Iter {iteration+1:4d} "
                              f"pull_diverse → restart={current_cost:.4f}")
                stagnation_count = 0

        # 最终精化
        final_gs = local_search_2opt(best_gs, inst, max_no_improve=2)
        final_gs = local_search_or_opt(final_gs, inst,
                                       segment_sizes=[1, 2, 3], max_no_improve=3)
        try:
            fc = evaluate_gs(final_gs, inst)
            if fc < best_cost - 1e-9:
                best_gs = final_gs.copy()
                best_cost = fc
        except Exception:
            pass

        final_gs2 = try_add_remove_breakpoints(best_gs, inst, fast_mode=False)
        try:
            fc2 = evaluate_gs(final_gs2, inst)
            if fc2 < best_cost - 1e-9:
                best_gs = final_gs2.copy()
                best_cost = fc2
        except Exception:
            pass

        final_gs3 = self.pso.optimize(best_gs, inst)
        try:
            fc3 = evaluate_gs(final_gs3, inst)
            if fc3 < best_cost - 1e-9:
                best_gs = final_gs3.copy()
                best_cost = fc3
        except Exception:
            pass

        best_sol = gs_to_solution(best_gs, inst)
        best_sol.breakpoints = list(best_gs.breakpoints)

        pool.push(best_sol, best_cost, thread_id, best_gs)
        if verbose:
            print(f"  [T{thread_id}] 完成，最优费用: {best_cost:.4f}")

        return best_sol, self.best_cost_history


# ============================================================
# 14. 公共解池 + 并行求解
# ============================================================

def _gs_signature(gs: GiantRouteSolution) -> frozenset:
    """Giant Route 的断点拓扑签名"""
    return frozenset(ei for ei, lam in enumerate(gs.breakpoints) if lam is not None)


def _jaccard(sig_a: frozenset, sig_b: frozenset) -> float:
    if not sig_a and not sig_b:
        return 1.0
    union = sig_a | sig_b
    inter = sig_a & sig_b
    return len(inter) / len(union)


class SharedPool:
    """
    线程安全的公共解池（存储 Solution + GiantRouteSolution 对）。
    """
    def __init__(self, capacity: int = 5, diversity_threshold: float = 0.8):
        self._lock = threading.Lock()
        # 存储 (cost, Solution, GiantRouteSolution)
        self._pool: List[Tuple[float, Solution, GiantRouteSolution]] = []
        self.capacity = capacity
        self.diversity_threshold = diversity_threshold
        self.total_pushes = 0
        self.total_pulls = 0

    def push(self, sol: Solution, cost: float, thread_id: int,
             gs: GiantRouteSolution = None):
        with self._lock:
            new_sig = _gs_signature(gs) if gs is not None else frozenset()
            sol_copy = sol.copy()
            gs_copy = gs.copy() if gs is not None else None

            best_sim = 0.0
            best_sim_idx = -1
            for idx, (c, s, g) in enumerate(self._pool):
                sig = _gs_signature(g) if g is not None else frozenset()
                sim = _jaccard(new_sig, sig)
                if sim > best_sim:
                    best_sim = sim
                    best_sim_idx = idx

            if best_sim >= self.diversity_threshold and best_sim_idx >= 0:
                if cost < self._pool[best_sim_idx][0]:
                    self._pool[best_sim_idx] = (cost, sol_copy, gs_copy)
                    self._pool.sort(key=lambda x: x[0])
            else:
                self._pool.append((cost, sol_copy, gs_copy))
                self._pool.sort(key=lambda x: x[0])
                if len(self._pool) > self.capacity:
                    self._pool = self._pool[:self.capacity]

            self.total_pushes += 1

    def global_best(self) -> Tuple[Optional[Solution], float]:
        with self._lock:
            if self._pool:
                cost, sol, _ = self._pool[0]
                return sol.copy(), cost
            return None, float('inf')

    def pull_diverse_gs(
        self, current_gs: GiantRouteSolution
    ) -> Tuple[Optional[GiantRouteSolution], float]:
        """拉取与当前解结构差异最大的 GiantRouteSolution"""
        with self._lock:
            self.total_pulls += 1
            if not self._pool:
                return None, float('inf')
            current_sig = _gs_signature(current_gs)
            best_div = -1.0
            best_idx = 0
            for idx, (c, s, g) in enumerate(self._pool):
                sig = _gs_signature(g) if g is not None else frozenset()
                div = 1.0 - _jaccard(current_sig, sig)
                if div > best_div:
                    best_div = div
                    best_idx = idx
            if best_div < 0.2:
                best_idx = 0
            cost, sol, gs = self._pool[best_idx]
            return (gs.copy() if gs is not None else None), cost

    def stats(self) -> str:
        with self._lock:
            n = len(self._pool)
            best = self._pool[0][0] if self._pool else float('inf')
        return (f"SharedPool: size={n}/{self.capacity}, "
                f"best={best:.4f}, "
                f"pushes={self.total_pushes}, pulls={self.total_pulls}")


def parallel_solve(
    inst: Instance,
    num_threads: int = 4,
    max_iter: int = 500,
    push_freq: int = 50,
    stagnation_limit: int = 100,
    pool_capacity: int = 5,
    pso_freq: int = 50,
    pso_particles: int = 20,
    pso_iter: int = 30,
    ls_freq: int = 25,
    verbose: bool = True,
) -> Tuple[Solution, List[float]]:
    """异步并行 Giant Route ALNS+PSO 求解"""
    pool = SharedPool(capacity=pool_capacity)

    if verbose:
        print(f"\n{'='*60}")
        print(f"异步并行 Giant Route ALNS+PSO 启动")
        print(f"  线程数: {num_threads}  迭代数/线程: {max_iter}")
        print(f"{'='*60}")

    threads = []
    thread_results: List[Optional[Tuple[Solution, List[float]]]] = [None] * num_threads

    def worker(tid: int):
        solver = GiantRouteALNSSolver(
            inst,
            max_iter=max_iter,
            pso_freq=pso_freq,
            pso_particles=pso_particles,
            pso_iter=pso_iter,
            ls_freq=ls_freq,
        )
        sol, hist = solver.solve_parallel_worker(
            pool=pool, thread_id=tid,
            push_freq=push_freq,
            stagnation_limit=stagnation_limit,
            verbose=verbose,
        )
        thread_results[tid] = (sol, hist)
        if verbose:
            cost = compute_cost(sol, inst)
            print(f"  [线程 {tid+1}/{num_threads}] 完成，本地最优: {cost:.4f}")

    for tid in range(num_threads):
        t = threading.Thread(target=worker, args=(tid,), daemon=True)
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if verbose:
        print(f"\n{pool.stats()}")

    best_sol, best_cost = pool.global_best()

    if best_sol is None:
        for res in thread_results:
            if res is not None:
                sol, hist = res
                c = compute_cost(sol, inst)
                if c < float('inf'):
                    best_sol = sol
                    break

    all_hists = [res[1] for res in thread_results if res is not None]
    if all_hists:
        max_len = max(len(h) for h in all_hists)
        padded = [h + [h[-1]] * (max_len - len(h)) for h in all_hists]
        merged_hist = [min(padded[t][i] for t in range(len(padded)))
                       for i in range(max_len)]
    else:
        merged_hist = []

    if verbose:
        print(f"并行求解完成！全局最优费用: {best_cost:.4f}")

    return best_sol, merged_hist


# ============================================================
# 15. 结果输出与可视化
# ============================================================

def print_solution_detail(sol: Solution, inst: Instance, solve_time: float = None):
    """打印解的详细信息"""
    print("\n" + "="*60)
    print("解的详细信息（Giant Route + Split）")
    print("="*60)
    total_cost = compute_cost(sol, inst)
    depot_x, depot_y = inst.node_coord(inst.depot_idx)

    print(f"\n总费用: {total_cost:.4f} 元")
    if solve_time is not None:
        print(f"求解用时: {solve_time:.2f} 秒")
    print(f"电池容量: {inst.battery:.1f} m")

    print("\n--- 断点配置 ---")
    any_bp = False
    for ei, bp in enumerate(sol.breakpoints):
        u, v = inst.edges[ei]
        if bp is not None:
            edge_len = inst.edge_length(ei)
            bp_coord = inst.point_on_edge(ei, bp)
            print(f"  边({u},{v}) 长度={edge_len:.2f}m: "
                  f"断点位置 λ={bp:.4f}, "
                  f"坐标=({bp_coord[0]:.2f}, {bp_coord[1]:.2f})")
            any_bp = True
    if not any_bp:
        print("  （无断点）")

    print("\n--- 各无人机路径 ---")
    for di, route in enumerate(sol.routes):
        if not route.sub_edges:
            print(f"\n  无人机 {di+1}: 未使用")
            continue
        energy = compute_route_distance(route, inst)
        inspect_d, transfer_d = compute_route_raw_distance(route, inst)
        cost = compute_route_cost(route, inst)
        feasible = energy <= inst.battery
        print(f"\n  无人机 {di+1}:")
        print(f"    总飞行距离: {inspect_d+transfer_d:.2f}m "
              f"(巡检: {inspect_d:.2f}m + 转移: {transfer_d:.2f}m)")
        print(f"    能量消耗: {energy:.4f}  "
              f"(电池容量: {inst.battery:.4f})  "
              f"{'[可行]' if feasible else '[超出电池!]'}")
        print(f"    路径费用: {cost:.4f} 元")
        print(f"    路径（完整飞行顺序）:")
        cur_label = "基站"
        for i, (se, direction) in enumerate(zip(route.sub_edges, route.directions)):
            u_orig, v_orig = inst.edges[se.origin_edge_idx]
            bp = sol.breakpoints[se.origin_edge_idx]
            ux, uy = inst.node_coord(u_orig)
            vx, vy = inst.node_coord(v_orig)
            if bp is not None:
                bp_label = "bp"
            else:
                bp_label = None
            if se.seg == 0:
                enter_label = str(u_orig) if direction else str(v_orig)
                leave_label = str(v_orig) if direction else str(u_orig)
            elif se.seg == 1:
                enter_label = str(u_orig) if direction else bp_label
                leave_label = bp_label if direction else str(u_orig)
            else:
                enter_label = bp_label if direction else str(v_orig)
                leave_label = str(v_orig) if direction else bp_label
            svc_str = f"边({u_orig},{v_orig})({enter_label}→{leave_label})"
            print(f"      {cur_label} 转移到 {svc_str}")
            cur_label = svc_str
        print(f"      {cur_label} 转移到 基站")
        print(f"    费用明细:")
        print(f"      调用成本: {inst.call_cost:.2f} 元")
        print(f"      巡检成本: {inspect_d:.2f}m × {inst.inspect_coef} × "
              f"{inst.energy_cost} = "
              f"{inspect_d * inst.inspect_coef * inst.energy_cost:.4f} 元")
        print(f"      转移成本: {transfer_d:.2f}m × {inst.transfer_coef} × "
              f"{inst.energy_cost} = "
              f"{transfer_d * inst.transfer_coef * inst.energy_cost:.4f} 元")
    print("\n" + "="*60)


def plot_convergence(cost_history: List[float], output_path: str):
    """绘制收敛曲线"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cost_history, color='#2196F3', linewidth=1.2, alpha=0.8,
            label='Best Cost')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Cost', fontsize=12)
    ax.set_title('Giant Route ALNS+PSO Convergence Curve', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"收敛曲线已保存: {output_path}")


def plot_solution(sol: Solution, inst: Instance, output_path: str,
                  title: str = "UAV Arc Routing Solution (Giant Route)"):
    """可视化解"""
    fig, ax = plt.subplots(figsize=(12, 10))
    depot_x, depot_y = inst.node_coord(inst.depot_idx)

    for ei, (u, v) in enumerate(inst.edges):
        ux, uy = inst.node_coord(u)
        vx, vy = inst.node_coord(v)
        ax.plot([ux, vx], [uy, vy], color='#CCCCCC', linewidth=2.5,
                zorder=1, solid_capstyle='round')

    drone_colors = ['#E74C3C', '#2ECC71', '#3498DB', '#F39C12',
                    '#9B59B6', '#1ABC9C', '#E67E22', '#2980B9']
    legend_handles = []

    for di, route in enumerate(sol.routes):
        if not route.sub_edges:
            continue
        color = drone_colors[di % len(drone_colors)]
        path_points = [(depot_x, depot_y)]
        path_types = []
        for i, (se, direction) in enumerate(zip(route.sub_edges, route.directions)):
            sx, sy = (se.ax, se.ay) if direction else (se.bx, se.by)
            ex, ey = (se.bx, se.by) if direction else (se.ax, se.ay)
            path_types.append('transfer')
            path_points.append((sx, sy))
            path_types.append('inspect')
            path_points.append((ex, ey))
        path_types.append('transfer')
        path_points.append((depot_x, depot_y))

        for i in range(len(path_types)):
            x0, y0 = path_points[i]
            x1, y1 = path_points[i + 1]
            if path_types[i] == 'inspect':
                ax.plot([x0, x1], [y0, y1], color=color, linewidth=3.0,
                        zorder=3, solid_capstyle='round')
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                dx, dy = x1 - x0, y1 - y0
                norm = math.hypot(dx, dy)
                if norm > 1e-9:
                    ax.annotate('', xy=(mx + dx/norm*5, my + dy/norm*5),
                                xytext=(mx - dx/norm*5, my - dy/norm*5),
                                arrowprops=dict(arrowstyle='->', color=color,
                                                lw=1.5), zorder=4)
            else:
                ax.plot([x0, x1], [y0, y1], color=color, linewidth=1.5,
                        linestyle='--', alpha=0.6, zorder=2)

        insp_d, trans_d = compute_route_raw_distance(route, inst)
        cost = compute_route_cost(route, inst)
        patch = mpatches.Patch(
            color=color,
            label=f'UAV {di+1} (dist={insp_d+trans_d:.0f}m, cost={cost:.2f})')
        legend_handles.append(patch)

    for ei, bp in enumerate(sol.breakpoints):
        if bp is not None:
            bpx, bpy = inst.point_on_edge(ei, bp)
            ax.scatter(bpx, bpy, marker='*', s=200, color='#FF6B6B',
                       zorder=6, edgecolors='#C0392B', linewidths=0.8)

    for ni in range(1, inst.total_nodes):
        nx_, ny_ = inst.node_coord(ni)
        ax.scatter(nx_, ny_, marker='o', s=80, color='#5B9BD5',
                   zorder=5, edgecolors='white', linewidths=0.8)
        ax.annotate(str(ni), (nx_, ny_), textcoords='offset points',
                    xytext=(4, 4), fontsize=7, color='#2C3E50')

    ax.scatter(depot_x, depot_y, marker='s', s=300, color='#E74C3C',
               zorder=7, edgecolors='white', linewidths=1.5)
    ax.annotate('DEPOT', (depot_x, depot_y), textcoords='offset points',
                xytext=(5, 5), fontsize=9, color='#E74C3C', fontweight='bold')

    if any(bp is not None for bp in sol.breakpoints):
        bp_patch = mpatches.Patch(color='#FF6B6B', label='Breakpoint')
        legend_handles.append(bp_patch)

    ax.legend(handles=legend_handles, loc='best', fontsize=8, framealpha=0.8)
    ax.set_aspect('equal', adjustable='datalim')
    ax.margins(0.15)
    ax.set_title(title, fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.2)
    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"路径可视化图已保存: {output_path}")


def save_solution_txt(sol: Solution, inst: Instance, output_path: str,
                      cost_history: List[float], solve_time: float = None):
    """保存解到txt文件"""
    total_cost = compute_cost(sol, inst)
    depot_x, depot_y = inst.node_coord(inst.depot_idx)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("Giant Route ALNS+PSO 求解结果\n")
        f.write("="*60 + "\n")
        f.write(f"\n总费用: {total_cost:.6f} 元\n")
        if solve_time is not None:
            f.write(f"求解用时: {solve_time:.2f} 秒\n")
        if cost_history:
            f.write(f"初始解费用: {cost_history[0]:.6f} 元\n")
            if cost_history[0] > 0:
                f.write(f"改进率: {(cost_history[0] - total_cost) / cost_history[0] * 100:.2f}%\n")

        f.write("\n--- 断点配置 ---\n")
        for ei, bp in enumerate(sol.breakpoints):
            u, v = inst.edges[ei]
            edge_len = inst.edge_length(ei)
            if bp is not None:
                bp_coord = inst.point_on_edge(ei, bp)
                f.write(f"  边({u},{v}) 长度={edge_len:.2f}m: "
                        f"断点位置 lambda={bp:.6f}, "
                        f"坐标=({bp_coord[0]:.4f}, {bp_coord[1]:.4f})\n")
            else:
                f.write(f"  边({u},{v}) 长度={edge_len:.2f}m: 无断点\n")

        f.write("\n--- 各无人机路径 ---\n")
        for di, route in enumerate(sol.routes):
            if not route.sub_edges:
                f.write(f"\n无人机 {di+1}: 未使用\n")
                continue
            energy = compute_route_distance(route, inst)
            inspect_d, transfer_d = compute_route_raw_distance(route, inst)
            cost = compute_route_cost(route, inst)
            feasible = energy <= inst.battery
            f.write(f"\n无人机 {di+1}:\n")
            f.write(f"  总飞行距离: {inspect_d+transfer_d:.4f}m "
                    f"(巡检: {inspect_d:.4f}m, 转移: {transfer_d:.4f}m)\n")
            f.write(f"  能量消耗: {energy:.6f}  "
                    f"(电池容量: {inst.battery:.4f}) "
                    f"{'[可行]' if feasible else '[超出电池!]'}\n")
            f.write(f"  路径费用: {cost:.6f} 元\n")
            f.write(f"  路径（完整飞行顺序）:\n")
            cur_label = "基站"
            for i, (se, direction) in enumerate(zip(route.sub_edges, route.directions)):
                u_orig, v_orig = inst.edges[se.origin_edge_idx]
                bp = sol.breakpoints[se.origin_edge_idx]
                if bp is not None:
                    bp_label = "bp"
                else:
                    bp_label = None
                if se.seg == 0:
                    enter_label = str(u_orig) if direction else str(v_orig)
                    leave_label = str(v_orig) if direction else str(u_orig)
                elif se.seg == 1:
                    enter_label = str(u_orig) if direction else bp_label
                    leave_label = bp_label if direction else str(u_orig)
                else:
                    enter_label = bp_label if direction else str(v_orig)
                    leave_label = str(v_orig) if direction else bp_label
                svc_str = f"边({u_orig},{v_orig})({enter_label}→{leave_label})"
                f.write(f"    {cur_label} 转移到 {svc_str}\n")
                cur_label = svc_str
            f.write(f"    {cur_label} 转移到 基站\n")
            f.write(f"  费用明细:\n")
            f.write(f"    调用成本: {inst.call_cost:.4f} 元\n")
            f.write(f"    巡检成本: {inspect_d:.4f}m × {inst.inspect_coef} × "
                    f"{inst.energy_cost} = "
                    f"{inspect_d * inst.inspect_coef * inst.energy_cost:.6f} 元\n")
            f.write(f"    转移成本: {transfer_d:.4f}m × {inst.transfer_coef} × "
                    f"{inst.energy_cost} = "
                    f"{transfer_d * inst.transfer_coef * inst.energy_cost:.6f} 元\n")

        f.write("\n--- 收敛过程（每100次迭代）---\n")
        if cost_history:
            step = max(1, len(cost_history) // 100)
            for i in range(0, len(cost_history), step):
                f.write(f"  Iter {i:5d}: {cost_history[i]:.6f}\n")
            f.write(f"  Iter {len(cost_history)-1:5d}: {cost_history[-1]:.6f}\n")

    print(f"结果文件已保存: {output_path}")


# ============================================================
# 16. 主函数入口
# ============================================================

def _mirror_output_dir(instance_path: str,
                       src_root: str = "算例",
                       dst_root: str = "结果") -> str:
    abs_path = os.path.abspath(instance_path)
    parts = abs_path.replace(os.sep, '/').split('/')
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == src_root:
            parts[i] = dst_root
            return '/'.join(parts[:-1])
    return os.path.dirname(abs_path)


def solve_instance(instance_path: str,
                   output_dir: str = None,
                   max_iter: int = 500,
                   pso_freq: int = 50,
                   pso_particles: int = 20,
                   pso_iter: int = 30,
                   ls_freq: int = 25,
                   num_threads: int = 1,
                   push_freq: int = 50,
                   stagnation_limit: int = 100,
                   verbose: bool = True) -> Tuple[Solution, float]:
    """对单个算例文件运行 Giant Route ALNS+PSO 求解"""
    inst = parse_instance(instance_path)
    basename = os.path.splitext(os.path.basename(instance_path))[0]

    if output_dir is None:
        output_dir = _mirror_output_dir(instance_path, src_root="算例", dst_root="结果")
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"算例: {basename}")
        print(f"  路网节点: {inst.num_road_nodes}, 需求边: {inst.num_edges}, "
              f"无人机: {inst.num_drones}")
        print(f"  电池容量: {inst.battery:.1f}")
        print(f"  求解模式: {'异步并行 ' + str(num_threads) + ' 线程' if num_threads > 1 else '单线程'}")
        print(f"  编码方式: Giant Route + Split DP")
        print(f"{'='*60}")

    t_start = time.time()

    if num_threads > 1:
        best_sol, cost_history = parallel_solve(
            inst,
            num_threads=num_threads,
            max_iter=max_iter,
            push_freq=push_freq,
            stagnation_limit=stagnation_limit,
            pso_freq=pso_freq,
            pso_particles=pso_particles,
            pso_iter=pso_iter,
            ls_freq=ls_freq,
            verbose=verbose,
        )
    else:
        solver = GiantRouteALNSSolver(
            inst,
            max_iter=max_iter,
            pso_freq=pso_freq,
            pso_particles=pso_particles,
            pso_iter=pso_iter,
            ls_freq=ls_freq,
        )
        best_sol, cost_history = solver.solve(verbose=verbose)

    solve_time = time.time() - t_start
    best_cost = compute_cost(best_sol, inst)

    if verbose:
        print_solution_detail(best_sol, inst, solve_time=solve_time)

    txt_path = os.path.join(output_dir, f"{basename}-GR.txt")
    save_solution_txt(best_sol, inst, txt_path, cost_history, solve_time=solve_time)

    conv_path = os.path.join(output_dir, f"{basename}-GR_convergence.png")
    plot_convergence(cost_history, conv_path)

    vis_path = os.path.join(output_dir, f"{basename}-GR_solution.png")
    plot_solution(best_sol, inst, vis_path,
                  title=f"{basename} | Cost={best_cost:.2f} (GiantRoute)")

    return best_sol, best_cost


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Giant Route + Split ALNS+PSO 无人机弧路径问题求解器"
    )
    parser.add_argument("instance", nargs="?", default=None,
                        help="算例文件路径（.txt），不指定则使用默认测试算例")
    parser.add_argument("--output_dir", "-o", default=None, help="输出目录")
    parser.add_argument("--max_iter", "-n", type=int, default=500,
                        help="ALNS 最大迭代次数（默认500）")
    parser.add_argument("--pso_freq", type=int, default=50,
                        help="每隔多少ALNS迭代运行一次PSO（默认50）")
    parser.add_argument("--pso_particles", type=int, default=20,
                        help="PSO粒子数（默认20）")
    parser.add_argument("--pso_iter", type=int, default=30,
                        help="PSO迭代次数（默认30）")
    parser.add_argument("--ls_freq", type=int, default=25,
                        help="每隔多少ALNS迭代执行一次局部搜索（默认25）")
    parser.add_argument("--num_threads", "-j", type=int, default=1,
                        help="并行线程数（默认1）")
    parser.add_argument("--push_freq", type=int, default=50,
                        help="并行模式：推送频率（默认50）")
    parser.add_argument("--stagnation_limit", type=int, default=100,
                        help="并行模式：停滞阈值（默认100）")
    parser.add_argument("--batch", action="store_true", help="批量求解")
    parser.add_argument("--batch_dir", default=None, help="批量求解目录")

    args = parser.parse_args()

    if args.batch and args.batch_dir:
        txt_files = sorted([
            os.path.join(args.batch_dir, f)
            for f in os.listdir(args.batch_dir)
            if f.endswith('.txt')
        ])
        print(f"批量求解 {len(txt_files)} 个算例，目录: {args.batch_dir}")
        results = []
        for fp in txt_files:
            try:
                out_dir = (args.output_dir if args.output_dir
                           else _mirror_output_dir(fp, "算例", "结果"))
                sol, cost = solve_instance(
                    fp, output_dir=out_dir,
                    max_iter=args.max_iter,
                    pso_freq=args.pso_freq,
                    pso_particles=args.pso_particles,
                    pso_iter=args.pso_iter,
                    ls_freq=args.ls_freq,
                    num_threads=args.num_threads,
                    push_freq=args.push_freq,
                    stagnation_limit=args.stagnation_limit,
                    verbose=False,
                )
                results.append((os.path.basename(fp), out_dir, cost, True))
                print(f"  {os.path.basename(fp)}: cost={cost:.4f}")
            except Exception as e:
                results.append((os.path.basename(fp), None, None, False))
                print(f"  {os.path.basename(fp)}: 错误 - {e}")
        print(f"\n批量完成:")
        for name, out_d, cost, ok in results:
            status = f"cost={cost:.4f}, 输出→ {out_d}" if ok else "FAILED"
            print(f"  {name}: {status}")
    else:
        if args.instance is None:
            default_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "算例", "随机算例", "0-Small"
            )
            candidates = sorted([f for f in os.listdir(default_dir)
                                  if f.endswith('.txt')])
            if candidates:
                instance_path = os.path.join(default_dir, candidates[0])
                print(f"未指定算例，使用默认: {instance_path}")
            else:
                print("错误：未找到算例文件，请指定 instance 参数")
                exit(1)
        else:
            instance_path = args.instance

        solve_instance(
            instance_path,
            output_dir=args.output_dir,
            max_iter=args.max_iter,
            pso_freq=args.pso_freq,
            pso_particles=args.pso_particles,
            pso_iter=args.pso_iter,
            ls_freq=args.ls_freq,
            num_threads=args.num_threads,
            push_freq=args.push_freq,
            stagnation_limit=args.stagnation_limit,
            verbose=True,
        )

