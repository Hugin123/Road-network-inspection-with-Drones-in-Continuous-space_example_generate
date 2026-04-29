"""
ALNS + PSO 元启发式算法
解决无人机弧路径问题（连续空间断点决策）

问题描述：
- 多架无人机从基站出发并返回基站（只飞一趟）
- 所有需求边必须被服务（每条边或其子边）
- 每条需求边可选择设置一个断点（连续位置 λ ∈ (0,1)）
  - 有断点：将边分为两段独立子边，各自可被任意无人机以任意顺序服务
  - 无断点：必须一架无人机从一端到另一端完整飞过
- 目标：最小化总费用（巡检能耗 + 转移能耗 + 调用成本）

算法框架：
- ALNS：自适应大邻域搜索，优化路径分配和子边访问顺序
- PSO：粒子群优化，优化每条边的断点连续位置
"""

import math
import os
import random
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
    # 基本信息
    num_depots: int        # 基站数量
    num_road_nodes: int    # 路网节点数（不含基站）
    total_nodes: int       # 总节点数（含基站）
    num_edges: int         # 需求边数量
    num_drones: int        # 无人机数量

    # 无人机参数
    battery: float         # 电池容量（m，即最大飞行距离）
    speed: float           # 速度 m/s
    energy_cost: float     # 能源成本系数 元/kwh（这里当作 元/m 使用）
    call_cost: float       # 一次调用成本 元
    inspect_coef: float    # 巡检能耗系数 kwh/m（或元/m，与energy_cost配合）
    transfer_coef: float   # 转移能耗系数 kwh/m
    big_m: float           # 大常数

    # 坐标（0-indexed，节点0=基站）
    x: np.ndarray          # x坐标列表，长度=total_nodes，最后一个是基站
    y: np.ndarray          # y坐标列表

    # 需求边列表（0-indexed节点编号），不含 (0,0)
    edges: List[Tuple[int, int]] = field(default_factory=list)

    # 派生属性（初始化后计算）
    depot_idx: int = 0     # 基站在坐标列表中的索引（最后一个节点）

    def edge_length(self, edge_idx: int) -> float:
        """计算原始需求边长度"""
        u, v = self.edges[edge_idx]
        ux, uy = self.node_coord(u)
        vx, vy = self.node_coord(v)
        return math.hypot(ux - vx, uy - vy)

    def point_on_edge(self, edge_idx: int, lam: float) -> Tuple[float, float]:
        """
        计算边上位置 lam ∈ [0,1] 对应的坐标
        lam=0 对应端点u, lam=1 对应端点v
        """
        u, v = self.edges[edge_idx]
        ux, uy = self.node_coord(u)
        vx, vy = self.node_coord(v)
        px = ux + lam * (vx - ux)
        py = uy + lam * (vy - uy)
        return px, py

    def euclidean(self, ax: float, ay: float, bx: float, by: float) -> float:
        """欧氏距离"""
        return math.hypot(ax - bx, ay - by)

    def node_coord(self, node_idx: int) -> Tuple[float, float]:
        """
        获取节点坐标。
        坐标列表格式：x[0]=节点1, x[1]=节点2, ..., x[N-1]=节点N, x[N]=基站(节点0)
        node_idx=0 表示基站，node_idx=k(k≥1) 表示路网节点k。
        """
        if node_idx == 0:
            # 基站：坐标列表最后一个
            return float(self.x[self.total_nodes - 1]), float(self.y[self.total_nodes - 1])
        else:
            # 路网节点 k：对应坐标列表第 k-1 个（0-indexed）
            return float(self.x[node_idx - 1]), float(self.y[node_idx - 1])


def parse_instance(filepath: str) -> Instance:
    """从txt文件解析算例"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines()]

    # 过滤空行
    non_empty = [l for l in lines if l]

    idx = 0
    num_depots    = int(non_empty[idx]); idx += 1
    num_road_nodes = int(non_empty[idx]); idx += 1
    total_nodes   = int(non_empty[idx]); idx += 1
    num_edges     = int(non_empty[idx]); idx += 1
    num_drones    = int(non_empty[idx]); idx += 1

    battery       = float(non_empty[idx]); idx += 1
    speed         = float(non_empty[idx]); idx += 1
    energy_cost   = float(non_empty[idx]); idx += 1
    call_cost     = float(non_empty[idx]); idx += 1
    inspect_coef  = float(non_empty[idx]); idx += 1
    transfer_coef = float(non_empty[idx]); idx += 1

    # 兼容有无 drive_cost / big_m 行：
    # 连续读取所有不含 ',' 且不含 '(' 的行作为额外参数
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
        big_m = extra_params[-1]  # 最后一个数为 big_m

    x_list = [float(v.strip()) for v in non_empty[idx].split(',')]; idx += 1
    y_list = [float(v.strip()) for v in non_empty[idx].split(',')]; idx += 1

    x_arr = np.array(x_list)
    y_arr = np.array(y_list)

    # 解析需求边（跳过 (0,0)）
    edges = []
    while idx < len(non_empty):
        token = non_empty[idx].strip(); idx += 1
        token = token.replace('(', '').replace(')', '')
        parts = token.split(',')
        u, v = int(parts[0]), int(parts[1])
        if u == 0 and v == 0:
            continue  # 跳过基站自环
        edges.append((u, v))

    # 基站在坐标列表中的位置（最后一个）
    depot_idx = total_nodes - 1

    # 修正 num_edges 为实际解析到的需求边数（排除了基站自环）
    actual_num_edges = len(edges)

    inst = Instance(
        num_depots=num_depots,
        num_road_nodes=num_road_nodes,
        total_nodes=total_nodes,
        num_edges=actual_num_edges,
        num_drones=num_drones,
        battery=battery,
        speed=speed,
        energy_cost=energy_cost,
        call_cost=call_cost,
        inspect_coef=inspect_coef,
        transfer_coef=transfer_coef,
        big_m=big_m,
        x=x_arr,
        y=y_arr,
        edges=edges,
        depot_idx=depot_idx,
    )
    return inst


# ============================================================
# 2. 解的表示
# ============================================================

@dataclass
class SubEdge:
    """
    子边：原始边的一段（无断点时为整条边，有断点时为其中一段）
    end_a, end_b 为子边的两个端点的坐标（非节点索引，因断点不是节点）
    origin_edge_idx: 所属原始边的索引
    seg: 0=原始边（无断点）, 1=第一段（从u到断点）, 2=第二段（从断点到v）
    """
    origin_edge_idx: int   # 原始边索引
    seg: int               # 0=整条, 1=第一段(u→bp), 2=第二段(bp→v)
    ax: float              # 起点x（飞行时可正反两方向）
    ay: float              # 起点y
    bx: float              # 终点x
    by: float              # 终点y

    @property
    def length(self) -> float:
        return math.hypot(self.ax - self.bx, self.ay - self.by)


@dataclass
class DroneRoute:
    """
    单架无人机的路径：
    visit_order: 访问的子边列表（有向，每条子边记录飞行方向）
    directions: 每条子边的飞行方向，True=从a到b，False=从b到a
    """
    sub_edges: List[SubEdge] = field(default_factory=list)
    directions: List[bool] = field(default_factory=list)  # True: a->b, False: b->a

    def start_point(self, i: int) -> Tuple[float, float]:
        """第i条子边的实际起点"""
        se = self.sub_edges[i]
        if self.directions[i]:
            return se.ax, se.ay
        else:
            return se.bx, se.by

    def end_point(self, i: int) -> Tuple[float, float]:
        """第i条子边的实际终点"""
        se = self.sub_edges[i]
        if self.directions[i]:
            return se.bx, se.by
        else:
            return se.ax, se.ay


class Solution:
    """
    完整解：路径分配 + 断点位置
    breakpoints[i]: 第i条原始边的断点位置（None=无断点, 0<λ<1=有断点）
    routes: 每架无人机的路径
    """
    def __init__(self, num_drones: int, num_edges: int):
        self.num_drones = num_drones
        self.num_edges = num_edges
        self.breakpoints: List[Optional[float]] = [None] * num_edges
        self.routes: List[DroneRoute] = [DroneRoute() for _ in range(num_drones)]
        self._cost: Optional[float] = None  # 缓存费用

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


# ============================================================
# 3. 费用计算
# ============================================================

def compute_cost(sol: Solution, inst: Instance) -> float:
    """
    计算解的总费用：
    费用 = Σ_k [ call_cost  (若路径非空)
               + Σ_{子边} inspect_coef * 子边长度 * energy_cost
               + Σ_{转移} transfer_coef * 转移距离 * energy_cost
               + 基站到第一条子边的转移
               + 最后一条子边到基站的转移 ]
    """
    if sol._cost is not None:
        return sol._cost

    depot_x, depot_y = inst.node_coord(inst.depot_idx)

    total_cost = 0.0

    for drone_idx, route in enumerate(sol.routes):
        if not route.sub_edges:
            continue

        route_cost = inst.call_cost  # 调用成本

        # 从基站飞到第一条子边起点
        sx, sy = route.start_point(0)
        dist = inst.euclidean(depot_x, depot_y, sx, sy)
        route_cost += dist * inst.transfer_coef * inst.energy_cost

        # 逐条子边巡检 + 子边间转移
        for i, se in enumerate(route.sub_edges):
            # 巡检该子边
            route_cost += se.length * inst.inspect_coef * inst.energy_cost
            # 转移到下一条子边
            if i + 1 < len(route.sub_edges):
                ex, ey = route.end_point(i)
                nsx, nsy = route.start_point(i + 1)
                dist = inst.euclidean(ex, ey, nsx, nsy)
                route_cost += dist * inst.transfer_coef * inst.energy_cost

        # 从最后一条子边终点飞回基站
        ex, ey = route.end_point(len(route.sub_edges) - 1)
        dist = inst.euclidean(ex, ey, depot_x, depot_y)
        route_cost += dist * inst.transfer_coef * inst.energy_cost

        total_cost += route_cost

    sol._cost = total_cost
    return total_cost


def compute_route_distance(route: DroneRoute, inst: Instance) -> float:
    """
    计算单条路径的总能量消耗（用于检查电池约束）
    能量 = 巡检距离 × inspect_coef + 转移距离 × transfer_coef
    """
    if not route.sub_edges:
        return 0.0

    depot_x, depot_y = inst.node_coord(inst.depot_idx)

    inspect_dist = 0.0
    transfer_dist = 0.0

    # 基站 → 第一条子边起点（转移）
    sx, sy = route.start_point(0)
    transfer_dist += inst.euclidean(depot_x, depot_y, sx, sy)

    # 子边巡检 + 子边间转移
    for i, se in enumerate(route.sub_edges):
        inspect_dist += se.length
        if i + 1 < len(route.sub_edges):
            ex, ey = route.end_point(i)
            nsx, nsy = route.start_point(i + 1)
            transfer_dist += inst.euclidean(ex, ey, nsx, nsy)

    # 最后子边终点 → 基站（转移）
    ex, ey = route.end_point(len(route.sub_edges) - 1)
    transfer_dist += inst.euclidean(ex, ey, depot_x, depot_y)

    total_energy = inspect_dist * inst.inspect_coef + transfer_dist * inst.transfer_coef
    return total_energy


def compute_route_raw_distance(route: DroneRoute, inst: Instance) -> Tuple[float, float]:
    """
    计算单条路径的巡检距离和转移距离（用于输出明细）
    返回 (inspect_dist, transfer_dist)
    """
    if not route.sub_edges:
        return 0.0, 0.0

    depot_x, depot_y = inst.node_coord(inst.depot_idx)

    inspect_dist = sum(se.length for se in route.sub_edges)
    transfer_dist = 0.0

    sx, sy = route.start_point(0)
    transfer_dist += inst.euclidean(depot_x, depot_y, sx, sy)

    for i in range(len(route.sub_edges) - 1):
        ex, ey = route.end_point(i)
        nsx, nsy = route.start_point(i + 1)
        transfer_dist += inst.euclidean(ex, ey, nsx, nsy)

    ex, ey = route.end_point(len(route.sub_edges) - 1)
    transfer_dist += inst.euclidean(ex, ey, depot_x, depot_y)

    return inspect_dist, transfer_dist


def is_feasible(sol: Solution, inst: Instance) -> bool:
    """检查解是否可行（电池约束）"""
    for route in sol.routes:
        dist = compute_route_distance(route, inst)
        if dist > inst.battery:
            return False
    return True


def compute_route_cost(route: DroneRoute, inst: Instance) -> float:
    """计算单条路径的费用"""
    if not route.sub_edges:
        return 0.0
    depot_x, depot_y = inst.node_coord(inst.depot_idx)

    cost = inst.call_cost
    sx, sy = route.start_point(0)
    cost += inst.euclidean(depot_x, depot_y, sx, sy) * inst.transfer_coef * inst.energy_cost

    for i, se in enumerate(route.sub_edges):
        cost += se.length * inst.inspect_coef * inst.energy_cost
        if i + 1 < len(route.sub_edges):
            ex, ey = route.end_point(i)
            nsx, nsy = route.start_point(i + 1)
            cost += inst.euclidean(ex, ey, nsx, nsy) * inst.transfer_coef * inst.energy_cost

    ex, ey = route.end_point(len(route.sub_edges) - 1)
    cost += inst.euclidean(ex, ey, depot_x, depot_y) * inst.transfer_coef * inst.energy_cost
    return cost


# ============================================================
# 4. 子边构建工具
# ============================================================

def build_sub_edges(inst: Instance, breakpoints: List[Optional[float]]) -> List[SubEdge]:
    """
    根据当前断点配置，构建所有子边列表。
    无断点的边产生1条子边（整条），有断点的边产生2条子边。
    """
    sub_edges = []
    for ei, (u, v) in enumerate(inst.edges):
        ux, uy = inst.node_coord(u)
        vx, vy = inst.node_coord(v)
        lam = breakpoints[ei]
        if lam is None:
            # 无断点：整条边
            sub_edges.append(SubEdge(
                origin_edge_idx=ei, seg=0,
                ax=ux, ay=uy, bx=vx, by=vy
            ))
        else:
            # 有断点：断点坐标
            bpx = ux + lam * (vx - ux)
            bpy = uy + lam * (vy - uy)
            # 第一段 u→bp
            sub_edges.append(SubEdge(
                origin_edge_idx=ei, seg=1,
                ax=ux, ay=uy, bx=bpx, by=bpy
            ))
            # 第二段 bp→v
            sub_edges.append(SubEdge(
                origin_edge_idx=ei, seg=2,
                ax=bpx, ay=bpy, bx=vx, by=vy
            ))
    return sub_edges


def get_sub_edge_id(edge_idx: int, seg: int) -> Tuple[int, int]:
    """子边唯一标识符"""
    return (edge_idx, seg)


def rebuild_routes_with_new_breakpoints(
    sol: Solution, inst: Instance, new_breakpoints: List[Optional[float]]
) -> Solution:
    """
    根据新断点重建解中的所有子边引用（路径顺序不变，但子边坐标更新）
    注意：若一条无断点边被设置了断点，需要将原子边拆成两段；
         若一条有断点边断点被去除，需要将两段合并回一条。
    此函数仅适用于断点位置平移（不改变有无断点的状态），
    完整的路径重建由 PSO 中的坐标更新处理。
    """
    new_sol = Solution(sol.num_drones, sol.num_edges)
    new_sol.breakpoints = list(new_breakpoints)

    for di, route in enumerate(sol.routes):
        new_route = DroneRoute()
        for se, direction in zip(route.sub_edges, route.directions):
            ei = se.origin_edge_idx
            seg = se.seg
            lam = new_breakpoints[ei]
            u, v = inst.edges[ei]
            ux, uy = inst.node_coord(u)
            vx, vy = inst.node_coord(v)

            if lam is None:
                # 无断点子边（整条或保持原来的状态）
                if seg == 0:
                    new_se = SubEdge(origin_edge_idx=ei, seg=0,
                                     ax=ux, ay=uy, bx=vx, by=vy)
                    new_route.sub_edges.append(new_se)
                    new_route.directions.append(direction)
                # seg=1或2但新状态无断点：跳过（不应发生，保险处理）
            else:
                bpx = ux + lam * (vx - ux)
                bpy = uy + lam * (vy - uy)
                if seg == 0:
                    # 原来无断点，现在有断点：只保留整条方向，忽略（此处仅更新坐标用）
                    new_se = SubEdge(origin_edge_idx=ei, seg=0,
                                     ax=ux, ay=uy, bx=vx, by=vy)
                    new_route.sub_edges.append(new_se)
                    new_route.directions.append(direction)
                elif seg == 1:
                    new_se = SubEdge(origin_edge_idx=ei, seg=1,
                                     ax=ux, ay=uy, bx=bpx, by=bpy)
                    new_route.sub_edges.append(new_se)
                    new_route.directions.append(direction)
                elif seg == 2:
                    new_se = SubEdge(origin_edge_idx=ei, seg=2,
                                     ax=bpx, ay=bpy, bx=vx, by=vy)
                    new_route.sub_edges.append(new_se)
                    new_route.directions.append(direction)

        new_sol.routes[di] = new_route

    return new_sol


# ============================================================
# 5. 初始解构建（贪心）
# ============================================================

def best_direction(se: SubEdge, curr_x: float, curr_y: float) -> Tuple[bool, float, float]:
    """
    选择子边的最优飞行方向（最小化从当前位置到子边起点的距离）
    返回 (direction, start_x, start_y)
    """
    dist_ab = math.hypot(curr_x - se.ax, curr_y - se.ay)
    dist_ba = math.hypot(curr_x - se.bx, curr_y - se.by)
    if dist_ab <= dist_ba:
        return True, se.ax, se.ay
    else:
        return False, se.bx, se.by


def greedy_initial_solution(inst: Instance) -> Solution:
    """
    贪心初始解构建：
    1. 初始断点：全部无断点
    2. 对每架无人机，贪心地选择下一条最近的未分配子边（nearest neighbor）
    3. 若加入子边后超出电池容量，则停止该无人机的路径（尝试下一架）
    4. 若无法在任意无人机路径内完成某条子边，则为该边添加断点，将其拆分重试
    """
    sol = Solution(inst.num_drones, inst.num_edges)
    # 初始断点：全部无断点
    sol.breakpoints = [None] * inst.num_edges

    depot_x, depot_y = inst.node_coord(inst.depot_idx)

    # 构建初始子边集合
    all_sub_edges = build_sub_edges(inst, sol.breakpoints)
    unassigned = list(range(len(all_sub_edges)))

    # 每架无人机的当前位置和已用距离
    curr_pos = [(depot_x, depot_y)] * inst.num_drones
    curr_dist = [0.0] * inst.num_drones

    max_attempts = len(all_sub_edges) * inst.num_drones * 3

    attempt = 0
    while unassigned and attempt < max_attempts:
        attempt += 1
        placed = False

        # 找到已用距离最少的无人机
        drone_order = sorted(range(inst.num_drones), key=lambda d: curr_dist[d])

        for di in drone_order:
            cx, cy = curr_pos[di]
            used = curr_dist[di]

            # 找当前无人机可访问的最近未分配子边
            best_se_local = None
            best_cost_local = float('inf')
            best_uid = -1
            best_dir = True

            for uid in unassigned:
                se = all_sub_edges[uid]
                direction, sx, sy = best_direction(se, cx, cy)
                # 飞到子边起点的距离
                to_start = inst.euclidean(cx, cy, sx, sy)
                # 飞完子边后到基站的距离
                if direction:
                    ex, ey = se.bx, se.by
                else:
                    ex, ey = se.ax, se.ay
                to_depot = inst.euclidean(ex, ey, depot_x, depot_y)

                total = used + to_start + se.length + to_depot
                if total <= inst.battery:
                    # 贪心选最近
                    greedy_metric = to_start
                    if greedy_metric < best_cost_local:
                        best_cost_local = greedy_metric
                        best_se_local = se
                        best_uid = uid
                        best_dir = direction

            if best_se_local is not None:
                # 分配给当前无人机
                route = sol.routes[di]
                route.sub_edges.append(best_se_local)
                route.directions.append(best_dir)

                if best_dir:
                    ex, ey = best_se_local.bx, best_se_local.by
                    sx, sy = best_se_local.ax, best_se_local.ay
                else:
                    ex, ey = best_se_local.ax, best_se_local.ay
                    sx, sy = best_se_local.bx, best_se_local.by

                dist_to_start = inst.euclidean(curr_pos[di][0], curr_pos[di][1], sx, sy)
                curr_dist[di] += dist_to_start + best_se_local.length
                curr_pos[di] = (ex, ey)

                unassigned.remove(best_uid)
                placed = True
                break

        if not placed:
            # 没有任何无人机能容纳剩余子边，对第一条未分配子边尝试加断点
            uid = unassigned[0]
            se = all_sub_edges[uid]
            ei = se.origin_edge_idx

            if sol.breakpoints[ei] is None and se.seg == 0:
                # 为该边加断点（初始位置 0.5）
                sol.breakpoints[ei] = 0.5
                # 重建子边集合
                new_sub_edges = build_sub_edges(inst, sol.breakpoints)

                # 找到该边对应的两条新子边索引
                new_unassigned = []
                for new_uid, new_se in enumerate(new_sub_edges):
                    # 检查是否已在某条路径中
                    already_assigned = False
                    for route in sol.routes:
                        for existing_se in route.sub_edges:
                            if (existing_se.origin_edge_idx == new_se.origin_edge_idx and
                                    existing_se.seg == new_se.seg):
                                already_assigned = True
                                break
                        if already_assigned:
                            break
                    if not already_assigned:
                        new_unassigned.append(new_uid)

                all_sub_edges = new_sub_edges
                unassigned = new_unassigned
            else:
                # 已有断点但仍无法放置：强制分配给剩余空间最多的无人机
                di = min(range(inst.num_drones), key=lambda d: curr_dist[d])
                route = sol.routes[di]
                direction, sx, sy = best_direction(se, curr_pos[di][0], curr_pos[di][1])
                route.sub_edges.append(se)
                route.directions.append(direction)

                if direction:
                    ex, ey = se.bx, se.by
                else:
                    ex, ey = se.ax, se.ay

                dist_to_start = inst.euclidean(curr_pos[di][0], curr_pos[di][1], sx, sy)
                curr_dist[di] += dist_to_start + se.length
                curr_pos[di] = (ex, ey)
                unassigned.remove(uid)

    sol.invalidate_cache()
    return sol


# ============================================================
# 6. ALNS 破坏与修复算子
# ============================================================

# ---------- 破坏算子 ----------

def destroy_random_removal(sol: Solution, inst: Instance,
                           removal_fraction: float = 0.3) -> Tuple[Solution, List[SubEdge]]:
    """随机移除一定比例的子边"""
    new_sol = sol.copy()
    all_sub_edges_in_routes = []
    for di, route in enumerate(new_sol.routes):
        for i, se in enumerate(route.sub_edges):
            all_sub_edges_in_routes.append((di, i, se))

    if not all_sub_edges_in_routes:
        return new_sol, []

    num_remove = max(1, int(len(all_sub_edges_in_routes) * removal_fraction))
    num_remove = min(num_remove, len(all_sub_edges_in_routes))
    to_remove = random.sample(all_sub_edges_in_routes, num_remove)

    # 按位置倒序删除
    to_remove_sorted = sorted(to_remove, key=lambda x: (x[0], x[1]), reverse=True)
    removed = []
    for di, i, se in to_remove_sorted:
        new_sol.routes[di].sub_edges.pop(i)
        new_sol.routes[di].directions.pop(i)
        removed.append(se)

    new_sol.invalidate_cache()
    return new_sol, removed


def destroy_worst_removal(sol: Solution, inst: Instance,
                          removal_fraction: float = 0.3) -> Tuple[Solution, List[SubEdge]]:
    """
    移除最"昂贵"的子边（即移除后路径费用降低最多的子边）
    """
    new_sol = sol.copy()

    depot_x, depot_y = inst.node_coord(inst.depot_idx)

    # 计算每条子边的移除收益（近似：转移代价降低量）
    sub_edge_costs = []
    for di, route in enumerate(new_sol.routes):
        n = len(route.sub_edges)
        for i, se in enumerate(route.sub_edges):
            # 该子边的巡检费用
            inspect_cost = se.length * inst.inspect_coef * inst.energy_cost

            # 当前：前转移 + 巡检 + 后转移
            # 移除后：前驱直接连后继的转移
            if i == 0:
                prev_x, prev_y = depot_x, depot_y
            else:
                prev_x, prev_y = route.end_point(i - 1)

            if route.directions[i]:
                sx, sy = se.ax, se.ay
                ex, ey = se.bx, se.by
            else:
                sx, sy = se.bx, se.by
                ex, ey = se.ax, se.ay

            if i == n - 1:
                next_x, next_y = depot_x, depot_y
            else:
                next_x, next_y = route.start_point(i + 1)

            current_transfer = (inst.euclidean(prev_x, prev_y, sx, sy) +
                                 inst.euclidean(ex, ey, next_x, next_y))
            new_transfer = inst.euclidean(prev_x, prev_y, next_x, next_y)

            saving = inspect_cost + (current_transfer - new_transfer) * inst.transfer_coef * inst.energy_cost
            sub_edge_costs.append((saving, di, i, se))

    if not sub_edge_costs:
        return new_sol, []

    sub_edge_costs.sort(key=lambda x: -x[0])
    num_remove = max(1, int(len(sub_edge_costs) * removal_fraction))
    to_remove = sub_edge_costs[:num_remove]

    to_remove_sorted = sorted(to_remove, key=lambda x: (x[1], x[2]), reverse=True)
    removed = []
    for saving, di, i, se in to_remove_sorted:
        new_sol.routes[di].sub_edges.pop(i)
        new_sol.routes[di].directions.pop(i)
        removed.append(se)

    new_sol.invalidate_cache()
    return new_sol, removed


def destroy_route_removal(sol: Solution, inst: Instance,
                          removal_fraction: float = 0.3) -> Tuple[Solution, List[SubEdge]]:
    """
    移除某一架无人机的整条路径（或其中一段）
    """
    new_sol = sol.copy()
    non_empty_routes = [di for di, r in enumerate(new_sol.routes) if r.sub_edges]
    if not non_empty_routes:
        return new_sol, []

    di = random.choice(non_empty_routes)
    route = new_sol.routes[di]

    n = len(route.sub_edges)
    num_remove = max(1, int(n * removal_fraction))
    start_i = random.randint(0, n - num_remove)

    removed = []
    for _ in range(num_remove):
        removed.append(route.sub_edges[start_i])
        route.sub_edges.pop(start_i)
        route.directions.pop(start_i)

    new_sol.invalidate_cache()
    return new_sol, removed


# ---------- 修复算子 ----------

def repair_greedy_insert(sol: Solution, inst: Instance,
                         removed: List[SubEdge]) -> Solution:
    """
    贪心插入：对每条被移除的子边，找到插入后费用增加最小的位置
    """
    new_sol = sol.copy()
    depot_x, depot_y = inst.node_coord(inst.depot_idx)

    # 随机打乱插入顺序
    random.shuffle(removed)

    for se in removed:
        best_delta = float('inf')
        best_di = -1
        best_pos = -1
        best_dir = True

        for di in range(inst.num_drones):
            route = new_sol.routes[di]
            n = len(route.sub_edges)

            for pos in range(n + 1):  # 可插入的位置
                for direction in [True, False]:
                    if direction:
                        sx, sy = se.ax, se.ay
                        ex, ey = se.bx, se.by
                    else:
                        sx, sy = se.bx, se.by
                        ex, ey = se.ax, se.ay

                    # 前驱坐标
                    if pos == 0:
                        prev_x, prev_y = depot_x, depot_y
                    else:
                        prev_x, prev_y = route.end_point(pos - 1)

                    # 后继坐标
                    if pos == n:
                        next_x, next_y = depot_x, depot_y
                    else:
                        next_x, next_y = route.start_point(pos)

                    # 插入前的转移距离
                    old_transfer = inst.euclidean(prev_x, prev_y, next_x, next_y)
                    # 插入后的转移距离 + 巡检
                    new_transfer = (inst.euclidean(prev_x, prev_y, sx, sy) +
                                    se.length +
                                    inst.euclidean(ex, ey, next_x, next_y))

                    delta = ((new_transfer - old_transfer) * inst.transfer_coef * inst.energy_cost +
                             se.length * inst.inspect_coef * inst.energy_cost)

                    # 检查插入后是否满足电池约束
                    temp_route = DroneRoute(
                        sub_edges=route.sub_edges[:pos] + [se] + route.sub_edges[pos:],
                        directions=route.directions[:pos] + [direction] + route.directions[pos:]
                    )
                    dist = compute_route_distance(temp_route, inst)

                    if dist <= inst.battery and delta < best_delta:
                        best_delta = delta
                        best_di = di
                        best_pos = pos
                        best_dir = direction

        if best_di >= 0:
            route = new_sol.routes[best_di]
            route.sub_edges.insert(best_pos, se)
            route.directions.insert(best_pos, best_dir)
        else:
            # 无法满足约束，强制插入到距离最短的位置（违约插入）
            min_dist = float('inf')
            fb_di, fb_pos, fb_dir = 0, 0, True
            for di in range(inst.num_drones):
                route = new_sol.routes[di]
                n = len(route.sub_edges)
                for direction in [True, False]:
                    if direction:
                        sx, sy = se.ax, se.ay
                    else:
                        sx, sy = se.bx, se.by
                    if n == 0:
                        d = inst.euclidean(depot_x, depot_y, sx, sy)
                        if d < min_dist:
                            min_dist = d
                            fb_di, fb_pos, fb_dir = di, 0, direction
                    else:
                        prev_x, prev_y = route.end_point(n - 1)
                        d = inst.euclidean(prev_x, prev_y, sx, sy)
                        if d < min_dist:
                            min_dist = d
                            fb_di, fb_pos, fb_dir = di, n, direction
            route = new_sol.routes[fb_di]
            route.sub_edges.insert(fb_pos, se)
            route.directions.insert(fb_pos, fb_dir)

    new_sol.invalidate_cache()
    return new_sol


def repair_random_insert(sol: Solution, inst: Instance,
                         removed: List[SubEdge]) -> Solution:
    """随机插入：随机选择插入位置和方向"""
    new_sol = sol.copy()
    depot_x, depot_y = inst.node_coord(inst.depot_idx)

    random.shuffle(removed)

    for se in removed:
        # 随机尝试多次找满足约束的位置
        feasible_positions = []
        for di in range(inst.num_drones):
            route = new_sol.routes[di]
            n = len(route.sub_edges)
            for pos in range(n + 1):
                for direction in [True, False]:
                    temp_route = DroneRoute(
                        sub_edges=route.sub_edges[:pos] + [se] + route.sub_edges[pos:],
                        directions=route.directions[:pos] + [direction] + route.directions[pos:]
                    )
                    dist = compute_route_distance(temp_route, inst)
                    if dist <= inst.battery:
                        feasible_positions.append((di, pos, direction))

        if feasible_positions:
            di, pos, direction = random.choice(feasible_positions)
        else:
            # 随机强制插入
            di = random.randint(0, inst.num_drones - 1)
            pos = random.randint(0, len(new_sol.routes[di].sub_edges))
            direction = random.choice([True, False])

        route = new_sol.routes[di]
        route.sub_edges.insert(pos, se)
        route.directions.insert(pos, direction)

    new_sol.invalidate_cache()
    return new_sol


# ============================================================
# 7. PSO 优化断点位置
# ============================================================

class PSOBreakpointOptimizer:
    """
    粒子群优化器：优化哪些边设置断点以及断点位置
    决策变量：对每条边，一个值 p_i ∈ [0, 1]
      - p_i < 0.5: 不设断点
      - p_i ≥ 0.5: 设断点，断点位置 λ = (p_i - 0.5) * 2 ∈ (0, 1)
    """
    def __init__(self, inst: Instance, num_particles: int = 20,
                 max_iter: int = 30, w: float = 0.7,
                 c1: float = 1.5, c2: float = 1.5):
        self.inst = inst
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w    # 惯性权重
        self.c1 = c1  # 认知加速系数
        self.c2 = c2  # 社会加速系数
        self.dim = inst.num_edges

    def _decode_breakpoints(self, position: np.ndarray) -> List[Optional[float]]:
        """将粒子位置解码为断点配置"""
        bps = []
        for i, p in enumerate(position):
            if p >= 0.5:
                lam = (p - 0.5) * 2.0  # ∈ (0, 1)
                lam = max(0.05, min(0.95, lam))  # 防止断点在端点上
                bps.append(lam)
            else:
                bps.append(None)
        return bps

    def optimize(self, base_route_sol: Solution, evaluator_fn) -> Tuple[List[Optional[float]], float]:
        """
        在固定路径结构（base_route_sol）下，优化断点位置。
        evaluator_fn(breakpoints) -> cost: 给定断点配置，评估当前路径下的费用。

        注意：路径中子边的拓扑结构（哪些子边访问哪条原始边的哪一段）
        需要与断点配置匹配。本函数仅适用于对当前解中已有断点的边优化其位置。
        """
        # 初始化粒子
        # 从当前解的断点状态初始化
        init_pos = np.zeros(self.dim)
        for i, bp in enumerate(base_route_sol.breakpoints):
            if bp is not None:
                init_pos[i] = 0.5 + bp * 0.5
            else:
                init_pos[i] = random.uniform(0.0, 0.49)

        particles = np.tile(init_pos, (self.num_particles, 1))
        # 添加随机扰动
        noise = np.random.uniform(-0.2, 0.2, (self.num_particles, self.dim))
        particles = np.clip(particles + noise, 0.0, 1.0)
        particles[0] = init_pos  # 保留原始解

        velocities = np.random.uniform(-0.1, 0.1, (self.num_particles, self.dim))

        # 评估初始粒子
        pbest_pos = particles.copy()
        pbest_cost = np.array([evaluator_fn(self._decode_breakpoints(p)) for p in particles])

        gbest_idx = np.argmin(pbest_cost)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_cost = pbest_cost[gbest_idx]

        for iteration in range(self.max_iter):
            for i in range(self.num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                # 速度更新
                velocities[i] = (self.w * velocities[i] +
                                  self.c1 * r1 * (pbest_pos[i] - particles[i]) +
                                  self.c2 * r2 * (gbest_pos - particles[i]))
                velocities[i] = np.clip(velocities[i], -0.3, 0.3)

                # 位置更新
                particles[i] = np.clip(particles[i] + velocities[i], 0.0, 1.0)

                # 评估
                cost = evaluator_fn(self._decode_breakpoints(particles[i]))
                if cost < pbest_cost[i]:
                    pbest_cost[i] = cost
                    pbest_pos[i] = particles[i].copy()
                    if cost < gbest_cost:
                        gbest_cost = cost
                        gbest_pos = particles[i].copy()

            # 惯性权重衰减
            self.w = max(0.4, self.w * 0.98)

        return self._decode_breakpoints(gbest_pos), gbest_cost


# ============================================================
# 8. ALNS 主框架
# ============================================================

class ALNSPSOSolver:
    """
    ALNS + PSO 求解器
    """
    def __init__(self, inst: Instance,
                 # ALNS 参数
                 max_iter: int = 500,
                 segment_size: int = 50,
                 removal_min: float = 0.1,
                 removal_max: float = 0.4,
                 sigma1: float = 33.0,  # 发现新全局最优
                 sigma2: float = 9.0,   # 改进当前解
                 sigma3: float = 3.0,   # 接受但未改进
                 decay: float = 0.8,    # 权重衰减率
                 sa_temp_init: float = None,  # SA初始温度（None自动）
                 sa_cooling: float = 0.998,
                 # PSO 参数
                 pso_freq: int = 50,    # 每隔多少ALNS迭代运行一次PSO
                 pso_particles: int = 15,
                 pso_iter: int = 20,
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

        self.pso = PSOBreakpointOptimizer(inst, pso_particles, pso_iter)

        # 破坏算子
        self.destroy_ops = [
            destroy_random_removal,
            destroy_worst_removal,
            destroy_route_removal,
        ]
        self.destroy_names = ["random", "worst", "route"]
        self.destroy_weights = [1.0] * len(self.destroy_ops)
        self.destroy_scores = [0.0] * len(self.destroy_ops)
        self.destroy_counts = [0] * len(self.destroy_ops)

        # 修复算子
        self.repair_ops = [
            repair_greedy_insert,
            repair_random_insert,
        ]
        self.repair_names = ["greedy", "random"]
        self.repair_weights = [1.0] * len(self.repair_ops)
        self.repair_scores = [0.0] * len(self.repair_ops)
        self.repair_counts = [0] * len(self.repair_ops)

        # SA初始温度自动设置
        self.sa_temp_init = sa_temp_init
        self.sa_temp = sa_temp_init

        # 历史记录
        self.cost_history = []
        self.best_cost_history = []

    def _roulette_select(self, weights: List[float]) -> int:
        """轮盘赌选择"""
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

    def _update_weights(self, destroy_idx: int, repair_idx: int, score: float):
        """更新算子权重"""
        self.destroy_scores[destroy_idx] += score
        self.destroy_counts[destroy_idx] += 1
        self.repair_scores[repair_idx] += score
        self.repair_counts[repair_idx] += 1

    def _normalize_weights(self):
        """每个 segment 结束后更新权重"""
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
        """模拟退火接受准则"""
        if new_cost <= current_cost:
            return True
        if self.sa_temp is None or self.sa_temp <= 0:
            return False
        delta = new_cost - current_cost
        prob = math.exp(-delta / self.sa_temp)
        return random.random() < prob

    def _run_pso_on_solution(self, sol: Solution) -> Solution:
        """
        在当前路径结构下，用PSO优化断点位置。
        仅对已有断点的边优化其位置；不改变路径结构（子边访问顺序不变）。
        同时也探索为某些边添加/去除断点。
        """
        inst = self.inst

        def evaluator(new_bps: List[Optional[float]]) -> float:
            """
            给定新断点，重建子边坐标后重新评估费用。
            路径中的子边访问顺序保持不变，但需要匹配新断点的拓扑。
            """
            # 检查新断点与当前路径的拓扑兼容性
            # 路径中每条子边的 (origin_edge_idx, seg) 需要与新断点匹配
            for di, route in enumerate(sol.routes):
                for se in route.sub_edges:
                    ei = se.origin_edge_idx
                    new_bp = new_bps[ei]
                    if new_bp is None and se.seg in (1, 2):
                        # 路径期望断点但新配置无断点：不兼容，惩罚
                        return float('inf')
                    if new_bp is not None and se.seg == 0:
                        # 路径期望完整边但新配置有断点：不兼容，惩罚
                        return float('inf')

            # 拓扑兼容，只更新子边坐标
            new_sol = rebuild_routes_with_new_breakpoints(sol, inst, new_bps)
            cost = compute_cost(new_sol, inst)
            return cost

        new_bps, new_cost = self.pso.optimize(sol, evaluator)
        new_sol = rebuild_routes_with_new_breakpoints(sol, inst, new_bps)
        new_sol.invalidate_cache()
        return new_sol

    def _try_add_remove_breakpoints(self, sol: Solution) -> Solution:
        """
        局部搜索：尝试为某条边添加断点或去除断点，看是否能改善费用。
        添加断点后需要将子边拆分并重新分配（贪心），去除断点需要合并子边。
        """
        inst = self.inst
        best_sol = sol
        best_cost = compute_cost(sol, inst)

        # 随机选几条边尝试
        edge_indices = random.sample(range(inst.num_edges), min(3, inst.num_edges))

        for ei in edge_indices:
            trial_sol = sol.copy()

            if trial_sol.breakpoints[ei] is None:
                # 尝试为该边添加断点（位置 0.5 初始，后续PSO会优化）
                trial_sol.breakpoints[ei] = 0.5

                # 找到路径中该边对应的子边（整条，seg=0），将其替换为两段
                u, v = inst.edges[ei]
                ux, uy = inst.node_coord(u)
                vx, vy = inst.node_coord(v)
                bpx = ux + 0.5 * (vx - ux)
                bpy = uy + 0.5 * (vy - uy)

                se_seg1 = SubEdge(origin_edge_idx=ei, seg=1,
                                  ax=ux, ay=uy, bx=bpx, by=bpy)
                se_seg2 = SubEdge(origin_edge_idx=ei, seg=2,
                                  ax=bpx, ay=bpy, bx=vx, by=vy)

                # 找到原来整条边在哪个路径中的哪个位置
                found = False
                for di, route in enumerate(trial_sol.routes):
                    for i, se in enumerate(route.sub_edges):
                        if se.origin_edge_idx == ei and se.seg == 0:
                            direction = route.directions[i]
                            # 替换：将整条边替换为两段
                            route.sub_edges.pop(i)
                            route.directions.pop(i)
                            if direction:
                                route.sub_edges.insert(i, se_seg2)
                                route.directions.insert(i, True)
                                route.sub_edges.insert(i, se_seg1)
                                route.directions.insert(i, True)
                            else:
                                route.sub_edges.insert(i, se_seg1)
                                route.directions.insert(i, False)
                                route.sub_edges.insert(i, se_seg2)
                                route.directions.insert(i, False)
                            found = True
                            break
                    if found:
                        break

            else:
                # 尝试去除断点，将两段合并为整条边
                old_lam = trial_sol.breakpoints[ei]
                trial_sol.breakpoints[ei] = None

                u, v = inst.edges[ei]
                ux, uy = inst.node_coord(u)
                vx, vy = inst.node_coord(v)
                whole_se = SubEdge(origin_edge_idx=ei, seg=0,
                                   ax=ux, ay=uy, bx=vx, by=vy)

                # 找到两段子边在路径中的位置，决定如何合并
                seg1_loc = None  # (di, i)
                seg2_loc = None

                for di, route in enumerate(trial_sol.routes):
                    for i, se in enumerate(route.sub_edges):
                        if se.origin_edge_idx == ei:
                            if se.seg == 1:
                                seg1_loc = (di, i)
                            elif se.seg == 2:
                                seg2_loc = (di, i)

                if seg1_loc is None or seg2_loc is None:
                    continue  # 找不到对应子边，跳过

                di1, i1 = seg1_loc
                di2, i2 = seg2_loc

                if di1 == di2:
                    # 同一条路径：直接合并
                    # 删除两段，插入整条
                    positions = sorted([i1, i2], reverse=True)
                    dirs = [trial_sol.routes[di1].directions[i1],
                            trial_sol.routes[di1].directions[i2]]
                    for pos in positions:
                        trial_sol.routes[di1].sub_edges.pop(pos)
                        trial_sol.routes[di1].directions.pop(pos)
                    insert_pos = min(i1, i2)
                    # 方向：根据第一段方向决定整体方向
                    dir1 = dirs[0] if i1 < i2 else dirs[1]
                    trial_sol.routes[di1].sub_edges.insert(insert_pos, whole_se)
                    trial_sol.routes[di1].directions.insert(insert_pos, dir1)
                else:
                    # 不同路径：将seg2的子边移到seg1所在路径
                    # 先删除seg2
                    trial_sol.routes[di2].sub_edges.pop(i2)
                    trial_sol.routes[di2].directions.pop(i2)
                    # 删除seg1，插入整条
                    trial_sol.routes[di1].sub_edges.pop(i1)
                    trial_sol.routes[di1].directions.pop(i1)
                    trial_sol.routes[di1].sub_edges.insert(i1, whole_se)
                    trial_sol.routes[di1].directions.insert(i1, True)

            trial_sol.invalidate_cache()
            trial_cost = compute_cost(trial_sol, inst)
            if trial_cost < best_cost:
                best_cost = trial_cost
                best_sol = trial_sol

        return best_sol

    def solve(self, initial_sol: Optional[Solution] = None,
              verbose: bool = True) -> Tuple[Solution, List[float]]:
        """
        主求解函数
        返回 (best_solution, cost_history)
        """
        inst = self.inst

        # 生成初始解
        if initial_sol is None:
            if verbose:
                print("生成贪心初始解...")
            current_sol = greedy_initial_solution(inst)
        else:
            current_sol = initial_sol.copy()

        current_cost = compute_cost(current_sol, inst)
        best_sol = current_sol.copy()
        best_cost = current_cost

        # 自动设置SA初始温度（约为初始解费用的5%）
        if self.sa_temp is None:
            self.sa_temp = current_cost * 0.05
            if verbose:
                print(f"SA初始温度自动设置为: {self.sa_temp:.2f}")

        if verbose:
            print(f"初始解费用: {current_cost:.4f}")

        self.cost_history = [current_cost]
        self.best_cost_history = [best_cost]

        start_time = time.time()

        for iteration in range(self.max_iter):
            removal_fraction = random.uniform(self.removal_min, self.removal_max)

            # 选择破坏算子
            di_op = self._roulette_select(self.destroy_weights)
            ri_op = self._roulette_select(self.repair_weights)

            # 执行破坏
            destroyed_sol, removed = self.destroy_ops[di_op](
                current_sol, inst, removal_fraction
            )

            # 执行修复
            new_sol = self.repair_ops[ri_op](destroyed_sol, inst, removed)

            # 每隔 pso_freq 次迭代，对当前解运行PSO优化断点
            if (iteration + 1) % self.pso_freq == 0:
                new_sol = self._run_pso_on_solution(new_sol)
                new_sol = self._try_add_remove_breakpoints(new_sol)

            new_cost = compute_cost(new_sol, inst)

                        # 判断接受/更新
            score = 0.0
            if new_cost < best_cost:
                best_sol = new_sol.copy()
                best_cost = new_cost
                current_sol = new_sol.copy()
                current_cost = new_cost
                score = self.sigma1
            elif new_cost < current_cost:
                current_sol = new_sol.copy()
                current_cost = new_cost
                score = self.sigma2
            elif self._sa_accept(current_cost, new_cost):
                current_sol = new_sol.copy()
                current_cost = new_cost
                score = self.sigma3

            # 更新算子得分
            self._update_weights(di_op, ri_op, score)

            # SA 降温
            self.sa_temp *= self.sa_cooling

            # 每个 segment 结束后归一化权重
            if (iteration + 1) % self.segment_size == 0:
                self._normalize_weights()
                if verbose:
                    elapsed = time.time() - start_time
                    print(f"  Iter {iteration+1:4d}/{self.max_iter} | "
                          f"current: {current_cost:.4f} | best: {best_cost:.4f} | "
                          f"T: {self.sa_temp:.4f} | time: {elapsed:.1f}s")

            self.cost_history.append(current_cost)
            self.best_cost_history.append(best_cost)

        # 最终对最优解做一次PSO优化
        if verbose:
            print("最终 PSO 精化断点位置...")
        final_sol = self._run_pso_on_solution(best_sol)
        final_cost = compute_cost(final_sol, inst)
        if final_cost < best_cost:
            best_sol = final_sol
            best_cost = final_cost

        if verbose:
            print(f"\n求解完成！最优费用: {best_cost:.4f}")

        return best_sol, self.best_cost_history


# ============================================================
# 9. 结果输出与可视化
# ============================================================

def print_solution_detail(sol: Solution, inst: Instance):
    """打印解的详细信息"""
    print("\n" + "="*60)
    print("解的详细信息")
    print("="*60)

    total_cost = compute_cost(sol, inst)
    depot_x, depot_y = inst.node_coord(inst.depot_idx)

    print(f"\n总费用: {total_cost:.4f} 元")
    print(f"电池容量: {inst.battery:.1f} m")

    # 断点信息
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

    # 路径信息
    print("\n--- 各无人机路径 ---")
    for di, route in enumerate(sol.routes):
        if not route.sub_edges:
            print(f"\n  无人机 {di+1}: 未使用")
            continue

        energy = compute_route_distance(route, inst)
        inspect_dist_raw, transfer_dist_raw = compute_route_raw_distance(route, inst)
        total_dist_raw = inspect_dist_raw + transfer_dist_raw
        cost = compute_route_cost(route, inst)
        feasible = energy <= inst.battery

        print(f"\n  无人机 {di+1}:")
        print(f"    总飞行距离: {total_dist_raw:.2f} m  (巡检: {inspect_dist_raw:.2f}m + 转移: {transfer_dist_raw:.2f}m)")
        print(f"    能量消耗: {energy:.4f} kwh  (电池容量: {inst.battery:.4f} kwh)  {'[可行]' if feasible else '[超出电池!]'}")
        print(f"    路径费用: {cost:.4f} 元")
        print(f"    路径:")
        print(f"      基站({depot_x:.1f},{depot_y:.1f})")

        for i, (se, direction) in enumerate(zip(route.sub_edges, route.directions)):
            u_orig, v_orig = inst.edges[se.origin_edge_idx]
            seg_str = {0: "整条", 1: "第1段", 2: "第2段"}[se.seg]
            if direction:
                sx, sy = se.ax, se.ay
                ex, ey = se.bx, se.by
                dir_str = "正向"
            else:
                sx, sy = se.bx, se.by
                ex, ey = se.ax, se.ay
                dir_str = "反向"

            print(f"      → [子边{i+1}] 原边({u_orig},{v_orig}) {seg_str} "
                  f"{dir_str} ({sx:.1f},{sy:.1f})→({ex:.1f},{ey:.1f}) "
                  f"长={se.length:.2f}m")

        print(f"      → 基站({depot_x:.1f},{depot_y:.1f})")

        # 费用明细
        print(f"    费用明细:")
        print(f"      调用成本: {inst.call_cost:.2f} 元")
        print(f"      巡检成本: {inspect_dist_raw:.2f}m × {inst.inspect_coef} × {inst.energy_cost} = "
              f"{inspect_dist_raw * inst.inspect_coef * inst.energy_cost:.4f} 元")
        print(f"      转移成本: {transfer_dist_raw:.2f}m × {inst.transfer_coef} × {inst.energy_cost} = "
              f"{transfer_dist_raw * inst.transfer_coef * inst.energy_cost:.4f} 元")

    print("\n" + "="*60)


def plot_convergence(cost_history: List[float], output_path: str):
    """绘制收敛曲线"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cost_history, color='#2196F3', linewidth=1.2, alpha=0.8, label='Best Cost')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Cost', fontsize=12)
    ax.set_title('ALNS+PSO Convergence Curve', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"收敛曲线已保存: {output_path}")


def plot_solution(sol: Solution, inst: Instance, output_path: str,
                  title: str = "UAV Arc Routing Solution"):
    """
    可视化解：
    - 灰色：需求边（原始路网）
    - 彩色线：各无人机路径（巡检 + 转移虚线）
    - 星号：断点位置
    - 方块：基站
    - 圆圈：路网节点
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    depot_x, depot_y = inst.node_coord(inst.depot_idx)

    # 画所有需求边（背景，灰色）
    for ei, (u, v) in enumerate(inst.edges):
        ux, uy = inst.node_coord(u)
        vx, vy = inst.node_coord(v)
        ax.plot([ux, vx], [uy, vy], color='#CCCCCC', linewidth=2.5,
                zorder=1, solid_capstyle='round')

    # 无人机颜色
    drone_colors = ['#E74C3C', '#2ECC71', '#3498DB', '#F39C12',
                    '#9B59B6', '#1ABC9C', '#E67E22', '#2980B9']

    legend_handles = []
    active_drones = 0

    for di, route in enumerate(sol.routes):
        if not route.sub_edges:
            continue

        color = drone_colors[di % len(drone_colors)]
        active_drones += 1

        # 构建路径点序列
        path_points = [(depot_x, depot_y)]
        path_types = []  # 'transfer' or 'inspect'

        for i, (se, direction) in enumerate(zip(route.sub_edges, route.directions)):
            if direction:
                sx, sy = se.ax, se.ay
                ex, ey = se.bx, se.by
            else:
                sx, sy = se.bx, se.by
                ex, ey = se.ax, se.ay

            # 转移段（基站到子边，或上一条子边到本条子边）
            path_types.append('transfer')
            path_points.append((sx, sy))
            # 巡检段
            path_types.append('inspect')
            path_points.append((ex, ey))

        # 返回基站
        path_types.append('transfer')
        path_points.append((depot_x, depot_y))

        # 绘制路径
        for i in range(len(path_types)):
            x0, y0 = path_points[i]
            x1, y1 = path_points[i + 1]
            ptype = path_types[i]
            if ptype == 'inspect':
                ax.plot([x0, x1], [y0, y1], color=color, linewidth=3.0,
                        zorder=3, solid_capstyle='round')
                # 箭头指示方向
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                dx, dy = x1 - x0, y1 - y0
                norm = math.hypot(dx, dy)
                if norm > 1e-9:
                    ax.annotate('', xy=(mx + dx/norm*5, my + dy/norm*5),
                                xytext=(mx - dx/norm*5, my - dy/norm*5),
                                arrowprops=dict(arrowstyle='->', color=color,
                                                lw=1.5),
                                zorder=4)
            else:
                ax.plot([x0, x1], [y0, y1], color=color, linewidth=1.5,
                        linestyle='--', alpha=0.6, zorder=2)

        insp_d, trans_d = compute_route_raw_distance(route, inst)
        cost = compute_route_cost(route, inst)
        patch = mpatches.Patch(color=color,
                               label=f'UAV {di+1} (dist={insp_d+trans_d:.0f}m, cost={cost:.2f})')
        legend_handles.append(patch)

    # 画断点
    for ei, bp in enumerate(sol.breakpoints):
        if bp is not None:
            bpx, bpy = inst.point_on_edge(ei, bp)
            ax.scatter(bpx, bpy, marker='*', s=200, color='#FF6B6B',
                       zorder=6, edgecolors='#C0392B', linewidths=0.8)

    # 画路网节点
    for ni in range(1, inst.total_nodes):  # 跳过基站
        nx_, ny_ = inst.node_coord(ni)
        ax.scatter(nx_, ny_, marker='o', s=80, color='#5B9BD5',
                   zorder=5, edgecolors='white', linewidths=0.8)
        ax.annotate(str(ni), (nx_, ny_), textcoords='offset points',
                    xytext=(4, 4), fontsize=7, color='#2C3E50')

    # 画基站
    ax.scatter(depot_x, depot_y, marker='s', s=300, color='#E74C3C',
               zorder=7, edgecolors='white', linewidths=1.5)
    ax.annotate('DEPOT', (depot_x, depot_y), textcoords='offset points',
                xytext=(5, 5), fontsize=9, color='#E74C3C', fontweight='bold')

    # 如果有断点，添加图例项
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
                      cost_history: List[float]):
    """保存解到txt文件"""
    total_cost = compute_cost(sol, inst)
    depot_x, depot_y = inst.node_coord(inst.depot_idx)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("ALNS+PSO 求解结果\n")
        f.write("="*60 + "\n")
        f.write(f"\n总费用: {total_cost:.6f} 元\n")
        f.write(f"初始解费用: {cost_history[0]:.6f} 元\n")
        f.write(f"改进率: {(cost_history[0] - total_cost) / cost_history[0] * 100:.2f}%\n")

        f.write("\n--- 断点配置 ---\n")
        for ei, bp in enumerate(sol.breakpoints):
            u, v = inst.edges[ei]
            if bp is not None:
                edge_len = inst.edge_length(ei)
                bp_coord = inst.point_on_edge(ei, bp)
                f.write(f"  边({u},{v}) 长度={edge_len:.2f}m: "
                        f"断点位置 lambda={bp:.6f}, "
                        f"坐标=({bp_coord[0]:.4f}, {bp_coord[1]:.4f})\n")
            else:
                u_c, v_c = inst.node_coord(u), inst.node_coord(v)
                edge_len = inst.edge_length(ei)
                f.write(f"  边({u},{v}) 长度={edge_len:.2f}m: 无断点\n")

        f.write("\n--- 各无人机路径 ---\n")
        for di, route in enumerate(sol.routes):
            if not route.sub_edges:
                f.write(f"\n无人机 {di+1}: 未使用\n")
                continue

            energy = compute_route_distance(route, inst)
            inspect_dist, transfer_dist = compute_route_raw_distance(route, inst)
            cost = compute_route_cost(route, inst)
            feasible = energy <= inst.battery

            f.write(f"\n无人机 {di+1}:\n")
            f.write(f"  总飞行距离: {inspect_dist+transfer_dist:.4f} m  "
                    f"(巡检: {inspect_dist:.4f}m, 转移: {transfer_dist:.4f}m)\n")
            f.write(f"  能量消耗: {energy:.6f} kwh  "
                    f"(电池容量: {inst.battery:.4f} kwh) "
                    f"{'[可行]' if feasible else '[超出电池!]'}\n")
            f.write(f"  路径费用: {cost:.6f} 元\n")
            f.write(f"  路径:\n")
            f.write(f"    基站({depot_x:.2f},{depot_y:.2f})\n")

            for i, (se, direction) in enumerate(zip(route.sub_edges, route.directions)):
                u_orig, v_orig = inst.edges[se.origin_edge_idx]
                seg_str = {0: "整条", 1: "第1段(u→bp)", 2: "第2段(bp→v)"}[se.seg]
                if direction:
                    sx, sy = se.ax, se.ay
                    ex, ey = se.bx, se.by
                    dir_str = "正向"
                else:
                    sx, sy = se.bx, se.by
                    ex, ey = se.ax, se.ay
                    dir_str = "反向"

                f.write(f"    → [子边{i+1}] 原边({u_orig},{v_orig}) {seg_str} "
                        f"{dir_str} ({sx:.2f},{sy:.2f})→({ex:.2f},{ey:.2f}) "
                        f"长={se.length:.4f}m\n")

            f.write(f"    → 基站({depot_x:.2f},{depot_y:.2f})\n")

            f.write(f"  费用明细:\n")
            f.write(f"    调用成本: {inst.call_cost:.4f} 元\n")
            f.write(f"    巡检成本: {inspect_dist:.4f}m × {inst.inspect_coef} × "
                    f"{inst.energy_cost} = "
                    f"{inspect_dist * inst.inspect_coef * inst.energy_cost:.6f} 元\n")
            f.write(f"    转移成本: {transfer_dist:.4f}m × {inst.transfer_coef} × "
                    f"{inst.energy_cost} = "
                    f"{transfer_dist * inst.transfer_coef * inst.energy_cost:.6f} 元\n")

        f.write("\n--- 收敛过程（每100次迭代）---\n")
        step = max(1, len(cost_history) // 100)
        for i in range(0, len(cost_history), step):
            f.write(f"  Iter {i:5d}: {cost_history[i]:.6f}\n")
        f.write(f"  Iter {len(cost_history)-1:5d}: {cost_history[-1]:.6f}\n")

    print(f"结果文件已保存: {output_path}")


# ============================================================
# 10. 主函数入口
# ============================================================

def _mirror_output_dir(instance_path: str, src_root: str = "算例", dst_root: str = "结果") -> str:
    """
    将算例路径中的根目录名替换为结果根目录名，保持子目录结构。
    例如：.../算例/小中规模算例/ → .../结果/小中规模算例/
    若路径中不含 src_root，则回退为与算例同级目录。
    """
    abs_path = os.path.abspath(instance_path)
    parts = abs_path.replace(os.sep, '/').split('/')
    # 从右向左找第一个名为 src_root 的目录
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == src_root:
            parts[i] = dst_root
            # 输出目录为文件所在目录（去掉文件名部分）
            new_dir = '/'.join(parts[:-1])
            return new_dir
    # 未找到 src_root，回退到算例文件所在目录
    return os.path.dirname(abs_path)


def solve_instance(instance_path: str,
                   output_dir: str = None,
                   max_iter: int = 500,
                   pso_freq: int = 50,
                   pso_particles: int = 15,
                   pso_iter: int = 20,
                   verbose: bool = True) -> Tuple[Solution, float]:
    """
    对单个算例文件运行 ALNS+PSO 求解。
    默认输出目录：将路径中的 '算例' 替换为 '结果'，保持子目录结构不变。
    """
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
        print(f"  电池容量: {inst.battery:.1f} m")
        print(f"{'='*60}")

    solver = ALNSPSOSolver(
        inst,
        max_iter=max_iter,
        pso_freq=pso_freq,
        pso_particles=pso_particles,
        pso_iter=pso_iter,
    )

    best_sol, cost_history = solver.solve(verbose=verbose)
    best_cost = compute_cost(best_sol, inst)

    if verbose:
        print_solution_detail(best_sol, inst)

    # 保存结果文件（文件名与算例文件名相同，仅目录改为 结果/...）
    txt_path = os.path.join(output_dir, f"{basename}.txt")
    save_solution_txt(best_sol, inst, txt_path, cost_history)

    # 收敛曲线
    conv_path = os.path.join(output_dir, f"{basename}_convergence.png")
    plot_convergence(cost_history, conv_path)

    # 路径可视化
    vis_path = os.path.join(output_dir, f"{basename}_solution.png")
    plot_solution(best_sol, inst, vis_path,
                  title=f"{basename} | Cost={best_cost:.2f}")

    return best_sol, best_cost


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ALNS+PSO 无人机弧路径问题求解器（连续空间断点决策）"
    )
    parser.add_argument(
        "instance",
        nargs="?",
        default=None,
        help="算例文件路径（.txt），不指定则使用默认测试算例"
    )
    parser.add_argument(
        "--output_dir", "-o",
        default=None,
        help="输出目录（默认与算例文件同目录）"
    )
    parser.add_argument(
        "--max_iter", "-n",
        type=int,
        default=500,
        help="ALNS 最大迭代次数（默认500）"
    )
    parser.add_argument(
        "--pso_freq",
        type=int,
        default=50,
        help="每隔多少ALNS迭代运行一次PSO（默认50）"
    )
    parser.add_argument(
        "--pso_particles",
        type=int,
        default=15,
        help="PSO粒子数（默认15）"
    )
    parser.add_argument(
        "--pso_iter",
        type=int,
        default=20,
        help="PSO迭代次数（默认20）"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="批量求解某目录下所有算例"
    )
    parser.add_argument(
        "--batch_dir",
        default=None,
        help="批量求解的目录（配合 --batch 使用）"
    )

    args = parser.parse_args()

    if args.batch and args.batch_dir:
        # 批量模式
        txt_files = [
            os.path.join(args.batch_dir, f)
            for f in os.listdir(args.batch_dir)
            if f.endswith('.txt')
        ]
        txt_files.sort()
        print(f"批量求解 {len(txt_files)} 个算例，目录: {args.batch_dir}")

        results = []
        for fp in txt_files:
            try:
                # 批量模式下若未指定输出目录，自动镜像到 结果/ 目录
                if args.output_dir:
                    out_dir = args.output_dir
                else:
                    out_dir = _mirror_output_dir(fp, src_root="算例", dst_root="结果")
                sol, cost = solve_instance(
                    fp,
                    output_dir=out_dir,
                    max_iter=args.max_iter,
                    pso_freq=args.pso_freq,
                    pso_particles=args.pso_particles,
                    pso_iter=args.pso_iter,
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
        # 单算例模式
        if args.instance is None:
            # 默认使用小中规模算例
            default_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "算例", "小中规模算例"
            )
            candidates = [f for f in os.listdir(default_dir) if f.endswith('.txt')]
            if candidates:
                candidates.sort()
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
            verbose=True,
        )

