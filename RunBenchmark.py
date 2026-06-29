"""
RunBenchmark.py
===============
对「算例/随机算例」三种规模下的所有算例，分别用 ALNS_PSO_UAV 和 Giant_Heuristic
各跑 N_RUNS 次，保存每种算法最优的一次结果（txt / png），
并将5次运行的统计汇总实时追加写入三个 CSV 文件（小/中/大规模各一个）。

运行方式（项目根目录）：
    ~/miniforge3/bin/python3 RunBenchmark.py

可调整参数（在文件顶部）：
    N_RUNS        每种算法每个算例跑几次（默认 5）
    NUM_THREADS   并行线程数（默认 4）
    MAX_ITER      每次搜索的最大迭代次数
"""

from __future__ import annotations

import csv
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------
# 可调参数
# --------------------------------------------------------------------------
N_RUNS        = 3
NUM_THREADS   = 3
MAX_ITER      = 500
PSO_FREQ      = 30
PSO_PARTICLES = 15
PSO_ITER      = 20
PUSH_FREQ     = 50
STAGNATION    = 100
GR_LS_FREQ    = 25
# 早停：连续多少次迭代全局最优无改进则提前终止（None=不启用，推荐值=250）
NO_IMPROVE_LIMIT = 100

# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------
ROOT        = os.path.dirname(os.path.abspath(__file__))
INST_ROOT   = os.path.join(ROOT, "算例", "随机算例")
RESULT_ROOT = os.path.join(ROOT, "结果", "随机算例")

# 三个规模对应的 CSV 输出路径
CSV_PATHS = {
    "0-Small":  os.path.join(ROOT, "结果", "随机算例求解结果_小规模.csv"),
    "1-Medium": os.path.join(ROOT, "结果", "随机算例求解结果_中等规模.csv"),
    "2-Large":  os.path.join(ROOT, "结果", "随机算例求解结果_大规模.csv"),
}

SCALES = [
    # ("0-Small",  "小规模"),
    ("1-Medium", "中等规模"),
    # ("2-Large",  "大规模"),
]

# ALNS / GR 的算子名（静态已知）
ALNS_D_NAMES = ["random", "worst", "route", "bp_split"]
ALNS_R_NAMES = ["greedy", "random", "bp_aware"]
GR_D_NAMES   = ["random", "worst", "segment"]
GR_R_NAMES   = ["greedy", "random", "regret"]


# --------------------------------------------------------------------------
# 延迟导入（避免 top-level 副作用）
# --------------------------------------------------------------------------
_alns_mod = None
_cpp_mod  = None   # Cplus/call_cpp.py 模块

def _get_alns_module():
    global _alns_mod
    if _alns_mod is None:
        import ALNS_PSO_UAV
        _alns_mod = ALNS_PSO_UAV
    return _alns_mod

def _get_cpp_module():
    """延迟导入 Cplus/call_cpp.py（动态插入 sys.path）。"""
    global _cpp_mod
    if _cpp_mod is None:
        cplus_dir = os.path.join(ROOT, "Cplus")
        if cplus_dir not in sys.path:
            sys.path.insert(0, cplus_dir)
        import call_cpp
        _cpp_mod = call_cpp
    return _cpp_mod

# 保留向后兼容（旧代码调用 _get_modules()）
def _get_modules():
    return _get_alns_module(), None


# --------------------------------------------------------------------------
# 辅助：从 Solution 提取使用无人机数 & 总飞行距离
# --------------------------------------------------------------------------
def _extract_solution_info(sol, inst, compute_route_raw_distance_fn):
    """返回 (num_used_drones: int, total_flight_dist: float)"""
    num_used   = 0
    total_dist = 0.0
    for route in sol.routes:
        if route.sub_edges:
            num_used += 1
            insp, trans = compute_route_raw_distance_fn(route, inst)
            total_dist += insp + trans
    return num_used, total_dist


# --------------------------------------------------------------------------
# C++ JSON 结果 -> GR txt 报告
# --------------------------------------------------------------------------
def _write_gr_txt(cpp_result: dict, txt_path: str) -> None:
    """
    根据 C++ 输出的 JSON 字典，生成与 Giant_Heuristic.py 格式一致的 txt 报告。

    C++ JSON 字段说明：
      per_thread_stats[i] = { "destroy": [{name,calls,time,impr_cur,impr_best},...],
                               "repair":  [...], "time_dr", "time_ls", "time_pso" }
      instance_params = { battery, speed, energy_cost, call_cost,
                          inspect_coef, transfer_coef }
      routes[i].sub_edges[j] = { edge_idx, seg, direction, ax,ay,bx,by,
                                  start_x,start_y, end_x,end_y, length }
    """
    total_cost   = cpp_result.get("total_cost", 0.0)
    solve_time   = cpp_result.get("solve_time", 0.0)
    cost_history = cpp_result.get("cost_history", [])
    routes       = cpp_result.get("routes", [])
    edges        = cpp_result.get("edges", [])
    params       = cpp_result.get("instance_params", {})
    battery      = params.get("battery", float('inf'))
    speed        = params.get("speed", 1.0)
    insp_coef    = params.get("inspect_coef", 1.5)
    tran_coef    = params.get("transfer_coef", 1.5)
    call_cost_p  = params.get("call_cost", 10.0)

    init_cost = cost_history[0] if cost_history else total_cost
    impr_pct  = (init_cost - total_cost) / init_cost * 100 if init_cost > 0 else 0.0

    # 构建 edge_idx -> (u, v) 映射，方便路径文字描述
    edge_uv: Dict[int, tuple] = {}
    for e in edges:
        edge_uv[e.get("edge_idx", -1)] = (e.get("u", "?"), e.get("v", "?"))

    def _se_label(se: dict) -> str:
        """
        根据 sub_edge 数据生成 '边(u,v)(起点端→终点端)' 格式的飞行段标签。

        seg=0: 整边(无断点), a端=u, b端=v
        seg=1: 前半段(u..bp), a端=u, b端=bp
        seg=2: 后半段(bp..v), a端=bp, b端=v
        direction=True  → 飞行方向 a→b
        direction=False → 飞行方向 b→a
        """
        ei  = se.get("edge_idx", -1)
        seg = se.get("seg", 0)       # 0=整边, 1=前半段(u->bp), 2=后半段(bp->v)
        d   = se.get("direction", True)  # True = a->b
        u, v = edge_uv.get(ei, ("?", "?"))

        if seg == 0:
            a_name, b_name = str(u), str(v)
        elif seg == 1:
            a_name, b_name = str(u), "bp"
        else:
            a_name, b_name = "bp", str(v)

        start_name = a_name if d else b_name
        end_name   = b_name if d else a_name
        return f"边({u},{v})({start_name}→{end_name})"

    lines = []
    lines.append("=" * 60)
    lines.append("Giant Route ALNS+PSO 求解结果")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"总费用: {total_cost:.6f} 元")
    lines.append(f"求解用时: {solve_time:.2f} 秒")
    lines.append(f"初始解费用: {init_cost:.6f} 元")
    lines.append(f"改进率: {impr_pct:.2f}%")
    lines.append("")

    # ---- 算子统计（每线程） ----
    per_thread = cpp_result.get("per_thread_stats", [])
    if per_thread:
        lines.append("=" * 80)
        lines.append("算子统计报告（按进程/线程区分）")
        lines.append("=" * 80)
        lines.append("")
        for ti, ts in enumerate(per_thread):
            lines.append(f"── 进程 {ti + 1} ──")
            time_dr  = ts.get("time_dr",  0.0)
            time_ls  = ts.get("time_ls",  0.0)
            time_pso = ts.get("time_pso", 0.0)
            lines.append("  模块耗时汇总:")
            lines.append(f"    破坏+修复 (DR)          : {time_dr:.3f} s")
            lines.append(f"    局部搜索/断点邻域 (LS)  : {time_ls:.3f} s")
            lines.append(f"    PSO 优化               : {time_pso:.3f} s")
            lines.append("")

            destroy = ts.get("destroy", [])
            if destroy:
                lines.append("  破坏算子统计:")
                lines.append(f"  {'算子名称':<14} {'调用次数':>8} {'总耗时(s)':>10} {'改进当前解':>10} {'改进最优解':>10}")
                lines.append("  " + "-" * 14 + " " + "-" * 8 + " " + "-" * 10 + " " + "-" * 10 + " " + "-" * 10)
                for d in destroy:
                    lines.append(f"  {d['name']:<14} {d['calls']:>8} {d['time']:>10.3f} {d['impr_cur']:>10} {d['impr_best']:>10}")
                lines.append("")

            repair = ts.get("repair", [])
            if repair:
                lines.append("  修复算子统计:")
                lines.append(f"  {'算子名称':<14} {'调用次数':>8} {'总耗时(s)':>10} {'改进当前解':>10} {'改进最优解':>10}")
                lines.append("  " + "-" * 14 + " " + "-" * 8 + " " + "-" * 10 + " " + "-" * 10 + " " + "-" * 10)
                for r in repair:
                    lines.append(f"  {r['name']:<14} {r['calls']:>8} {r['time']:>10.3f} {r['impr_cur']:>10} {r['impr_best']:>10}")
                lines.append("")

        lines.append("=" * 80)
        lines.append("")

    # ---- 断点配置 ----
    lines.append("--- 断点配置 ---")
    for e in edges:
        u, v   = e.get("u", "?"), e.get("v", "?")
        length = e.get("length", 0.0)
        bp     = e.get("breakpoint")
        if bp is not None:
            lam = bp.get("lambda", 0.0)
            bx  = bp.get("x", 0.0)
            by  = bp.get("y", 0.0)
            lines.append(f"  边({u},{v}) 长度={length:.2f}m: 断点位置 lambda={lam:.6f}, 坐标=({bx:.4f}, {by:.4f})")
        else:
            lines.append(f"  边({u},{v}) 长度={length:.2f}m: 无断点")
    lines.append("")

    # ---- 各无人机路径 ----
    lines.append("--- 各无人机路径 ---")
    for route in routes:
        di = route.get("drone_idx", 0)
        if not route.get("used", False):
            lines.append(f"\n无人机 {di + 1}: 未使用")
            continue
        cost       = route.get("cost", 0.0)
        energy     = route.get("energy", 0.0)
        feasible   = route.get("feasible", True)
        insp_dist  = route.get("inspect_dist", 0.0)
        trans_dist = route.get("transfer_dist", 0.0)
        total_dist = route.get("total_dist", insp_dist + trans_dist)
        sub_edges_r = route.get("sub_edges", [])

        # 各项费用：C++ 未单独输出，在此手动计算
        insp_cost = insp_dist * speed * insp_coef
        tran_cost = trans_dist * speed * tran_coef

        feasible_str = "[可行]" if feasible else "[超出电池!]"
        lines.append(f"\n无人机 {di + 1}:")
        lines.append(f"  总飞行距离: {total_dist:.4f}m (巡检: {insp_dist:.4f}m, 转移: {trans_dist:.4f}m)")
        lines.append(f"  能量消耗: {energy:.6f}  (电池容量: {battery:.4f}) {feasible_str}")
        lines.append(f"  路径费用: {cost:.6f} 元")
        lines.append("  路径（完整飞行顺序）:")
        depot_name = "基站"
        if sub_edges_r:
            # 构建飞行顺序：每段之间有"转移"，格式同 GR txt
            # 路径点序列：depot, se[0], se[1], ..., se[n-1], depot
            # 每行："当前位置 转移到 下一位置"
            prev = depot_name
            for se in sub_edges_r:
                lbl = _se_label(se)
                lines.append(f"    {prev} 转移到 {lbl}")
                prev = lbl
            lines.append(f"    {prev} 转移到 {depot_name}")
        lines.append("  费用明细:")
        lines.append(f"    调用成本: {call_cost_p:.4f} 元")
        lines.append(f"    巡检成本: {insp_dist:.4f}m × {speed:.1f} × {insp_coef:.1f} = {insp_cost:.6f} 元")
        lines.append(f"    转移成本: {trans_dist:.4f}m × {speed:.1f} × {tran_coef:.1f} = {tran_cost:.6f} 元")
    lines.append("")

    # ---- 收敛过程（每5个iter记录一次，步长5） ----
    lines.append("--- 收敛过程（每100次迭代）---")
    for i, c in enumerate(cost_history):
        lines.append(f"  Iter {i * 5:>5}: {c:.6f}")
    lines.append("")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------------
# C++ JSON stats -> Python dict 格式转换
# --------------------------------------------------------------------------
def _cpp_stats_to_thread_format(cpp_stats: dict) -> Optional[dict]:
    """
    将 C++ 输出 JSON 中的 stats 字段转换为与 Python ALNS 统计兼容的格式。

    C++ stats 格式：
      {
        "destroy": [{"name":..., "calls":N, "time":T, "impr_cur":X, "impr_best":Y}, ...],
        "repair":  [...],
        "time_dr": T, "time_ls": T, "time_pso": T
      }

    Python 格式（per_thread_stats 中的单个 dict）：
      {
        "destroy_names": [...], "repair_names": [...],
        "destroy_calls": [...], "destroy_time": [...],
        "destroy_impr_cur": [...], "destroy_impr_best": [...],
        "repair_calls": [...], ...
        "time_dr": T, "time_ls": T, "time_pso": T
      }
    """
    if not cpp_stats:
        return None
    destroy = cpp_stats.get("destroy", [])
    repair  = cpp_stats.get("repair",  [])
    return {
        "destroy_names":     [d["name"]      for d in destroy],
        "repair_names":      [r["name"]      for r in repair],
        "destroy_calls":     [d["calls"]     for d in destroy],
        "destroy_time":      [d["time"]      for d in destroy],
        "destroy_impr_cur":  [d["impr_cur"]  for d in destroy],
        "destroy_impr_best": [d["impr_best"] for d in destroy],
        "repair_calls":      [r["calls"]     for r in repair],
        "repair_time":       [r["time"]      for r in repair],
        "repair_impr_cur":   [r["impr_cur"]  for r in repair],
        "repair_impr_best":  [r["impr_best"] for r in repair],
        "time_dr":  cpp_stats.get("time_dr",  0.0),
        "time_ls":  cpp_stats.get("time_ls",  0.0),
        "time_pso": cpp_stats.get("time_pso", 0.0),
    }


# --------------------------------------------------------------------------
# 算子统计汇总（所有线程求和）
# --------------------------------------------------------------------------
def _aggregate_thread_stats(per_thread_stats: List[Optional[dict]]) -> Optional[dict]:
    """把多个线程的算子统计 **求和** 汇总为单个 dict。"""
    valid = [s for s in per_thread_stats if s is not None]
    if not valid:
        return None
    ref = valid[0]
    nd = len(ref['destroy_names'])
    nr = len(ref['repair_names'])
    agg: Dict[str, Any] = {
        'destroy_names':     ref['destroy_names'],
        'repair_names':      ref['repair_names'],
        'destroy_calls':     [0] * nd,
        'destroy_time':      [0.0] * nd,
        'destroy_impr_cur':  [0] * nd,
        'destroy_impr_best': [0] * nd,
        'repair_calls':      [0] * nr,
        'repair_time':       [0.0] * nr,
        'repair_impr_cur':   [0] * nr,
        'repair_impr_best':  [0] * nr,
        'time_dr':  0.0,
        'time_ls':  0.0,
        'time_pso': 0.0,
    }
    for s in valid:
        for i in range(nd):
            agg['destroy_calls'][i]     += s['destroy_calls'][i]
            agg['destroy_time'][i]      += s['destroy_time'][i]
            agg['destroy_impr_cur'][i]  += s['destroy_impr_cur'][i]
            agg['destroy_impr_best'][i] += s['destroy_impr_best'][i]
        for i in range(nr):
            agg['repair_calls'][i]     += s['repair_calls'][i]
            agg['repair_time'][i]      += s['repair_time'][i]
            agg['repair_impr_cur'][i]  += s['repair_impr_cur'][i]
            agg['repair_impr_best'][i] += s['repair_impr_best'][i]
        agg['time_dr']  += s['time_dr']
        agg['time_ls']  += s['time_ls']
        agg['time_pso'] += s['time_pso']
    return agg


# --------------------------------------------------------------------------
# 对 N_RUNS 次「已汇总线程」的算子统计再取平均
# --------------------------------------------------------------------------
def _avg_stats_over_runs(stats_list: List[Optional[dict]]) -> Optional[dict]:
    valid = [s for s in stats_list if s is not None]
    if not valid:
        return None
    n  = len(valid)
    ref = valid[0]
    nd = len(ref['destroy_names'])
    nr = len(ref['repair_names'])
    avg: Dict[str, Any] = {
        'destroy_names':     ref['destroy_names'],
        'repair_names':      ref['repair_names'],
        'destroy_calls':     [0.0] * nd,
        'destroy_time':      [0.0] * nd,
        'destroy_impr_cur':  [0.0] * nd,
        'destroy_impr_best': [0.0] * nd,
        'repair_calls':      [0.0] * nr,
        'repair_time':       [0.0] * nr,
        'repair_impr_cur':   [0.0] * nr,
        'repair_impr_best':  [0.0] * nr,
        'time_dr':  0.0,
        'time_ls':  0.0,
        'time_pso': 0.0,
    }
    for s in valid:
        for i in range(nd):
            avg['destroy_calls'][i]     += s['destroy_calls'][i]
            avg['destroy_time'][i]      += s['destroy_time'][i]
            avg['destroy_impr_cur'][i]  += s['destroy_impr_cur'][i]
            avg['destroy_impr_best'][i] += s['destroy_impr_best'][i]
        for i in range(nr):
            avg['repair_calls'][i]     += s['repair_calls'][i]
            avg['repair_time'][i]      += s['repair_time'][i]
            avg['repair_impr_cur'][i]  += s['repair_impr_cur'][i]
            avg['repair_impr_best'][i] += s['repair_impr_best'][i]
        avg['time_dr']  += s['time_dr']
        avg['time_ls']  += s['time_ls']
        avg['time_pso'] += s['time_pso']
    for i in range(nd):
        avg['destroy_calls'][i]     /= n
        avg['destroy_time'][i]      /= n
        avg['destroy_impr_cur'][i]  /= n
        avg['destroy_impr_best'][i] /= n
    for i in range(nr):
        avg['repair_calls'][i]     /= n
        avg['repair_time'][i]      /= n
        avg['repair_impr_cur'][i]  /= n
        avg['repair_impr_best'][i] /= n
    avg['time_dr']  /= n
    avg['time_ls']  /= n
    avg['time_pso'] /= n
    return avg


# --------------------------------------------------------------------------
# 胜出线程分布字符串
# --------------------------------------------------------------------------
def _thread_dist_string(ids: List[int]) -> str:
    """[0,1,0,2,0] -> 'T0:3次,T1:1次,T2:1次'"""
    if not ids:
        return ""
    counter: Dict[int, int] = defaultdict(int)
    for tid in ids:
        counter[tid] += 1
    return ",".join(f"T{t}:{c}次" for t, c in sorted(counter.items()))


# --------------------------------------------------------------------------
# 核心：运行单个算例 × 单种算法 × N_RUNS 次
# --------------------------------------------------------------------------
def run_instance_algo(
    inst_path: str,
    final_output_dir: str,
    algo: str,            # "ALNS" | "GR"
) -> Dict:
    """
    对 inst_path 指定的算例，用 algo 跑 N_RUNS 次。
    每次跑都写入独立的临时目录；全部完成后把最优那次的文件拷贝到
    final_output_dir。

    返回 dict：
      costs / times / num_used_list / dist_list /
      pool_pushes / pool_pulls / best_thread_ids / thread_dist_str /
      avg_op_stats
    """
    costs:           List[float] = []
    times:           List[float] = []
    num_used_list:   List[int]   = []
    dist_list:       List[float] = []
    pool_pushes:     List[int]   = []
    pool_pulls:      List[int]   = []
    best_thread_ids: List[int]   = []
    run_op_stats:    List[Optional[dict]] = []

    best_cost   = float('inf')
    best_run_dir: Optional[str] = None     # 最优那次的临时输出目录

    # 使用系统临时目录保存每次中间结果
    tmp_root = tempfile.mkdtemp(prefix=f"bench_{algo}_")

    try:
        for run_idx in range(N_RUNS):
            run_dir = os.path.join(tmp_root, f"run{run_idx}")
            os.makedirs(run_dir, exist_ok=True)

            print(f"    [{algo}] run {run_idx+1}/{N_RUNS} ...", end=" ", flush=True)
            t0 = time.time()
            try:
                if algo == "ALNS":
                    # ---- Python ALNS_PSO_UAV 求解器 ----
                    alns_mod = _get_alns_module()
                    sol, cost, per_thread_stats, pool_stats = alns_mod.solve_instance(
                        inst_path,
                        output_dir=run_dir,
                        max_iter=MAX_ITER,
                        pso_freq=PSO_FREQ,
                        pso_particles=PSO_PARTICLES,
                        pso_iter=PSO_ITER,
                        num_threads=NUM_THREADS,
                        push_freq=PUSH_FREQ,
                        stagnation_limit=STAGNATION,
                        no_improve_limit=NO_IMPROVE_LIMIT,
                        verbose=False,
                    )
                    elapsed = time.time() - t0
                    print(f"cost={cost:.4f}, time={elapsed:.1f}s")

                    # 解析飞行距离
                    inst_obj    = alns_mod.parse_instance(inst_path)
                    num_used, total_dist = _extract_solution_info(
                        sol, inst_obj, alns_mod.compute_route_raw_distance
                    )

                    costs.append(cost)
                    times.append(elapsed)
                    num_used_list.append(num_used)
                    dist_list.append(total_dist)
                    pool_pushes.append(pool_stats.get('total_pushes', 0))
                    pool_pulls.append(pool_stats.get('total_pulls', 0))
                    best_thread_ids.append(pool_stats.get('best_thread_id', -1))

                    agg = _aggregate_thread_stats(per_thread_stats)
                    run_op_stats.append(agg)

                    if cost < best_cost:
                        best_cost    = cost
                        best_run_dir = run_dir

                else:  # GR -> C++ 求解器
                    # ---- C++ Giant Route ALNS+PSO 求解器（支持多线程并行）----
                    cpp_mod    = _get_cpp_module()
                    cpp_result = cpp_mod.run_giant_heuristic(
                        inst_path,
                        max_iter=MAX_ITER,
                        pso_freq=PSO_FREQ,
                        pso_particles=PSO_PARTICLES,
                        pso_iter=PSO_ITER,
                        ls_freq=GR_LS_FREQ,
                        no_improve_limit=NO_IMPROVE_LIMIT,
                        num_threads=NUM_THREADS,
                        push_freq=PUSH_FREQ,
                        stagnation_limit=STAGNATION,
                        verbose=False,
                    )
                    elapsed = time.time() - t0
                    cost    = cpp_result["total_cost"]
                    print(f"cost={cost:.4f}, time={elapsed:.1f}s")

                    # 从 JSON routes 中提取无人机数和总飞行距离
                    num_used   = sum(1 for r in cpp_result["routes"] if r.get("used", False))
                    total_dist = sum(
                        r.get("total_dist", 0.0)
                        for r in cpp_result["routes"]
                        if r.get("used", False)
                    )

                    costs.append(cost)
                    times.append(elapsed)
                    num_used_list.append(num_used)
                    dist_list.append(total_dist)

                    # 读取 SharedPool 统计（多线程时由 C++ 填入，单线程为 0）
                    cpp_pool = cpp_result.get("pool_stats", {})
                    pool_pushes.append(cpp_pool.get("total_pushes", 0))
                    pool_pulls.append(cpp_pool.get("total_pulls", 0))
                    best_thread_ids.append(cpp_pool.get("best_thread_id", 0))

                    # 优先使用 per_thread_stats（每线程独立统计），汇总为一条 agg
                    cpp_pts = cpp_result.get("per_thread_stats", [])
                    if cpp_pts:
                        converted = [_cpp_stats_to_thread_format(s) for s in cpp_pts]
                        agg = _aggregate_thread_stats(converted)
                    else:
                        # 向后兼容：旧版单线程只有 "stats" 字段
                        agg = _cpp_stats_to_thread_format(cpp_result.get("stats", {}))
                    run_op_stats.append(agg)

                    # 保存可视化文件到 run_dir（无论是否最优）
                    if cost < best_cost:
                        best_cost    = cost
                        best_run_dir = run_dir

                    try:
                        inst_stem = os.path.splitext(os.path.basename(inst_path))[0]
                        sol_path  = os.path.join(run_dir, f"{inst_stem}-GR_solution.png")
                        conv_path = os.path.join(run_dir, f"{inst_stem}-GR_convergence.png")
                        txt_path  = os.path.join(run_dir, f"{inst_stem}-GR.txt")
                        cpp_mod.plot_solution(
                            cpp_result, sol_path,
                            title=f"{inst_stem} | Cost={cost:.4f} (C++ Giant Route)"
                        )
                        cpp_mod.plot_convergence(cpp_result["cost_history"], conv_path)
                        _write_gr_txt(cpp_result, txt_path)
                    except Exception as plot_err:
                        print(f"  [WARN] 绘图/txt/JSON处理失败: {plot_err}")

            except Exception as e:
                print(f"ERROR: {e}")
                traceback.print_exc()
                continue

        # 把最优那次的结果文件拷贝到正式目录
        if best_run_dir is not None:
            os.makedirs(final_output_dir, exist_ok=True)
            for fname in os.listdir(best_run_dir):
                src = os.path.join(best_run_dir, fname)
                dst = os.path.join(final_output_dir, fname)
                shutil.copy2(src, dst)
            print(f"    [{algo}] 最优结果已保存到: {final_output_dir}")
    finally:
        # 清理临时目录
        shutil.rmtree(tmp_root, ignore_errors=True)

    avg_op_stats = _avg_stats_over_runs(run_op_stats)
    thread_dist  = _thread_dist_string(best_thread_ids)

    return {
        'costs':           costs,
        'times':           times,
        'num_used_list':   num_used_list,
        'dist_list':       dist_list,
        'pool_pushes':     pool_pushes,
        'pool_pulls':      pool_pulls,
        'best_thread_ids': best_thread_ids,
        'thread_dist_str': thread_dist,
        'avg_op_stats':    avg_op_stats,
    }


# --------------------------------------------------------------------------
# 解析算例文件名
# --------------------------------------------------------------------------
def parse_instance_name(fname: str):
    """
    兼容两种命名格式：
      小/中规模: '9-10-5-1-(0).txt'          -> (nodes=9,  edges=10, drones=5, station=1, seed='0')
      大规模:   '26-30-10-1-(grid-0).txt'    -> (nodes=26, edges=30, drones=10, station=1, seed='grid-0')
    返回 (nodes, edges, drones, station, seed_str)，seed 统一为字符串。
    """
    name = os.path.splitext(fname)[0]
    # 通用格式：4段数字 + 括号内任意非空内容
    m = re.match(r'(\d+)-(\d+)-(\d+)-(\d+)-\(([^)]+)\)', name)
    if m:
        return (int(m.group(1)), int(m.group(2)),
                int(m.group(3)), int(m.group(4)), m.group(5))
    return None


# --------------------------------------------------------------------------
# 构建 CSV 表头（单行）
# --------------------------------------------------------------------------
def build_header() -> List[str]:
    """返回单行列名列表，用于 CSV 表头。"""
    header: List[str] = []

    # 基本信息（5列）
    header += ["算例名", "节点数", "边数", "无人机数", "基站编号"]

    def _algo_cols(prefix: str, d_names: List[str], r_names: List[str]) -> List[str]:
        cols = [
            f"{prefix}_目标函数_最小值",
            f"{prefix}_目标函数_最大值",
            f"{prefix}_目标函数_均值",
            f"{prefix}_求解时间_最小(s)",
            f"{prefix}_求解时间_最大(s)",
            f"{prefix}_求解时间_均值(s)",
            f"{prefix}_实际使用无人机数_均值",
            f"{prefix}_总飞行距离_均值(m)",
            f"{prefix}_最优线程ID分布",
            f"{prefix}_公共池拉取次数_均值",
            f"{prefix}_推入公共池次数_均值",
        ]
        for dn in d_names:
            cols += [
                f"{prefix}_破坏_{dn}_调用次数_均值",
                f"{prefix}_破坏_{dn}_耗时_均值(s)",
                f"{prefix}_破坏_{dn}_改进当前解_均值",
                f"{prefix}_破坏_{dn}_改进最优解_均值",
            ]
        for rn in r_names:
            cols += [
                f"{prefix}_修复_{rn}_调用次数_均值",
                f"{prefix}_修复_{rn}_耗时_均值(s)",
                f"{prefix}_修复_{rn}_改进当前解_均值",
                f"{prefix}_修复_{rn}_改进最优解_均值",
            ]
        cols += [
            f"{prefix}_模块耗时_DR均值(s)",
            f"{prefix}_模块耗时_LS均值(s)",
            f"{prefix}_模块耗时_PSO均值(s)",
        ]
        return cols

    header += _algo_cols("ALNS_PSO",   ALNS_D_NAMES, ALNS_R_NAMES)
    header += _algo_cols("GiantRoute", GR_D_NAMES,   GR_R_NAMES)
    return header


# --------------------------------------------------------------------------
# 构建单行数据
# --------------------------------------------------------------------------
def _fill_algo_cols(result: Optional[Dict],
                    d_names: List[str], r_names: List[str]) -> List:
    n_op_cols = len(d_names) * 4 + len(r_names) * 4 + 3
    n_total   = 11 + n_op_cols    # 3+3+1+1+3=11 前置列

    if result is None or not result['costs']:
        return [None] * n_total

    vals: List = []
    costs = result['costs']
    times = result['times']
    n = len(costs)

    vals += [round(min(costs), 4), round(max(costs), 4), round(sum(costs)/n, 4)]
    vals += [round(min(times),  2), round(max(times),  2), round(sum(times)/n,  2)]
    nu = result['num_used_list']
    vals.append(round(sum(nu)/len(nu), 2) if nu else None)
    dl = result['dist_list']
    vals.append(round(sum(dl)/len(dl), 2) if dl else None)
    vals.append(result['thread_dist_str'])
    pp = result['pool_pulls']
    pu = result['pool_pushes']
    vals.append(round(sum(pp)/len(pp), 2) if pp else 0)
    vals.append(round(sum(pu)/len(pu), 2) if pu else 0)

    op = result['avg_op_stats']
    if op is not None:
        for i in range(len(d_names)):
            vals += [
                round(op['destroy_calls'][i],     2),
                round(op['destroy_time'][i],       4),
                round(op['destroy_impr_cur'][i],   2),
                round(op['destroy_impr_best'][i],  2),
            ]
        for i in range(len(r_names)):
            vals += [
                round(op['repair_calls'][i],     2),
                round(op['repair_time'][i],      4),
                round(op['repair_impr_cur'][i],  2),
                round(op['repair_impr_best'][i], 2),
            ]
        vals += [
            round(op['time_dr'],  4),
            round(op['time_ls'],  4),
            round(op['time_pso'], 4),
        ]
    else:
        vals += [None] * n_op_cols
    return vals


def build_row(
    inst_name: str,
    nodes: int, edges: int, drones: int, station: int,
    alns_result: Optional[Dict],
    gr_result:   Optional[Dict],
) -> List:
    row = [inst_name, nodes, edges, drones, station]
    row += _fill_algo_cols(alns_result, ALNS_D_NAMES, ALNS_R_NAMES)
    row += _fill_algo_cols(gr_result,   GR_D_NAMES,   GR_R_NAMES)
    return row


# --------------------------------------------------------------------------
# CSV 初始化：若文件不存在（或为空）则写入表头，否则追加模式续写
# --------------------------------------------------------------------------
def init_csv(scale_dir: str, header: List[str]):
    """
    初始化对应规模的 CSV 文件。
    - 文件不存在或内容为空时，创建并写入表头行。
    - 文件已存在且非空时，跳过（支持断点续跑追加）。
    """
    csv_path = CSV_PATHS[scale_dir]
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(header)
        print(f"CSV 已初始化（写入表头）: {csv_path}")
    else:
        print(f"CSV 已存在，追加模式续写: {csv_path}")


# --------------------------------------------------------------------------
# 追加单行数据到 CSV（即时落盘）
# --------------------------------------------------------------------------
def append_csv_row(scale_dir: str, row: List):
    """将一行结果立即追加写入对应规模的 CSV 文件。"""
    csv_path = CSV_PATHS[scale_dir]
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow(row)
    print(f"  [CSV] 已写入: {os.path.basename(csv_path)}", flush=True)


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main():
    header = build_header()

    # 初始化三个 CSV 文件（不存在则创建并写表头，已存在则追加续写）
    for scale_dir, _ in SCALES:
        init_csv(scale_dir, header)

    for scale_dir, scale_name in SCALES:
        inst_dir    = os.path.join(INST_ROOT, scale_dir)
        result_base = os.path.join(RESULT_ROOT, scale_dir)

        if not os.path.isdir(inst_dir):
            print(f"[SKIP] 目录不存在: {inst_dir}")
            continue

        all_files_raw = sorted(f for f in os.listdir(inst_dir) if f.endswith('.txt'))

        # 大规模只运行节点数 >= 46 的算例（46-70 及以上规模）
        if scale_dir == "2-Large":
            all_files = [
                f for f in all_files_raw
                if (parsed_tmp := parse_instance_name(f)) is not None
                and parsed_tmp[0] >= 46
            ]
        else:
            all_files = all_files_raw

        print(f"\n{'='*60}")
        print(f"规模: {scale_name} | 算例数: {len(all_files)}"
              + (f"（共 {len(all_files_raw)} 个，已筛选 nodes≥46）"
                 if scale_dir == "2-Large" else ""))
        print(f"{'='*60}")

        for fname in all_files:
            inst_path = os.path.join(inst_dir, fname)
            parsed    = parse_instance_name(fname)
            if parsed is None:
                print(f"  [WARN] 无法解析: {fname}")
                continue
            nodes, edges, drones, station, seed = parsed
            inst_name  = os.path.splitext(fname)[0]
            out_dir    = os.path.join(result_base, inst_name)

            print(f"\n  {inst_name}  nodes={nodes} edges={edges} "
                  f"drones={drones} station={station} seed={seed}")

            # ---- ALNS ----
            print(f"  -- ALNS_PSO ({N_RUNS} runs) --")
            try:
                alns_result = run_instance_algo(inst_path, out_dir, "ALNS")
            except Exception as e:
                print(f"  [ERROR] ALNS: {e}")
                traceback.print_exc()
                alns_result = None

            # ---- Giant Route ----
            print(f"  -- GiantRoute ({N_RUNS} runs) --")
            try:
                gr_result = run_instance_algo(inst_path, out_dir, "GR")
            except Exception as e:
                print(f"  [ERROR] GR: {e}")
                traceback.print_exc()
                gr_result = None

            # ---- 立即追加写入 CSV（出一行结果写一行，不等全部完成）----
            row = build_row(inst_name, nodes, edges, drones, station,
                            alns_result, gr_result)
            append_csv_row(scale_dir, row)
        # break


if __name__ == "__main__":
    main()

