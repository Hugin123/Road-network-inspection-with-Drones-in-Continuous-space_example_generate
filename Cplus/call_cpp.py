"""
call_cpp.py
===========
Python 调用 C++ 可执行文件（giant_heuristic）的接口脚本。
跨平台：自动判断可执行文件路径（无需硬编码 .exe 后缀）。

用法示例：
    python call_cpp.py <instance.txt> [选项]
    python call_cpp.py --help

或作为模块使用：
    from call_cpp import run_giant_heuristic, plot_solution, plot_convergence
    result = run_giant_heuristic("path/to/instance.txt")
    plot_solution(result, output_path="solution.png")
    plot_convergence(result["cost_history"], output_path="convergence.png")
"""

import os
import sys
import json
import subprocess
import platform
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ============================================================
# 1. 工具函数：定位可执行文件
# ============================================================

def find_executable(build_dir: str = None) -> str:
    """
    自动寻找编译好的 giant_heuristic 可执行文件。
    搜索顺序：
      1. build_dir 参数指定的目录
      2. 默认 build 目录（同脚本目录/build）
      3. cmake --install 后的系统路径（PATH）
    """
    script_dir = Path(__file__).parent.resolve()
    exe_name = "giant_heuristic"
    if platform.system() == "Windows":
        exe_name += ".exe"

    # 候选路径
    candidates = []
    if build_dir:
        candidates.append(Path(build_dir) / exe_name)
    # cmake 默认构建目录
    for build_subdir in ["build", "build/Release", "build/Debug", "cmake-build-release", "cmake-build-debug"]:
        candidates.append(script_dir / build_subdir / exe_name)

    for path in candidates:
        if path.exists():
            return str(path)

    # 尝试 PATH
    import shutil
    found = shutil.which(exe_name)
    if found:
        return found

    raise FileNotFoundError(
        f"找不到可执行文件 '{exe_name}'。\n"
        f"请先编译项目（在 Cplus 目录执行）：\n"
        f"  mkdir build && cd build && cmake .. && cmake --build . --config Release"
    )


# ============================================================
# 2. 核心调用函数
# ============================================================

def run_giant_heuristic(
    instance_path: str,
    build_dir: str = None,
    max_iter: int = 500,
    pso_freq: int = 50,
    pso_particles: int = 20,
    pso_iter: int = 30,
    ls_freq: int = 25,
    no_improve_limit: int = None,
    # 并行参数（对应 Giant_Heuristic.py::parallel_solve 的同名参数）
    num_threads: int = 1,
    push_freq: int = 50,
    stagnation_limit: int = 100,
    verbose: bool = False,
    timeout: float = None,
) -> dict:
    """
    调用 C++ 可执行文件求解算例，返回解析后的 JSON 字典。

    参数
    ------
    instance_path   : 算例文件路径（.txt）
    build_dir       : C++ build 目录（None 则自动查找）
    max_iter        : ALNS 最大迭代次数
    pso_freq        : PSO 触发频率
    pso_particles   : PSO 粒子数
    pso_iter        : PSO 迭代次数
    ls_freq         : 局部搜索频率
    no_improve_limit: 早停阈值（None=不启用）
    num_threads     : 并行线程数（>1 启用异步并行模式）
    push_freq       : 向 SharedPool 推送解的频率（每 N 次迭代）
    stagnation_limit: 触发从 SharedPool 拉取解的连续未改进阈值
    verbose         : 是否将 C++ stderr 进度输出到 Python stderr
    timeout         : 超时秒数（None=不限制）

    返回
    ------
    dict : C++ 输出的 JSON 字典，包含：
        - total_cost         总费用
        - solve_time         求解时间（秒）
        - num_drones         无人机数量
        - num_edges          需求边数量
        - num_nodes          节点数量
        - instance_params    算例参数
        - nodes              节点坐标列表
        - depot              基站信息
        - edges              边列表（含断点信息）
        - routes             各无人机路径（含飞行轨迹）
        - cost_history       合并后的收敛历史（每步取各线程最小值）
        - per_thread_stats   每线程独立算子统计列表（与 Python per_thread_stats 对应）
        - stats              第一个线程的统计（向后兼容）
        - pool_stats         SharedPool 统计（total_pushes/pulls/best_thread_id）
    """
    exe_path = find_executable(build_dir)
    instance_path = str(Path(instance_path).resolve())

    cmd = [exe_path, instance_path,
           "--max_iter", str(max_iter),
           "--pso_freq", str(pso_freq),
           "--pso_particles", str(pso_particles),
           "--pso_iter", str(pso_iter),
           "--ls_freq", str(ls_freq),
           "--num_threads", str(num_threads),
           "--push_freq", str(push_freq),
           "--stagnation_limit", str(stagnation_limit)]

    if no_improve_limit is not None:
        cmd += ["--no_improve_limit", str(no_improve_limit)]
    if verbose:
        cmd.append("--verbose")

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout
    )

    # 若 verbose，打印 stderr（C++ 进度信息）
    if verbose and proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")

    if proc.returncode != 0 and not proc.stdout.strip():
        raise RuntimeError(
            f"C++ 进程返回码 {proc.returncode}\nstderr: {proc.stderr}"
        )

    # 解析 JSON
    try:
        result = json.loads(proc.stdout.strip())
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"无法解析 C++ 输出的 JSON：{e}\n输出内容（前500字符）：{proc.stdout[:500]}"
        )

    if "error" in result:
        raise RuntimeError(f"C++ 求解错误：{result['error']}")

    return result


# ============================================================
# 3. 绘图函数
# ============================================================

# 颜色列表（与 Giant_Heuristic.py 一致）
ROUTE_COLORS = [
    '#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6',
    '#1ABC9C', '#E67E22', '#34495E', '#E91E63', '#00BCD4',
    '#FF5722', '#607D8B', '#795548', '#CDDC39', '#FF9800',
]


def plot_solution(
    result: dict,
    output_path: str = "solution.png",
    title: str = None,
    figsize: tuple = (12, 10),
    dpi: int = 150,
) -> None:
    """
    绘制解的路径可视化图（仿照 Giant_Heuristic.py 的 plot_solution）。

    参数
    ------
    result      : run_giant_heuristic 返回的字典
    output_path : 输出图片路径
    title       : 图标题（None 则自动生成）
    figsize     : 图形大小
    dpi         : 分辨率
    """
    fig, ax = plt.subplots(figsize=figsize)

    nodes = result["nodes"]
    edges = result["edges"]
    routes = result["routes"]
    depot = result["depot"]
    total_cost = result["total_cost"]
    num_edges = result["num_edges"]

    # 节点坐标数组
    x_arr = np.array([n["x"] for n in nodes])
    y_arr = np.array([n["y"] for n in nodes])

    # ---- 画路网背景边（灰色细线）----
    for edge in edges:
        ax.plot([edge["ux"], edge["vx"]], [edge["uy"], edge["vy"]],
                color='#CCCCCC', linewidth=1.0, zorder=1, alpha=0.6)

    # ---- 画各无人机路径 ----
    legend_handles = []
    used_drone_count = 0

    for route_data in routes:
        if not route_data["used"]:
            continue
        di = route_data["drone_idx"]
        color = ROUTE_COLORS[di % len(ROUTE_COLORS)]
        traj = route_data["trajectory"]
        sub_edges_data = route_data["sub_edges"]

        # 分离巡检段坐标（实线）和转移段坐标（虚线）
        depot_x = depot["x"]
        depot_y = depot["y"]

        # 构建轨迹：depot -> (transfer虚线 -> inspect实线) * N -> depot
        # trajectory 格式：depot, inspect_start, inspect_end, ..., depot_return
        cx, cy = depot_x, depot_y

        inspect_xs = []
        inspect_ys = []
        transfer_lines = []  # 每段 [(x1,y1),(x2,y2)]

        for se_data in sub_edges_data:
            sx, sy = se_data["start_x"], se_data["start_y"]
            ex, ey = se_data["end_x"],   se_data["end_y"]
            # 转移段
            transfer_lines.append([(cx, cy), (sx, sy)])
            # 巡检段
            inspect_xs += [sx, ex, None]
            inspect_ys += [sy, ey, None]
            cx, cy = ex, ey

        # 最后返回 depot
        transfer_lines.append([(cx, cy), (depot_x, depot_y)])

        # 画转移段（虚线）
        for (tx1, ty1), (tx2, ty2) in transfer_lines:
            if abs(tx1 - tx2) > 1e-12 or abs(ty1 - ty2) > 1e-12:
                ax.plot([tx1, tx2], [ty1, ty2],
                        color=color, linestyle='--', linewidth=1.0,
                        alpha=0.5, zorder=3)

        # 画巡检段（实线，加粗箭头）
        for se_data in sub_edges_data:
            sx, sy = se_data["start_x"], se_data["start_y"]
            ex, ey = se_data["end_x"],   se_data["end_y"]
            ax.annotate(
                '', xy=(ex, ey), xytext=(sx, sy),
                arrowprops=dict(
                    arrowstyle='->', color=color,
                    lw=2.0, mutation_scale=12
                ), zorder=4)

        cost = route_data["cost"]
        energy = route_data["energy"]
        feasible = route_data["feasible"]
        label = (f"Drone {di+1}: cost={cost:.2f}, "
                 f"E={energy:.3f}{'✓' if feasible else '✗'}")
        handle = mpatches.Patch(color=color, label=label)
        legend_handles.append(handle)
        used_drone_count += 1

    # ---- 画断点 ----
    for edge in edges:
        if edge["breakpoint"] is not None:
            bpx = edge["breakpoint"]["x"]
            bpy = edge["breakpoint"]["y"]
            ax.scatter(bpx, bpy, marker='*', s=200, color='#FF6B6B',
                       zorder=6, edgecolors='#C0392B', linewidths=0.8)

    # ---- 画路网节点 ----
    for node in nodes:
        ni = node["id"]
        if ni == depot["idx"]:
            continue
        ax.scatter(node["x"], node["y"], marker='o', s=80, color='#5B9BD5',
                   zorder=5, edgecolors='white', linewidths=0.8)
        ax.annotate(str(ni), (node["x"], node["y"]),
                    textcoords='offset points', xytext=(4, 4),
                    fontsize=7, color='#2C3E50')

    # ---- 画 depot ----
    ax.scatter(depot["x"], depot["y"], marker='s', s=300, color='#E74C3C',
               zorder=7, edgecolors='white', linewidths=1.5)
    ax.annotate('DEPOT', (depot["x"], depot["y"]),
                textcoords='offset points', xytext=(5, 5),
                fontsize=9, color='#E74C3C', fontweight='bold')

    # 断点图例
    has_bp = any(e["breakpoint"] is not None for e in edges)
    if has_bp:
        legend_handles.append(
            mpatches.Patch(color='#FF6B6B', label='Breakpoint'))

    ax.legend(handles=legend_handles, loc='best', fontsize=8, framealpha=0.8)
    ax.set_aspect('equal', adjustable='datalim')
    ax.margins(0.15)

    if title is None:
        title = (f"Giant Route C++ Solver | "
                 f"Cost={total_cost:.2f} | "
                 f"Drones used={used_drone_count}")
    ax.set_title(title, fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.2)
    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"路径可视化图已保存: {output_path}")


def plot_convergence(
    cost_history: list,
    output_path: str = "convergence.png",
    figsize: tuple = (10, 5),
    dpi: int = 150,
) -> None:
    """
    绘制收敛曲线。

    参数
    ------
    cost_history : 费用历史列表
    output_path  : 输出图片路径
    figsize      : 图形大小
    dpi          : 分辨率
    """
    fig, ax = plt.subplots(figsize=figsize)
    iters = list(range(len(cost_history)))
    ax.plot(iters, cost_history, color='#2196F3', linewidth=1.2, label='Best Cost')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Cost', fontsize=11)
    ax.set_title('Convergence Curve (C++ Giant Route ALNS+PSO)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)
    if cost_history:
        ax.set_ylim(bottom=min(cost_history) * 0.95)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"收敛曲线已保存: {output_path}")


def plot_operator_stats(
    result: dict,
    output_path: str = "operator_stats.png",
    figsize: tuple = (12, 5),
    dpi: int = 150,
) -> None:
    """
    绘制算子使用统计条形图。
    """
    stats = result.get("stats", {})
    destroy_stats = stats.get("destroy", [])
    repair_stats  = stats.get("repair", [])

    if not destroy_stats and not repair_stats:
        return

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    def draw_bar(ax_, data, title):
        if not data:
            return
        names  = [d["name"] for d in data]
        calls  = [d["calls"] for d in data]
        i_best = [d["impr_best"] for d in data]
        x = np.arange(len(names))
        width = 0.35
        ax_.bar(x - width/2, calls,  width, label='Calls',      color='#42A5F5')
        ax_.bar(x + width/2, i_best, width, label='Impr Best',  color='#EF5350')
        ax_.set_xticks(x)
        ax_.set_xticklabels(names, fontsize=10)
        ax_.set_title(title, fontsize=11)
        ax_.legend(fontsize=9)
        ax_.grid(True, axis='y', linestyle='--', alpha=0.3)

    draw_bar(axes[0], destroy_stats, 'Destroy Operator Stats')
    draw_bar(axes[1], repair_stats,  'Repair Operator Stats')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"算子统计图已保存: {output_path}")


def print_summary(result: dict) -> None:
    """打印求解结果摘要。"""
    print("=" * 60)
    print("C++ Giant Route ALNS+PSO 求解结果")
    print("=" * 60)
    print(f"总费用     : {result['total_cost']:.6f}")
    print(f"求解时间   : {result['solve_time']:.2f} 秒")
    print(f"无人机数量 : {result['num_drones']}")
    print(f"需求边数量 : {result['num_edges']}")
    ch = result.get("cost_history", [])
    if len(ch) > 1:
        impr = (ch[0] - ch[-1]) / ch[0] * 100 if ch[0] > 0 else 0
        print(f"初始费用   : {ch[0]:.6f}")
        print(f"改进率     : {impr:.2f}%")
    print("\n各无人机路径：")
    params = result.get("instance_params", {})
    battery = params.get("battery", float('inf'))
    for route in result["routes"]:
        di = route["drone_idx"]
        if not route["used"]:
            print(f"  无人机 {di+1}: 未使用")
            continue
        feasible = "✓ 可行" if route["feasible"] else "✗ 超出电池!"
        print(f"  无人机 {di+1}: "
              f"cost={route['cost']:.4f}, "
              f"energy={route['energy']:.4f}/{battery:.4f} {feasible}, "
              f"子边数={len(route['sub_edges'])}")
    print("=" * 60)


# ============================================================
# 4. 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="调用 C++ Giant Route ALNS+PSO 求解器并绘图"
    )
    parser.add_argument("instance", help="算例文件路径（.txt）")
    parser.add_argument("--build_dir", "-b", default=None,
                        help="C++ build 目录（默认自动查找）")
    parser.add_argument("--output_dir", "-o", default=None,
                        help="输出目录（默认与算例同目录）")
    parser.add_argument("--max_iter", "-n", type=int, default=500)
    parser.add_argument("--pso_freq", type=int, default=50)
    parser.add_argument("--pso_particles", type=int, default=20)
    parser.add_argument("--pso_iter", type=int, default=30)
    parser.add_argument("--ls_freq", type=int, default=25)
    parser.add_argument("--no_improve_limit", type=int, default=None)
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="显示 C++ 求解进度")
    parser.add_argument("--timeout", type=float, default=None,
                        help="超时秒数")
    parser.add_argument("--save_json", action="store_true",
                        help="同时保存 JSON 结果文件")
    args = parser.parse_args()

    # 确定输出目录
    instance_path = Path(args.instance).resolve()
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = instance_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    basename = instance_path.stem

    print(f"正在求解: {instance_path}")
    print(f"输出目录: {output_dir}")

    # 调用 C++
    result = run_giant_heuristic(
        str(instance_path),
        build_dir=args.build_dir,
        max_iter=args.max_iter,
        pso_freq=args.pso_freq,
        pso_particles=args.pso_particles,
        pso_iter=args.pso_iter,
        ls_freq=args.ls_freq,
        no_improve_limit=args.no_improve_limit,
        verbose=args.verbose,
        timeout=args.timeout,
    )

    # 打印摘要
    print_summary(result)

    # 绘图
    sol_path  = str(output_dir / f"{basename}-CPP_solution.png")
    conv_path = str(output_dir / f"{basename}-CPP_convergence.png")
    ops_path  = str(output_dir / f"{basename}-CPP_operator_stats.png")

    plot_solution(result, sol_path,
                  title=f"{basename} | Cost={result['total_cost']:.2f} (C++ Giant Route)")
    plot_convergence(result["cost_history"], conv_path)
    plot_operator_stats(result, ops_path)

    # 可选：保存 JSON
    if args.save_json:
        json_path = str(output_dir / f"{basename}-CPP_result.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"JSON 结果已保存: {json_path}")


if __name__ == "__main__":
    main()
