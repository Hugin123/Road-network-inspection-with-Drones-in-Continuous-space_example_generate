#include "types.h"
#include "functions.h"
#include "json.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <chrono>

using json = nlohmann::json;

// 获取当前秒数
static double get_time_sec() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

// ============================================================
// 将 SolverStats 序列化为 JSON 对象
// ============================================================
static json stats_to_json(const SolverStats& stats) {
    json stats_json;
    json destroy_stats, repair_stats;
    for (int i = 0; i < (int)stats.destroy_names.size(); ++i) {
        json s;
        s["name"]      = stats.destroy_names[i];
        s["calls"]     = stats.destroy_calls[i];
        s["time"]      = stats.destroy_time[i];
        s["impr_cur"]  = stats.destroy_impr_cur[i];
        s["impr_best"] = stats.destroy_impr_best[i];
        destroy_stats.push_back(s);
    }
    for (int i = 0; i < (int)stats.repair_names.size(); ++i) {
        json s;
        s["name"]      = stats.repair_names[i];
        s["calls"]     = stats.repair_calls[i];
        s["time"]      = stats.repair_time[i];
        s["impr_cur"]  = stats.repair_impr_cur[i];
        s["impr_best"] = stats.repair_impr_best[i];
        repair_stats.push_back(s);
    }
    stats_json["destroy"]  = destroy_stats;
    stats_json["repair"]   = repair_stats;
    stats_json["time_dr"]  = stats.time_dr;
    stats_json["time_ls"]  = stats.time_ls;
    stats_json["time_pso"] = stats.time_pso;
    return stats_json;
}

// ============================================================
// JSON 输出构建
// ============================================================

json build_output_json(
    const Solution& sol,
    const Instance& inst,
    double best_cost,
    double solve_time,
    const std::vector<double>& cost_history,
    // 单线程时 per_thread_stats 只含一个元素
    const std::vector<SolverStats>& per_thread_stats,
    // 并行池统计（单线程时填 0）
    int pool_total_pushes    = 0,
    int pool_total_pulls     = 0,
    int pool_best_thread_id  = 0)
{
    auto [depot_x, depot_y] = inst.node_coord(inst.depot_idx);
    json out;

    // ---- 基本信息 ----
    out["total_cost"]      = best_cost;
    out["solve_time"]      = solve_time;
    out["num_drones"]      = inst.num_drones;          // 算例声明的无人机数
    out["num_drones_used"] = (int)sol.routes.size();   // 实际使用的无人机数（不限制时可超过声明值）
    out["num_edges"]       = inst.num_edges;
    out["num_nodes"]       = inst.total_nodes;

    // ---- 算例参数 ----
    json params;
    params["battery"]       = inst.battery;
    params["speed"]         = inst.speed;
    params["energy_cost"]   = inst.energy_cost;
    params["call_cost"]     = inst.call_cost;
    params["inspect_coef"]  = inst.inspect_coef;
    params["transfer_coef"] = inst.transfer_coef;
    out["instance_params"] = params;

    // ---- 节点坐标（供绘图用）----
    json nodes;
    for (int i = 0; i < inst.total_nodes; ++i) {
        json node;
        node["id"] = i;
        node["x"]  = inst.x[i];
        node["y"]  = inst.y[i];
        nodes.push_back(node);
    }
    out["nodes"] = nodes;

    // ---- depot ----
    out["depot"] = {{"idx", inst.depot_idx},
                    {"x", depot_x},
                    {"y", depot_y}};

    // ---- 原始边列表（供绘图：画路网背景）----
    json edges_list;
    for (int ei = 0; ei < inst.num_edges; ++ei) {
        auto [u, v] = inst.edges[ei];
        json edge;
        edge["edge_idx"] = ei;
        edge["u"] = u; edge["v"] = v;
        edge["ux"] = inst.x[u]; edge["uy"] = inst.y[u];
        edge["vx"] = inst.x[v]; edge["vy"] = inst.y[v];
        edge["length"] = inst.edge_length(ei);
        // 断点信息
        if (sol.breakpoints[ei].has_value()) {
            double lam = *sol.breakpoints[ei];
            auto [bpx, bpy] = inst.point_on_edge(ei, lam);
            edge["breakpoint"] = {{"lambda", lam}, {"x", bpx}, {"y", bpy}};
        } else {
            edge["breakpoint"] = nullptr;
        }
        edges_list.push_back(edge);
    }
    out["edges"] = edges_list;

    // ---- 各无人机路径（完整飞行数据，供绘图）----
    // 遍历实际路径数（不限制时可能超过 inst.num_drones）
    json routes_list;
    for (int di = 0; di < (int)sol.routes.size(); ++di) {
        const auto& route = sol.routes[di];
        json route_json;
        route_json["drone_idx"] = di;

        if (route.empty()) {
            route_json["used"] = false;
            routes_list.push_back(route_json);
            continue;
        }
        route_json["used"] = true;

        auto [inspect_dist, transfer_dist] = compute_route_raw_distance(route, inst);
        double energy = compute_route_distance(route, inst);
        double cost   = compute_route_cost(route, inst);

        route_json["inspect_dist"]  = inspect_dist;
        route_json["transfer_dist"] = transfer_dist;
        route_json["total_dist"]    = inspect_dist + transfer_dist;
        route_json["energy"]        = energy;
        route_json["cost"]          = cost;
        route_json["feasible"]      = (energy <= inst.battery);

        // 飞行轨迹点序列（depot → 子边 → depot），用于绘图折线
        json trajectory;
        trajectory.push_back({{"x", depot_x}, {"y", depot_y},
                               {"label", "depot"}, {"type", "depot"}});

        for (int i = 0; i < (int)route.sub_edges.size(); ++i) {
            const auto& se = route.sub_edges[i];
            bool dir = route.directions[i];
            auto [sx, sy] = route.start_point(i);
            auto [ex, ey] = route.end_point(i);

            auto [u, v] = inst.edges[se.origin_edge_idx];
            std::string seg_label = "edge(" + std::to_string(u)
                                    + "," + std::to_string(v) + ")";

            json inspect_start, inspect_end;
            inspect_start["x"] = sx;
            inspect_start["y"] = sy;
            inspect_start["type"] = "inspect_start";
            inspect_start["label"] = seg_label;
            inspect_start["edge_idx"] = se.origin_edge_idx;
            inspect_start["seg"] = se.seg;
            inspect_start["direction"] = dir;

            inspect_end["x"] = ex;
            inspect_end["y"] = ey;
            inspect_end["type"] = "inspect_end";
            inspect_end["label"] = seg_label;
            inspect_end["edge_idx"] = se.origin_edge_idx;
            inspect_end["seg"] = se.seg;
            inspect_end["direction"] = dir;

            trajectory.push_back(inspect_start);
            trajectory.push_back(inspect_end);
        }
        trajectory.push_back({{"x", depot_x}, {"y", depot_y},
                               {"label", "depot"}, {"type", "depot_return"}});

        route_json["trajectory"] = trajectory;

        // 子边列表（原始数据）
        json sub_edges_json;
        for (int i = 0; i < (int)route.sub_edges.size(); ++i) {
            const auto& se = route.sub_edges[i];
            bool dir = route.directions[i];
            json se_json;
            se_json["edge_idx"] = se.origin_edge_idx;
            se_json["seg"] = se.seg;
            se_json["ax"] = se.ax; se_json["ay"] = se.ay;
            se_json["bx"] = se.bx; se_json["by"] = se.by;
            se_json["length"] = se.length();
            se_json["direction"] = dir;
            auto [sx, sy] = route.start_point(i);
            auto [ex, ey] = route.end_point(i);
            se_json["start_x"] = sx; se_json["start_y"] = sy;
            se_json["end_x"] = ex;   se_json["end_y"] = ey;
            sub_edges_json.push_back(se_json);
        }
        route_json["sub_edges"] = sub_edges_json;

        routes_list.push_back(route_json);
    }
    out["routes"] = routes_list;

    // ---- 收敛历史 ----
    out["cost_history"] = cost_history;

    // ---- 算子统计（每线程独立输出）----
    // per_thread_stats 是数组，与 Python per_thread_stats 完全对应
    json stats_array;
    for (const auto& s : per_thread_stats) {
        stats_array.push_back(stats_to_json(s));
    }
    out["per_thread_stats"] = stats_array;

    // 向后兼容：保留 "stats" 字段（取第一个线程，或单线程时的统计）
    if (!per_thread_stats.empty()) {
        out["stats"] = stats_to_json(per_thread_stats[0]);
    }

    // ---- 并行池统计 ----
    json pool_stats_json;
    pool_stats_json["total_pushes"]    = pool_total_pushes;
    pool_stats_json["total_pulls"]     = pool_total_pulls;
    pool_stats_json["best_thread_id"]  = pool_best_thread_id;
    out["pool_stats"] = pool_stats_json;

    return out;
}

// ============================================================
// 命令行参数解析
// ============================================================
struct Args {
    std::string instance_path;
    int max_iter       = 500;
    int pso_freq       = 50;
    int pso_particles  = 20;
    int pso_iter       = 30;
    int ls_freq        = 25;
    int no_improve_limit = -1;
    // 并行参数
    int num_threads      = 1;
    int push_freq        = 50;
    int stagnation_limit = 100;
    bool verbose   = false;
    bool pretty    = false;
};

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <instance.txt> [options]\n"
              << "  --max_iter N          ALNS 最大迭代次数 (default: 500)\n"
              << "  --pso_freq N          PSO 触发频率 (default: 50)\n"
              << "  --pso_particles N     PSO 粒子数 (default: 20)\n"
              << "  --pso_iter N          PSO 迭代次数 (default: 30)\n"
              << "  --ls_freq N           局部搜索频率 (default: 25)\n"
              << "  --no_improve_limit N  早停阈值 (default: disabled)\n"
              << "  --num_threads N       并行线程数 (default: 1)\n"
              << "  --push_freq N         推送公共池频率 (default: 50)\n"
              << "  --stagnation_limit N  停滞重启阈值 (default: 100)\n"
              << "  --verbose             显示求解进度\n"
              << "  --pretty              格式化 JSON 输出\n";
}

static Args parse_args(int argc, char* argv[]) {
    Args args;
    if (argc < 2) {
        print_usage(argv[0]);
        throw std::runtime_error("Missing instance path");
    }
    args.instance_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--verbose") { args.verbose = true; }
        else if (arg == "--pretty") { args.pretty = true; }
        else if (arg == "--max_iter" && i + 1 < argc)
            args.max_iter = std::stoi(argv[++i]);
        else if (arg == "--pso_freq" && i + 1 < argc)
            args.pso_freq = std::stoi(argv[++i]);
        else if (arg == "--pso_particles" && i + 1 < argc)
            args.pso_particles = std::stoi(argv[++i]);
        else if (arg == "--pso_iter" && i + 1 < argc)
            args.pso_iter = std::stoi(argv[++i]);
        else if (arg == "--ls_freq" && i + 1 < argc)
            args.ls_freq = std::stoi(argv[++i]);
        else if (arg == "--no_improve_limit" && i + 1 < argc)
            args.no_improve_limit = std::stoi(argv[++i]);
        else if (arg == "--num_threads" && i + 1 < argc)
            args.num_threads = std::stoi(argv[++i]);
        else if (arg == "--push_freq" && i + 1 < argc)
            args.push_freq = std::stoi(argv[++i]);
        else if (arg == "--stagnation_limit" && i + 1 < argc)
            args.stagnation_limit = std::stoi(argv[++i]);
    }
    return args;
}

// ============================================================
// 主函数
// ============================================================
int main(int argc, char* argv[]) {
    Args args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    try {
        // 解析算例
        Instance inst = parse_instance(args.instance_path);

        if (args.verbose) {
            std::cerr << "算例: " << args.instance_path << "\n"
                      << "  路网节点: " << inst.num_road_nodes
                      << ", 需求边: " << inst.num_edges
                      << ", 无人机: " << inst.num_drones << "\n"
                      << "  电池容量: " << inst.battery << "\n"
                      << "  求解模式: "
                      << (args.num_threads > 1
                          ? "异步并行（" + std::to_string(args.num_threads) + " 线程）"
                          : "单线程")
                      << "\n";
        }

        double t_start = get_time_sec();
        json output;

        if (args.num_threads > 1) {
            // ---- 并行模式 ----
            ParallelSolveResult par = parallel_solve(
                inst,
                args.num_threads,
                args.max_iter,
                args.push_freq,
                args.stagnation_limit,
                5,   // pool_capacity
                args.pso_freq,
                args.pso_particles,
                args.pso_iter,
                args.ls_freq,
                args.no_improve_limit,
                args.verbose
            );
            double solve_time = get_time_sec() - t_start;

            output = build_output_json(
                par.best_sol,
                inst,
                par.best_cost,
                solve_time,
                par.merged_history,
                par.per_thread_stats,
                par.pool_total_pushes,
                par.pool_total_pulls,
                par.pool_best_thread_id
            );
        } else {
            // ---- 单线程模式 ----
            GiantRouteALNSSolver solver(
                inst,
                args.max_iter,
                50,
                0.1, 0.4,
                33.0, 9.0, 3.0,
                0.8,
                0.998,
                args.pso_freq,
                args.pso_particles,
                args.pso_iter,
                args.ls_freq,
                args.no_improve_limit
            );

            ALNSSolveResult result = solver.solve(args.verbose);
            double solve_time = get_time_sec() - t_start;

            // 单线程：per_thread_stats 只含一个元素
            output = build_output_json(
                result.best_sol,
                inst,
                result.best_cost,
                solve_time,
                result.cost_history,
                {result.stats},
                0, 0, 0  // 单线程无共享池
            );
        }

        // 输出到 stdout
        if (args.pretty) {
            std::cout << output.dump(2) << std::endl;
        } else {
            std::cout << output.dump() << std::endl;
        }

    } catch (const std::exception& e) {
        json err_out;
        err_out["error"] = e.what();
        std::cout << err_out.dump() << std::endl;
        return 1;
    }

    return 0;
}
