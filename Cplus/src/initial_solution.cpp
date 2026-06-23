#include "functions.h"
#include "types.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>

// 每线程独立随机引擎（thread_local 保证多线程安全）
std::mt19937& get_rng() {
    thread_local std::mt19937 rng(std::random_device{}());
    return rng;
}

// ============================================================
// 7. 初始 Giant Route 生成
// ============================================================

// 方案B：全局最近邻贪婪构建
static std::vector<SubEdge> greedy_giant_route_nn(
    const std::vector<SubEdge>& sub_edges,
    const Instance& inst)
{
    auto [depot_x, depot_y] = inst.node_coord(inst.depot_idx);
    std::vector<SubEdge> unvisited = sub_edges;
    std::vector<SubEdge> giant;
    double cx = depot_x, cy = depot_y;

    while (!unvisited.empty()) {
        int best_idx = 0;
        double best_dist = std::numeric_limits<double>::infinity();
        for (int i = 0; i < (int)unvisited.size(); ++i) {
            const auto& se = unvisited[i];
            double d = std::min(
                std::hypot(cx - se.ax, cy - se.ay),
                std::hypot(cx - se.bx, cy - se.by));
            if (d < best_dist) { best_dist = d; best_idx = i; }
        }
        const auto& se = unvisited[best_idx];
        giant.push_back(se);
        double da = std::hypot(cx - se.ax, cy - se.ay);
        double db = std::hypot(cx - se.bx, cy - se.by);
        if (da <= db) { cx = se.bx; cy = se.by; }
        else          { cx = se.ax; cy = se.ay; }
        unvisited.erase(unvisited.begin() + best_idx);
    }
    return giant;
}

// 方案A：贪心最近邻多路径解
static Solution greedy_multi_route_solution(
    const std::vector<SubEdge>& sub_edges,
    const Instance& inst)
{
    auto [depot_x, depot_y] = inst.node_coord(inst.depot_idx);
    int n_drones = inst.num_drones;
    Solution sol(n_drones, inst.num_edges);
    std::vector<SubEdge> unassigned = sub_edges;
    std::vector<std::pair<double,double>> curr_pos(n_drones, {depot_x, depot_y});
    std::vector<double> curr_energy(n_drones, 0.0);

    int max_outer = (int)sub_edges.size() * n_drones * 2;
    int attempt = 0;

    while (!unassigned.empty() && attempt < max_outer) {
        ++attempt;
        bool placed = false;

        // 按能量升序排列无人机
        std::vector<int> drone_order(n_drones);
        for (int i = 0; i < n_drones; ++i) drone_order[i] = i;
        std::sort(drone_order.begin(), drone_order.end(),
            [&](int a, int b){ return curr_energy[a] < curr_energy[b]; });

        for (int di : drone_order) {
            double cx = curr_pos[di].first, cy = curr_pos[di].second;
            int best_uid = -1;
            double best_metric = std::numeric_limits<double>::infinity();
            bool best_dir = true;

            for (int uid = 0; uid < (int)unassigned.size(); ++uid) {
                const auto& se = unassigned[uid];
                double da = std::hypot(cx - se.ax, cy - se.ay);
                double db = std::hypot(cx - se.bx, cy - se.by);
                bool direction = (da <= db);
                double ex = direction ? se.bx : se.ax;
                double ey = direction ? se.by : se.ay;
                double to_start = std::min(da, db);
                double to_depot = std::hypot(ex - depot_x, ey - depot_y);
                double energy_add = to_start * inst.transfer_coef
                                    + se.length() * inst.inspect_coef;
                double to_depot_e = to_depot * inst.transfer_coef;
                if (curr_energy[di] + energy_add + to_depot_e <= inst.battery) {
                    if (to_start < best_metric) {
                        best_metric = to_start;
                        best_uid = uid;
                        best_dir = direction;
                    }
                }
            }

            if (best_uid >= 0) {
                const auto& se = unassigned[best_uid];
                sol.routes[di].sub_edges.push_back(se);
                sol.routes[di].directions.push_back(best_dir);
                double sx = best_dir ? se.ax : se.bx;
                double sy = best_dir ? se.ay : se.by;
                double ex = best_dir ? se.bx : se.ax;
                double ey = best_dir ? se.by : se.ay;
                curr_energy[di] += (std::hypot(curr_pos[di].first - sx,
                                               curr_pos[di].second - sy)
                                    * inst.transfer_coef
                                    + se.length() * inst.inspect_coef);
                curr_pos[di] = {ex, ey};
                unassigned.erase(unassigned.begin() + best_uid);
                placed = true;
                break;
            }
        }

        if (!placed) {
            // 强制分配给能量最低的无人机
            int di = (int)(std::min_element(curr_energy.begin(), curr_energy.end())
                           - curr_energy.begin());
            const auto& se = unassigned[0];
            double da = std::hypot(curr_pos[di].first - se.ax,
                                   curr_pos[di].second - se.ay);
            double db = std::hypot(curr_pos[di].first - se.bx,
                                   curr_pos[di].second - se.by);
            bool direction = (da <= db);
            double ex = direction ? se.bx : se.ax;
            double ey = direction ? se.by : se.ay;
            sol.routes[di].sub_edges.push_back(se);
            sol.routes[di].directions.push_back(direction);
            curr_energy[di] += std::min(da, db) * inst.transfer_coef
                                + se.length() * inst.inspect_coef;
            curr_pos[di] = {ex, ey};
            unassigned.erase(unassigned.begin());
        }
    }
    sol.invalidate_cache();
    return sol;
}

GiantRouteSolution generate_initial_gs(
    const Instance& inst,
    const std::vector<std::optional<double>>& breakpoints,
    const std::string& strategy)
{
    auto sub_edges = build_sub_edges(inst, breakpoints);
    GiantRouteSolution gs(inst.num_edges);
    gs.breakpoints = breakpoints;

    if (strategy == "nearest_neighbor") {
        gs.giant_route = greedy_giant_route_nn(sub_edges, inst);
    } else if (strategy == "multi_route") {
        Solution sol = greedy_multi_route_solution(sub_edges, inst);
        gs.giant_route.clear();
        for (const auto& route : sol.routes) {
            for (const auto& se : route.sub_edges)
                gs.giant_route.push_back(se);
        }
        // 补充未进入任何路径的子边
        std::vector<const SubEdge*> assigned_ptrs;
        for (const auto& r : sol.routes)
            for (const auto& se : r.sub_edges)
                assigned_ptrs.push_back(&se);
        // 简单处理：遍历sub_edges，按origin+seg匹配
        std::vector<std::pair<int,int>> assigned_ids;
        for (const auto& r : sol.routes)
            for (const auto& se : r.sub_edges)
                assigned_ids.emplace_back(se.origin_edge_idx, se.seg);
        for (const auto& se : sub_edges) {
            bool found = false;
            for (auto& id : assigned_ids) {
                if (id.first == se.origin_edge_idx && id.second == se.seg) {
                    id = {-1, -1};  // 标记已用
                    found = true;
                    break;
                }
            }
            if (!found) gs.giant_route.push_back(se);
        }
    } else if (strategy == "random") {
        gs.giant_route = sub_edges;
        std::shuffle(gs.giant_route.begin(), gs.giant_route.end(), get_rng());
    } else {
        gs.giant_route = greedy_giant_route_nn(sub_edges, inst);
    }
    return gs;
}

GiantRouteSolution multi_start_initial_gs(const Instance& inst, int n_starts) {
    std::vector<std::pair<double, GiantRouteSolution>> candidates;

    auto try_gs = [&](const std::vector<std::optional<double>>& bps,
                      const std::string& strategy) {
        try {
            GiantRouteSolution gs = generate_initial_gs(inst, bps, strategy);
            double cost = evaluate_gs(gs, inst);
            candidates.emplace_back(cost, gs);
        } catch (...) {}
    };

    // 无断点
    std::vector<std::optional<double>> bps_none(inst.num_edges, std::nullopt);
    try_gs(bps_none, "nearest_neighbor");
    try_gs(bps_none, "multi_route");

    // 全部断点在 0.5
    std::vector<std::optional<double>> bps_all(inst.num_edges, 0.5);
    try_gs(bps_all, "nearest_neighbor");
    try_gs(bps_all, "multi_route");

    // 随机断点配置
    std::uniform_real_distribution<double> uni_lam(0.15, 0.85);
    std::uniform_real_distribution<double> uni_prob(0.0, 1.0);
    for (int _ = 0; _ < std::max(0, n_starts - 4); ++_) {
        std::vector<std::optional<double>> bps_rand(inst.num_edges);
        for (int i = 0; i < inst.num_edges; ++i) {
            if (uni_prob(get_rng()) < 0.4)
                bps_rand[i] = uni_lam(get_rng());
            else
                bps_rand[i] = std::nullopt;
        }
        try_gs(bps_rand, "nearest_neighbor");
    }

    if (candidates.empty())
        throw std::runtime_error("多启动初始解生成失败，请检查算例参数。");

    auto best_it = std::min_element(candidates.begin(), candidates.end(),
        [](const auto& a, const auto& b){ return a.first < b.first; });
    return best_it->second;
}
