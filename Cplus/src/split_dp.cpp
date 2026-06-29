#include "functions.h"
#include "types.h"
#include <cmath>
#include <stdexcept>
#include <limits>
#include <sstream>

// ============================================================
// 6. Split DP 算法
// ============================================================

Solution split_dp(
    const std::vector<SubEdge>& giant_route,
    const Instance& inst,
    int max_drones)  // max_drones <= 0 表示不限制
{
    int n = (int)giant_route.size();
    if (n == 0) {
        return Solution(inst.num_drones, inst.num_edges);
    }

    auto [depot_x, depot_y] = inst.node_coord(inst.depot_idx);
    constexpr double INF = std::numeric_limits<double>::infinity();

    std::vector<double> dp(n + 1, INF);
    std::vector<int> prev_cut(n + 1, -1);
    dp[0] = 0.0;

    // 预检：每条单独子边是否可行
    for (int k = 0; k < n; ++k) {
        const auto& se = giant_route[k];
        double dist_a = std::hypot(depot_x - se.ax, depot_y - se.ay);
        double dist_b = std::hypot(depot_x - se.bx, depot_y - se.by);
        double sx, sy, ex, ey;
        if (dist_a <= dist_b) { sx = se.ax; sy = se.ay; ex = se.bx; ey = se.by; }
        else                  { sx = se.bx; sy = se.by; ex = se.ax; ey = se.ay; }
        double energy_single = se.length() * inst.inspect_coef
            + (std::hypot(depot_x - sx, depot_y - sy)
               + std::hypot(ex - depot_x, ey - depot_y)) * inst.transfer_coef;
        if (energy_single > inst.battery) {
            std::ostringstream oss;
            oss << "子边 origin_edge=" << se.origin_edge_idx
                << "(seg=" << se.seg << ") 单独飞行能量 " << energy_single
                << " > 电池容量 " << inst.battery;
            throw std::runtime_error(oss.str());
        }
    }

    for (int i = 0; i < n; ++i) {
        if (dp[i] == INF) continue;

        double cx = depot_x, cy = depot_y;
        double inspect_dist = 0.0;
        double transfer_dist_to_first = 0.0;
        double transfer_dist_between = 0.0;

        for (int k = i; k < n; ++k) {
            const auto& se = giant_route[k];
            double dist_a = std::hypot(cx - se.ax, cy - se.ay);
            double dist_b = std::hypot(cx - se.bx, cy - se.by);
            double sx, sy, ex, ey;
            if (dist_a <= dist_b) { sx = se.ax; sy = se.ay; ex = se.bx; ey = se.by; }
            else                  { sx = se.bx; sy = se.by; ex = se.ax; ey = se.ay; }

            if (k == i) {
                transfer_dist_to_first = std::hypot(depot_x - sx, depot_y - sy);
                transfer_dist_between = 0.0;
            } else {
                transfer_dist_between += std::hypot(cx - sx, cy - sy);
            }

            inspect_dist += se.length();
            cx = ex; cy = ey;

            double to_depot = std::hypot(cx - depot_x, cy - depot_y);
            double total_transfer = transfer_dist_to_first + transfer_dist_between + to_depot;
            double energy = inspect_dist * inst.inspect_coef
                            + total_transfer * inst.transfer_coef;

            if (energy > inst.battery) break;

            double seg_cost = inst.call_cost
                              + inspect_dist * inst.inspect_coef * inst.energy_cost
                              + total_transfer * inst.transfer_coef * inst.energy_cost;

            double new_cost = dp[i] + seg_cost;
            if (new_cost < dp[k + 1]) {
                dp[k + 1] = new_cost;
                prev_cut[k + 1] = i;
            }
        }
    }

    if (dp[n] == INF) {
        std::ostringstream oss;
        oss << "Split DP 无法完成所有子边的覆盖（每条子边单独飞行能量均可行，" 
            << "但组合后无解，请检查算例参数）";
        throw std::runtime_error(oss.str());
    }

    // 回溯切割点
    std::vector<std::pair<int,int>> segments;
    int cur = n;
    while (cur > 0) {
        int j = prev_cut[cur];
        segments.emplace_back(j, cur - 1);
        cur = j;
    }
    std::reverse(segments.begin(), segments.end());

    // 不限制无人机数量：实际使用段数可能超过 inst.num_drones
    // Solution 内部路径槽数取二者较大值，确保容纳所有段
    int actual_drones = std::max(inst.num_drones, (int)segments.size());
    Solution sol(actual_drones, inst.num_edges);
    for (int drone_idx = 0; drone_idx < (int)segments.size(); ++drone_idx) {
        auto [si, ei_] = segments[drone_idx];
        std::vector<SubEdge> seg_ses(
            giant_route.begin() + si,
            giant_route.begin() + ei_ + 1);
        auto dirs = assign_directions_greedy(seg_ses, depot_x, depot_y);
        sol.routes[drone_idx] = DroneRoute{seg_ses, dirs};
    }
    sol.invalidate_cache();
    return sol;
}

Solution gs_to_solution(const GiantRouteSolution& gs, const Instance& inst) {
    // max_drones=0 表示不限制无人机数量，由 DP 自行决定最少切割数
    Solution sol = split_dp(gs.giant_route, inst, 0);
    sol.breakpoints = gs.breakpoints;
    return sol;
}

double evaluate_gs(GiantRouteSolution& gs, const Instance& inst) {
    if (gs.cost_cache.has_value()) return *gs.cost_cache;
    Solution sol = gs_to_solution(gs, inst);
    double cost = compute_cost(sol, inst);
    gs.cost_cache = cost;
    return cost;
}
