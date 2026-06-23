#include "types.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

// ============================================================
// 3. 子边构建工具
// ============================================================

std::vector<SubEdge> build_sub_edges(
    const Instance& inst,
    const std::vector<std::optional<double>>& breakpoints)
{
    std::vector<SubEdge> sub_edges;
    for (int ei = 0; ei < (int)inst.edges.size(); ++ei) {
        auto [u, v] = inst.edges[ei];
        double ux = inst.x[u], uy = inst.y[u];
        double vx = inst.x[v], vy = inst.y[v];
        const auto& lam_opt = breakpoints[ei];
        if (!lam_opt.has_value()) {
            sub_edges.push_back({ei, 0, ux, uy, vx, vy});
        } else {
            double lam = *lam_opt;
            double bpx = ux + lam * (vx - ux);
            double bpy = uy + lam * (vy - uy);
            sub_edges.push_back({ei, 1, ux, uy, bpx, bpy});
            sub_edges.push_back({ei, 2, bpx, bpy, vx, vy});
        }
    }
    return sub_edges;
}

// 断点位置更新后，重建 Giant Route 中所有子边的坐标
GiantRouteSolution rebuild_sub_edges_in_giant_route(
    const GiantRouteSolution& gs,
    const Instance& inst,
    const std::vector<std::optional<double>>& new_breakpoints)
{
    GiantRouteSolution new_gs(gs.num_edges);
    new_gs.breakpoints = new_breakpoints;
    for (const auto& se : gs.giant_route) {
        int ei = se.origin_edge_idx;
        auto [u, v] = inst.edges[ei];
        double ux = inst.x[u], uy = inst.y[u];
        double vx = inst.x[v], vy = inst.y[v];
        const auto& lam_opt = new_breakpoints[ei];

        SubEdge new_se = se;
        if (se.seg == 0) {
            new_se = {ei, 0, ux, uy, vx, vy};
        } else {
            double bpx, bpy;
            if (lam_opt.has_value()) {
                double lam = *lam_opt;
                bpx = ux + lam * (vx - ux);
                bpy = uy + lam * (vy - uy);
            } else {
                bpx = (ux + vx) / 2.0;
                bpy = (uy + vy) / 2.0;
            }
            if (se.seg == 1) {
                new_se = {ei, 1, ux, uy, bpx, bpy};
            } else {
                new_se = {ei, 2, bpx, bpy, vx, vy};
            }
        }
        new_gs.giant_route.push_back(new_se);
    }
    return new_gs;
}

// ============================================================
// 4. 方向贪婪确定
// ============================================================

std::vector<bool> assign_directions_greedy(
    const std::vector<SubEdge>& sub_edges,
    double depot_x, double depot_y)
{
    std::vector<bool> directions;
    double cx = depot_x, cy = depot_y;
    for (const auto& se : sub_edges) {
        double dist_a = std::hypot(cx - se.ax, cy - se.ay);
        double dist_b = std::hypot(cx - se.bx, cy - se.by);
        if (dist_a <= dist_b) {
            directions.push_back(true);  // a->b
            cx = se.bx; cy = se.by;
        } else {
            directions.push_back(false); // b->a
            cx = se.ax; cy = se.ay;
        }
    }
    return directions;
}

// ============================================================
// 5. 费用计算
// ============================================================

double compute_route_distance(const DroneRoute& route, const Instance& inst) {
    if (route.empty()) return 0.0;
    auto [depot_x, depot_y] = inst.node_coord(inst.depot_idx);
    int n = (int)route.sub_edges.size();
    double inspect_dist = 0.0;
    for (const auto& se : route.sub_edges)
        inspect_dist += se.length();

    auto [sx, sy] = route.start_point(0);
    double transfer_dist = std::hypot(depot_x - sx, depot_y - sy);
    for (int i = 0; i < n - 1; ++i) {
        auto [ex, ey] = route.end_point(i);
        auto [nsx, nsy] = route.start_point(i + 1);
        transfer_dist += std::hypot(ex - nsx, ey - nsy);
    }
    auto [ex, ey] = route.end_point(n - 1);
    transfer_dist += std::hypot(ex - depot_x, ey - depot_y);
    return inspect_dist * inst.inspect_coef + transfer_dist * inst.transfer_coef;
}

std::pair<double,double> compute_route_raw_distance(
    const DroneRoute& route, const Instance& inst)
{
    if (route.empty()) return {0.0, 0.0};
    auto [depot_x, depot_y] = inst.node_coord(inst.depot_idx);
    int n = (int)route.sub_edges.size();
    double inspect_dist = 0.0;
    for (const auto& se : route.sub_edges)
        inspect_dist += se.length();

    auto [sx, sy] = route.start_point(0);
    double transfer_dist = std::hypot(depot_x - sx, depot_y - sy);
    for (int i = 0; i < n - 1; ++i) {
        auto [ex, ey] = route.end_point(i);
        auto [nsx, nsy] = route.start_point(i + 1);
        transfer_dist += std::hypot(ex - nsx, ey - nsy);
    }
    auto [ex, ey] = route.end_point(n - 1);
    transfer_dist += std::hypot(ex - depot_x, ey - depot_y);
    return {inspect_dist, transfer_dist};
}

double compute_route_cost(const DroneRoute& route, const Instance& inst) {
    if (route.empty()) return 0.0;
    auto [depot_x, depot_y] = inst.node_coord(inst.depot_idx);
    int n = (int)route.sub_edges.size();
    double cost = inst.call_cost;
    auto [sx, sy] = route.start_point(0);
    cost += std::hypot(depot_x - sx, depot_y - sy) * inst.transfer_coef * inst.energy_cost;
    for (int i = 0; i < n; ++i) {
        cost += route.sub_edges[i].length() * inst.inspect_coef * inst.energy_cost;
        if (i + 1 < n) {
            auto [ex, ey] = route.end_point(i);
            auto [nsx, nsy] = route.start_point(i + 1);
            cost += std::hypot(ex - nsx, ey - nsy) * inst.transfer_coef * inst.energy_cost;
        }
    }
    auto [ex, ey] = route.end_point(n - 1);
    cost += std::hypot(ex - depot_x, ey - depot_y) * inst.transfer_coef * inst.energy_cost;
    return cost;
}

double compute_cost(Solution& sol, const Instance& inst) {
    if (sol.cost_cache.has_value()) return *sol.cost_cache;
    double total = 0.0;
    for (const auto& r : sol.routes)
        if (!r.empty())
            total += compute_route_cost(r, inst);
    sol.cost_cache = total;
    return total;
}

// 计算一段子边（从depot出发，贪婪方向）的费用和能量消耗
// 返回 (cost_with_call, energy)
std::pair<double,double> compute_segment_cost(
    const std::vector<SubEdge>& sub_edges,
    const Instance& inst,
    double depot_x, double depot_y)
{
    if (sub_edges.empty()) return {0.0, 0.0};
    int n = (int)sub_edges.size();
    auto dirs = assign_directions_greedy(sub_edges, depot_x, depot_y);

    auto start_fn = [&](int i) -> std::pair<double,double> {
        const auto& se = sub_edges[i];
        return dirs[i] ? std::make_pair(se.ax, se.ay) : std::make_pair(se.bx, se.by);
    };
    auto end_fn = [&](int i) -> std::pair<double,double> {
        const auto& se = sub_edges[i];
        return dirs[i] ? std::make_pair(se.bx, se.by) : std::make_pair(se.ax, se.ay);
    };

    double inspect_dist = 0.0;
    for (const auto& se : sub_edges)
        inspect_dist += se.length();

    auto [sx, sy] = start_fn(0);
    double transfer_dist = std::hypot(depot_x - sx, depot_y - sy);
    for (int i = 0; i < n - 1; ++i) {
        auto [ex, ey] = end_fn(i);
        auto [nsx, nsy] = start_fn(i + 1);
        transfer_dist += std::hypot(ex - nsx, ey - nsy);
    }
    auto [ex, ey] = end_fn(n - 1);
    transfer_dist += std::hypot(ex - depot_x, ey - depot_y);

    double energy = inspect_dist * inst.inspect_coef + transfer_dist * inst.transfer_coef;
    double cost = inst.call_cost
                  + inspect_dist * inst.inspect_coef * inst.energy_cost
                  + transfer_dist * inst.transfer_coef * inst.energy_cost;
    return {cost, energy};
}
