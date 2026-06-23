#pragma once
#include <vector>
#include <string>
#include <optional>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <memory>
#include <limits>
#include <mutex>
#include <functional>

// ============================================================
// 1. Instance - 算例数据
// ============================================================
struct Instance {
    int num_depots;
    int num_road_nodes;
    int total_nodes;
    int num_edges;
    int num_drones;
    double battery;
    double speed;
    double energy_cost;
    double call_cost;
    double inspect_coef;
    double transfer_coef;
    double big_m;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<std::pair<int,int>> edges;
    int depot_idx = 0;

    double edge_length(int edge_idx) const {
        auto [u, v] = edges[edge_idx];
        double ux = x[u], uy = y[u];
        double vx = x[v], vy = y[v];
        return std::hypot(ux - vx, uy - vy);
    }

    std::pair<double,double> point_on_edge(int edge_idx, double lam) const {
        auto [u, v] = edges[edge_idx];
        double ux = x[u], uy = y[u];
        double vx = x[v], vy = y[v];
        return {ux + lam * (vx - ux), uy + lam * (vy - uy)};
    }

    double euclidean(double ax, double ay, double bx, double by) const {
        return std::hypot(ax - bx, ay - by);
    }

    std::pair<double,double> node_coord(int node_idx) const {
        return {x[node_idx], y[node_idx]};
    }
};

// ============================================================
// 2. SubEdge - 子边
// ============================================================
struct SubEdge {
    int origin_edge_idx;
    int seg;   // 0=整边, 1=第一段(u->bp), 2=第二段(bp->v)
    double ax, ay;
    double bx, by;

    double length() const {
        return std::hypot(ax - bx, ay - by);
    }
};

// ============================================================
// 3. DroneRoute - 单架无人机路径
// ============================================================
struct DroneRoute {
    std::vector<SubEdge> sub_edges;
    std::vector<bool> directions;  // true: a->b

    std::pair<double,double> start_point(int i) const {
        const auto& se = sub_edges[i];
        if (directions[i]) return {se.ax, se.ay};
        return {se.bx, se.by};
    }

    std::pair<double,double> end_point(int i) const {
        const auto& se = sub_edges[i];
        if (directions[i]) return {se.bx, se.by};
        return {se.ax, se.ay};
    }

    bool empty() const { return sub_edges.empty(); }
};

// ============================================================
// 4. Solution - 完整解
// ============================================================
struct Solution {
    int num_drones;
    int num_edges;
    std::vector<std::optional<double>> breakpoints;
    std::vector<DroneRoute> routes;
    mutable std::optional<double> cost_cache;

    Solution() : num_drones(0), num_edges(0) {}

    Solution(int nd, int ne) : num_drones(nd), num_edges(ne),
        breakpoints(ne, std::nullopt), routes(nd) {}

    void invalidate_cache() const { cost_cache = std::nullopt; }

    Solution copy() const {
        return *this;
    }
};

// ============================================================
// 5. GiantRouteSolution - Giant Route 编码的解
// ============================================================
struct GiantRouteSolution {
    int num_edges;
    std::vector<SubEdge> giant_route;
    std::vector<std::optional<double>> breakpoints;
    mutable std::optional<double> cost_cache;

    GiantRouteSolution() : num_edges(0) {}

    explicit GiantRouteSolution(int ne) : num_edges(ne),
        breakpoints(ne, std::nullopt) {}

    void invalidate_cache() const { cost_cache = std::nullopt; }

    GiantRouteSolution copy() const {
        return *this;
    }
};
