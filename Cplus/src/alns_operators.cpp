#include "functions.h"
#include "types.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>
#include <vector>
#include <iostream>

extern std::mt19937& get_rng();

// ============================================================
// 8. ALNS 破坏算子（操作 Giant Route 排列）
// ============================================================

// 随机移除
std::pair<GiantRouteSolution, std::vector<SubEdge>>
destroy_random_removal(const GiantRouteSolution& gs, double removal_fraction)
{
    GiantRouteSolution new_gs = gs.copy();
    int n = (int)new_gs.giant_route.size();
    if (n == 0) return {new_gs, {}};

    int num_remove = std::max(1, (int)(n * removal_fraction));
    num_remove = std::min(num_remove, n);

    // 随机选择要删除的索引
    std::vector<int> indices(n);
    for (int i = 0; i < n; ++i) indices[i] = i;
    std::shuffle(indices.begin(), indices.end(), get_rng());
    indices.resize(num_remove);
    std::sort(indices.begin(), indices.end(), std::greater<int>());

    std::vector<SubEdge> removed;
    for (int idx : indices) {
        removed.push_back(new_gs.giant_route[idx]);
        new_gs.giant_route.erase(new_gs.giant_route.begin() + idx);
    }
    new_gs.invalidate_cache();
    return {new_gs, removed};
}

// 最差移除
std::pair<GiantRouteSolution, std::vector<SubEdge>>
destroy_worst_removal(const GiantRouteSolution& gs, const Instance& inst,
                      double removal_fraction)
{
    GiantRouteSolution new_gs = gs.copy();
    int n = (int)new_gs.giant_route.size();
    if (n == 0) return {new_gs, {}};

    auto [depot_x, depot_y] = inst.node_coord(inst.depot_idx);
    auto dirs = assign_directions_greedy(new_gs.giant_route, depot_x, depot_y);

    auto get_start = [&](int i) -> std::pair<double,double> {
        const auto& se = new_gs.giant_route[i];
        return dirs[i] ? std::make_pair(se.ax, se.ay) : std::make_pair(se.bx, se.by);
    };
    auto get_end = [&](int i) -> std::pair<double,double> {
        const auto& se = new_gs.giant_route[i];
        return dirs[i] ? std::make_pair(se.bx, se.by) : std::make_pair(se.ax, se.ay);
    };

    std::vector<std::pair<double,int>> scores;
    for (int i = 0; i < n; ++i) {
        const auto& se = new_gs.giant_route[i];
        auto [sx, sy] = get_start(i);
        auto [ex, ey] = get_end(i);
        double prev_x = (i == 0) ? depot_x : get_end(i - 1).first;
        double prev_y = (i == 0) ? depot_y : get_end(i - 1).second;
        double next_x = (i == n - 1) ? depot_x : get_start(i + 1).first;
        double next_y = (i == n - 1) ? depot_y : get_start(i + 1).second;

        double current_cost = std::hypot(prev_x - sx, prev_y - sy)
                              + se.length()
                              + std::hypot(ex - next_x, ey - next_y);
        double bypass_cost = std::hypot(prev_x - next_x, prev_y - next_y);
        double score = (current_cost - bypass_cost) * inst.transfer_coef
                       + se.length() * inst.inspect_coef;
        scores.emplace_back(score, i);
    }

    std::sort(scores.begin(), scores.end(),
              [](const auto& a, const auto& b){ return a.first > b.first; });

    int num_remove = std::max(1, (int)(n * removal_fraction));
    std::vector<int> remove_indices;
    for (int i = 0; i < num_remove; ++i)
        remove_indices.push_back(scores[i].second);
    std::sort(remove_indices.begin(), remove_indices.end(), std::greater<int>());

    std::vector<SubEdge> removed;
    for (int idx : remove_indices) {
        removed.push_back(new_gs.giant_route[idx]);
        new_gs.giant_route.erase(new_gs.giant_route.begin() + idx);
    }
    new_gs.invalidate_cache();
    return {new_gs, removed};
}

// 连续片段移除
std::pair<GiantRouteSolution, std::vector<SubEdge>>
destroy_segment_removal(const GiantRouteSolution& gs, double removal_fraction)
{
    GiantRouteSolution new_gs = gs.copy();
    int n = (int)new_gs.giant_route.size();
    if (n == 0) return {new_gs, {}};

    int num_remove = std::max(1, (int)(n * removal_fraction));
    num_remove = std::min(num_remove, n);

    std::uniform_int_distribution<int> dist(0, n - num_remove);
    int start_i = dist(get_rng());

    std::vector<SubEdge> removed(
        new_gs.giant_route.begin() + start_i,
        new_gs.giant_route.begin() + start_i + num_remove);
    new_gs.giant_route.erase(
        new_gs.giant_route.begin() + start_i,
        new_gs.giant_route.begin() + start_i + num_remove);
    new_gs.invalidate_cache();
    return {new_gs, removed};
}

// ============================================================
// 9. ALNS 修复算子
// ============================================================

// 贪婪插入
GiantRouteSolution repair_greedy_insert(
    const GiantRouteSolution& gs_in,
    const Instance& inst,
    std::vector<SubEdge> removed)
{
    GiantRouteSolution new_gs = gs_in.copy();
    auto [depot_x, depot_y] = inst.node_coord(inst.depot_idx);
    std::shuffle(removed.begin(), removed.end(), get_rng());

    for (const auto& se : removed) {
        int n = (int)new_gs.giant_route.size();
        double best_delta = std::numeric_limits<double>::infinity();
        int best_pos = 0;

        for (int pos = 0; pos <= n; ++pos) {
            double prev_x, prev_y;
            if (pos == 0) {
                prev_x = depot_x; prev_y = depot_y;
            } else {
                const auto& prev_se = new_gs.giant_route[pos - 1];
                double px0, py0;
                if (pos == 1) { px0 = depot_x; py0 = depot_y; }
                else {
                    const auto& ppse = new_gs.giant_route[pos - 2];
                    double da = std::hypot(depot_x - ppse.ax, depot_y - ppse.ay);
                    double db = std::hypot(depot_x - ppse.bx, depot_y - ppse.by);
                    px0 = (da <= db) ? ppse.bx : ppse.ax;
                    py0 = (da <= db) ? ppse.by : ppse.ay;
                }
                double da = std::hypot(px0 - prev_se.ax, py0 - prev_se.ay);
                double db = std::hypot(px0 - prev_se.bx, py0 - prev_se.by);
                prev_x = (da <= db) ? prev_se.bx : prev_se.ax;
                prev_y = (da <= db) ? prev_se.by : prev_se.ay;
            }

            double next_x, next_y;
            if (pos == n) {
                next_x = depot_x; next_y = depot_y;
            } else {
                const auto& next_se = new_gs.giant_route[pos];
                double da = std::hypot(prev_x - next_se.ax, prev_y - next_se.ay);
                double db = std::hypot(prev_x - next_se.bx, prev_y - next_se.by);
                next_x = (da <= db) ? next_se.ax : next_se.bx;
                next_y = (da <= db) ? next_se.ay : next_se.by;
            }

            double da = std::hypot(prev_x - se.ax, prev_y - se.ay);
            double db = std::hypot(prev_x - se.bx, prev_y - se.by);
            double sx, sy, ex, ey;
            if (da <= db) { sx = se.ax; sy = se.ay; ex = se.bx; ey = se.by; }
            else          { sx = se.bx; sy = se.by; ex = se.ax; ey = se.ay; }

            double old_transfer = std::hypot(prev_x - next_x, prev_y - next_y);
            double new_transfer = std::hypot(prev_x - sx, prev_y - sy)
                                  + se.length()
                                  + std::hypot(ex - next_x, ey - next_y);
            double delta = ((new_transfer - old_transfer) * inst.transfer_coef
                            + se.length() * inst.inspect_coef) * inst.energy_cost;

            if (delta < best_delta) {
                best_delta = delta;
                best_pos = pos;
            }
        }
        new_gs.giant_route.insert(new_gs.giant_route.begin() + best_pos, se);
    }
    new_gs.invalidate_cache();
    return new_gs;
}

// 随机插入
GiantRouteSolution repair_random_insert(
    const GiantRouteSolution& gs_in,
    std::vector<SubEdge> removed)
{
    GiantRouteSolution new_gs = gs_in.copy();
    std::shuffle(removed.begin(), removed.end(), get_rng());
    for (const auto& se : removed) {
        int n = (int)new_gs.giant_route.size();
        std::uniform_int_distribution<int> dist(0, n);
        int pos = dist(get_rng());
        new_gs.giant_route.insert(new_gs.giant_route.begin() + pos, se);
    }
    new_gs.invalidate_cache();
    return new_gs;
}

// Regret-2 插入
GiantRouteSolution repair_regret_insert(
    const GiantRouteSolution& gs_in,
    const Instance& inst,
    std::vector<SubEdge> removed)
{
    GiantRouteSolution new_gs = gs_in.copy();
    auto [depot_x, depot_y] = inst.node_coord(inst.depot_idx);

    auto calc_delta = [&](const GiantRouteSolution& cur_gs,
                          const SubEdge& se, int pos) -> double {
        int n = (int)cur_gs.giant_route.size();
        double prev_x, prev_y;
        if (pos == 0) {
            prev_x = depot_x; prev_y = depot_y;
        } else {
            const auto& prev_se = cur_gs.giant_route[pos - 1];
            double px0, py0;
            if (pos == 1) { px0 = depot_x; py0 = depot_y; }
            else {
                const auto& ppse = cur_gs.giant_route[pos - 2];
                double da = std::hypot(depot_x - ppse.ax, depot_y - ppse.ay);
                double db = std::hypot(depot_x - ppse.bx, depot_y - ppse.by);
                px0 = (da <= db) ? ppse.bx : ppse.ax;
                py0 = (da <= db) ? ppse.by : ppse.ay;
            }
            double da = std::hypot(px0 - prev_se.ax, py0 - prev_se.ay);
            double db = std::hypot(px0 - prev_se.bx, py0 - prev_se.by);
            prev_x = (da <= db) ? prev_se.bx : prev_se.ax;
            prev_y = (da <= db) ? prev_se.by : prev_se.ay;
        }
        double next_x, next_y;
        if (pos == n) {
            next_x = depot_x; next_y = depot_y;
        } else {
            const auto& next_se = cur_gs.giant_route[pos];
            double da = std::hypot(prev_x - next_se.ax, prev_y - next_se.ay);
            double db = std::hypot(prev_x - next_se.bx, prev_y - next_se.by);
            next_x = (da <= db) ? next_se.ax : next_se.bx;
            next_y = (da <= db) ? next_se.ay : next_se.by;
        }
        double da = std::hypot(prev_x - se.ax, prev_y - se.ay);
        double db = std::hypot(prev_x - se.bx, prev_y - se.by);
        double sx, sy, ex, ey;
        if (da <= db) { sx = se.ax; sy = se.ay; ex = se.bx; ey = se.by; }
        else          { sx = se.bx; sy = se.by; ex = se.ax; ey = se.ay; }
        double old_t = std::hypot(prev_x - next_x, prev_y - next_y);
        double new_t = std::hypot(prev_x - sx, prev_y - sy)
                       + se.length()
                       + std::hypot(ex - next_x, ey - next_y);
        return ((new_t - old_t) * inst.transfer_coef
                + se.length() * inst.inspect_coef) * inst.energy_cost;
    };

    while (!removed.empty()) {
        int best_se_idx = -1;
        int best_pos = 0;
        double best_regret = -std::numeric_limits<double>::infinity();

        for (int sei = 0; sei < (int)removed.size(); ++sei) {
            const auto& se = removed[sei];
            int n = (int)new_gs.giant_route.size();
            std::vector<double> deltas;
            for (int p = 0; p <= n; ++p)
                deltas.push_back(calc_delta(new_gs, se, p));
            std::sort(deltas.begin(), deltas.end());
            double best_d = deltas[0];
            double second_d = (deltas.size() > 1) ? deltas[1] : deltas[0];
            double regret = second_d - best_d;
            if (regret > best_regret) {
                best_regret = regret;
                best_se_idx = sei;
                // 找最优插入位置
                best_pos = 0;
                double min_delta = std::numeric_limits<double>::infinity();
                for (int p = 0; p <= n; ++p) {
                    double d = calc_delta(new_gs, se, p);
                    if (d < min_delta) { min_delta = d; best_pos = p; }
                }
            }
        }

        const auto& se = removed[best_se_idx];
        new_gs.giant_route.insert(new_gs.giant_route.begin() + best_pos, se);
        removed.erase(removed.begin() + best_se_idx);
    }
    new_gs.invalidate_cache();
    return new_gs;
}

// ============================================================
// 10. 局部搜索算子（2-opt 和 or-opt）
// ============================================================

GiantRouteSolution local_search_2opt(
    GiantRouteSolution gs, const Instance& inst, int max_no_improve)
{
    GiantRouteSolution best_gs = gs.copy();
    double best_cost = evaluate_gs(best_gs, inst);
    int n = (int)best_gs.giant_route.size();
    if (n < 4) return best_gs;

    int max_ni = (max_no_improve < 0) ? n * n : max_no_improve;
    int no_improve = 0;
    bool improved = true;

    while (improved && no_improve < max_ni) {
        improved = false;
        for (int i = 0; i < n - 1 && !improved; ++i) {
            for (int j = i + 2; j < n && !improved; ++j) {
                std::vector<SubEdge> new_route;
                for (int k = 0; k < i; ++k) new_route.push_back(best_gs.giant_route[k]);
                for (int k = j; k >= i; --k) new_route.push_back(best_gs.giant_route[k]);
                for (int k = j + 1; k < n; ++k) new_route.push_back(best_gs.giant_route[k]);

                GiantRouteSolution trial = best_gs.copy();
                trial.giant_route = new_route;
                trial.invalidate_cache();
                try {
                    double trial_cost = evaluate_gs(trial, inst);
                    if (trial_cost < best_cost - 1e-9) {
                        best_cost = trial_cost;
                        best_gs = trial.copy();
                        improved = true;
                        no_improve = 0;
                    }
                } catch (...) {}
            }
        }
        if (!improved) ++no_improve;
    }
    best_gs.cost_cache = best_cost;
    return best_gs;
}

GiantRouteSolution local_search_or_opt(
    GiantRouteSolution gs, const Instance& inst,
    std::vector<int> segment_sizes, int max_no_improve)
{
    if (segment_sizes.empty()) segment_sizes = {1, 2, 3};

    GiantRouteSolution best_gs = gs.copy();
    double best_cost = evaluate_gs(best_gs, inst);
    int n = (int)best_gs.giant_route.size();
    if (n < 3) return best_gs;

    int max_ni = (max_no_improve < 0) ? n * 2 : max_no_improve;
    int no_improve = 0;
    bool improved = true;

    while (improved && no_improve < max_ni) {
        improved = false;
        for (int k : segment_sizes) {
            if (k >= n) continue;
            for (int i = 0; i < n - k + 1 && !improved; ++i) {
                std::vector<SubEdge> segment(
                    best_gs.giant_route.begin() + i,
                    best_gs.giant_route.begin() + i + k);
                std::vector<SubEdge> rest;
                for (int t = 0; t < i; ++t) rest.push_back(best_gs.giant_route[t]);
                for (int t = i + k; t < n; ++t) rest.push_back(best_gs.giant_route[t]);

                for (int j = 0; j <= (int)rest.size() && !improved; ++j) {
                    if (j == i) continue;
                    std::vector<SubEdge> new_route;
                    for (int t = 0; t < j; ++t) new_route.push_back(rest[t]);
                    for (const auto& se : segment) new_route.push_back(se);
                    for (int t = j; t < (int)rest.size(); ++t) new_route.push_back(rest[t]);

                    GiantRouteSolution trial = best_gs.copy();
                    trial.giant_route = new_route;
                    trial.invalidate_cache();
                    try {
                        double trial_cost = evaluate_gs(trial, inst);
                        if (trial_cost < best_cost - 1e-9) {
                            best_cost = trial_cost;
                            best_gs = trial.copy();
                            improved = true;
                            no_improve = 0;
                        }
                    } catch (...) {}
                }
            }
        }
        if (!improved) ++no_improve;
    }
    best_gs.cost_cache = best_cost;
    return best_gs;
}

// ============================================================
// 11. 断点邻域搜索
// ============================================================

GiantRouteSolution try_add_remove_breakpoints(
    GiantRouteSolution gs, const Instance& inst, bool fast_mode)
{
    GiantRouteSolution best_gs = gs.copy();
    double best_cost = evaluate_gs(best_gs, inst);

    std::vector<int> edge_order(inst.num_edges);
    for (int i = 0; i < inst.num_edges; ++i) edge_order[i] = i;

    std::vector<double> candidates_for_add;
    if (fast_mode) {
        int n_try = std::min(inst.num_edges, std::max(2, inst.num_edges / 3));
        std::shuffle(edge_order.begin(), edge_order.end(), get_rng());
        edge_order.resize(n_try);
        std::uniform_real_distribution<double> uni(0.2, 0.8);
        candidates_for_add = {0.5, uni(get_rng())};
    } else {
        std::shuffle(edge_order.begin(), edge_order.end(), get_rng());
        std::uniform_real_distribution<double> uni1(0.1, 0.45);
        std::uniform_real_distribution<double> uni2(0.55, 0.9);
        candidates_for_add = {0.25, 0.33, 0.5, 0.67, 0.75,
                               uni1(get_rng()), uni2(get_rng())};
    }

    for (int ei : edge_order) {
        auto current_bps = best_gs.breakpoints;

        if (!current_bps[ei].has_value()) {
            // 尝试添加断点
            int se_idx = -1;
            for (int gi = 0; gi < (int)best_gs.giant_route.size(); ++gi) {
                if (best_gs.giant_route[gi].origin_edge_idx == ei
                    && best_gs.giant_route[gi].seg == 0) {
                    se_idx = gi; break;
                }
            }
            if (se_idx < 0) continue;

            auto [u, v] = inst.edges[ei];
            double ux = inst.x[u], uy = inst.y[u];
            double vx = inst.x[v], vy = inst.y[v];

            for (double lam : candidates_for_add) {
                double bpx = ux + lam * (vx - ux);
                double bpy = uy + lam * (vy - uy);
                SubEdge se1{ei, 1, ux, uy, bpx, bpy};
                SubEdge se2{ei, 2, bpx, bpy, vx, vy};

                GiantRouteSolution trial(best_gs.num_edges);
                trial.breakpoints = current_bps;
                trial.breakpoints[ei] = lam;
                trial.giant_route.clear();
                for (int gi = 0; gi < (int)best_gs.giant_route.size(); ++gi) {
                    if (gi == se_idx) {
                        trial.giant_route.push_back(se1);
                        trial.giant_route.push_back(se2);
                    } else {
                        trial.giant_route.push_back(best_gs.giant_route[gi]);
                    }
                }

                try {
                    double trial_cost = evaluate_gs(trial, inst);
                    if (trial_cost < best_cost - 1e-9) {
                        best_cost = trial_cost;
                        best_gs = trial.copy();
                        best_gs.cost_cache = trial_cost;
                        current_bps = trial.breakpoints;
                        break;
                    }
                } catch (...) {}
            }
        } else {
            // 尝试删除断点
            int idx1 = -1, idx2 = -1;
            for (int gi = 0; gi < (int)best_gs.giant_route.size(); ++gi) {
                if (best_gs.giant_route[gi].origin_edge_idx == ei) {
                    if (best_gs.giant_route[gi].seg == 1) idx1 = gi;
                    else if (best_gs.giant_route[gi].seg == 2) idx2 = gi;
                }
            }
            if (idx1 < 0 || idx2 < 0) continue;

            auto [u, v] = inst.edges[ei];
            double ux = inst.x[u], uy = inst.y[u];
            double vx = inst.x[v], vy = inst.y[v];
            SubEdge se_whole{ei, 0, ux, uy, vx, vy};

            int keep_idx = std::min(idx1, idx2);
            int del_idx  = std::max(idx1, idx2);
            GiantRouteSolution trial(best_gs.num_edges);
            trial.breakpoints = current_bps;
            trial.breakpoints[ei] = std::nullopt;
            trial.giant_route = best_gs.giant_route;
            trial.giant_route[keep_idx] = se_whole;
            trial.giant_route.erase(trial.giant_route.begin() + del_idx);

            double trial_cost;
            try {
                trial_cost = evaluate_gs(trial, inst);
            } catch (...) {
                trial_cost = std::numeric_limits<double>::infinity();
            }

            if (trial_cost < best_cost - 1e-9) {
                best_cost = trial_cost;
                best_gs = trial.copy();
                best_gs.cost_cache = trial_cost;
                current_bps = trial.breakpoints;
            } else if (!fast_mode) {
                // 不删除，尝试移动断点位置
                std::uniform_real_distribution<double> uni1(0.1, 0.45);
                std::uniform_real_distribution<double> uni2(0.55, 0.9);
                std::vector<double> move_cands = {0.2, 0.35, 0.5, 0.65, 0.8,
                                                  uni1(get_rng()), uni2(get_rng())};
                for (double lam : move_cands) {
                    auto new_bps = current_bps;
                    new_bps[ei] = lam;
                    GiantRouteSolution trial2 = rebuild_sub_edges_in_giant_route(
                        best_gs, inst, new_bps);
                    try {
                        double tc = evaluate_gs(trial2, inst);
                        if (tc < best_cost - 1e-9) {
                            best_cost = tc;
                            best_gs = trial2.copy();
                            best_gs.cost_cache = tc;
                            current_bps = new_bps;
                            break;
                        }
                    } catch (...) {}
                }
            }
        }
    }
    return best_gs;
}

// 三分搜索
static std::pair<double,double> ternary_search_lambda(
    double ux, double uy, double vx, double vy,
    double prev1_x, double prev1_y, double next1_x, double next1_y,
    double prev2_x, double prev2_y, double next2_x, double next2_y,
    double lo = 0.05, double hi = 0.95, int steps = 50)
{
    auto f = [&](double lam) -> double {
        double px = ux + lam * (vx - ux);
        double py = uy + lam * (vy - uy);
        return std::hypot(prev1_x - px, prev1_y - py)
             + std::hypot(px - next1_x, py - next1_y)
             + std::hypot(prev2_x - px, prev2_y - py)
             + std::hypot(px - next2_x, py - next2_y);
    };
    for (int i = 0; i < steps; ++i) {
        double m1 = lo + (hi - lo) / 3.0;
        double m2 = hi - (hi - lo) / 3.0;
        if (f(m1) < f(m2)) hi = m2;
        else lo = m1;
    }
    double best_lam = (lo + hi) / 2.0;
    return {best_lam, f(best_lam)};
}

GiantRouteSolution try_add_remove_breakpoints_cooperative(
    GiantRouteSolution gs, const Instance& inst)
{
    GiantRouteSolution best_gs = gs.copy();
    double best_cost = evaluate_gs(best_gs, inst);
    int n = (int)best_gs.giant_route.size();
    if (n < 2) return best_gs;

    auto [depot_x, depot_y] = inst.node_coord(inst.depot_idx);

    // 阶段1：对每条无断点的边，尝试多个候选 λ
    for (int ei = 0; ei < inst.num_edges; ++ei) {
        if (best_gs.breakpoints[ei].has_value()) continue;

        int se_idx = -1;
        for (int gi = 0; gi < (int)best_gs.giant_route.size(); ++gi) {
            if (best_gs.giant_route[gi].origin_edge_idx == ei
                && best_gs.giant_route[gi].seg == 0) {
                se_idx = gi; break;
            }
        }
        if (se_idx < 0) continue;

        auto [u, v] = inst.edges[ei];
        double ux = inst.x[u], uy = inst.y[u];
        double vx = inst.x[v], vy = inst.y[v];
        double edge_len = std::hypot(ux - vx, uy - vy);
        if (edge_len < 1e-9) continue;

        std::vector<double> candidate_lambdas = {0.25, 0.5, 0.75};

        // 三分搜索推荐
        double prev_cx, prev_cy, next_cx, next_cy;
        if (se_idx > 0) {
            const auto& prev_se = best_gs.giant_route[se_idx - 1];
            prev_cx = (prev_se.ax + prev_se.bx) / 2.0;
            prev_cy = (prev_se.ay + prev_se.by) / 2.0;
        } else {
            prev_cx = depot_x; prev_cy = depot_y;
        }
        if (se_idx < (int)best_gs.giant_route.size() - 1) {
            const auto& next_se = best_gs.giant_route[se_idx + 1];
            next_cx = (next_se.ax + next_se.bx) / 2.0;
            next_cy = (next_se.ay + next_se.by) / 2.0;
        } else {
            next_cx = depot_x; next_cy = depot_y;
        }

        auto [lam1, _1] = ternary_search_lambda(
            ux, uy, vx, vy,
            prev_cx, prev_cy, depot_x, depot_y,
            depot_x, depot_y, next_cx, next_cy);
        auto [lam2, _2] = ternary_search_lambda(
            ux, uy, vx, vy,
            depot_x, depot_y, next_cx, next_cy,
            prev_cx, prev_cy, depot_x, depot_y);
        candidate_lambdas.push_back(std::max(0.05, std::min(0.95, lam1)));
        candidate_lambdas.push_back(std::max(0.05, std::min(0.95, lam2)));

        for (double lam_c : candidate_lambdas) {
            double bpx = ux + lam_c * (vx - ux);
            double bpy = uy + lam_c * (vy - uy);
            SubEdge seg1_se{ei, 1, ux, uy, bpx, bpy};
            SubEdge seg2_se{ei, 2, bpx, bpy, vx, vy};

            GiantRouteSolution trial(best_gs.num_edges);
            trial.breakpoints = best_gs.breakpoints;
            trial.breakpoints[ei] = lam_c;
            trial.giant_route.clear();
            for (int gi = 0; gi < (int)best_gs.giant_route.size(); ++gi) {
                if (gi == se_idx) {
                    trial.giant_route.push_back(seg1_se);
                    trial.giant_route.push_back(seg2_se);
                } else {
                    trial.giant_route.push_back(best_gs.giant_route[gi]);
                }
            }
            try {
                double tc = evaluate_gs(trial, inst);
                if (tc < best_cost - 1e-9) {
                    best_cost = tc;
                    best_gs = trial.copy();
                    best_gs.cost_cache = tc;
                    break;
                }
            } catch (...) {}
        }
    }

    // 阶段2：对每条有断点的边，检验删除是否更优
    for (int ei = 0; ei < inst.num_edges; ++ei) {
        if (!best_gs.breakpoints[ei].has_value()) continue;

        int idx1 = -1, idx2 = -1;
        for (int gi = 0; gi < (int)best_gs.giant_route.size(); ++gi) {
            if (best_gs.giant_route[gi].origin_edge_idx == ei) {
                if (best_gs.giant_route[gi].seg == 1) idx1 = gi;
                else if (best_gs.giant_route[gi].seg == 2) idx2 = gi;
            }
        }
        if (idx1 < 0 || idx2 < 0) continue;

        auto [u, v] = inst.edges[ei];
        double ux = inst.x[u], uy = inst.y[u];
        double vx = inst.x[v], vy = inst.y[v];
        SubEdge se_whole{ei, 0, ux, uy, vx, vy};
        int keep_idx = std::min(idx1, idx2);
        int del_idx  = std::max(idx1, idx2);

        GiantRouteSolution trial(best_gs.num_edges);
        trial.breakpoints = best_gs.breakpoints;
        trial.breakpoints[ei] = std::nullopt;
        trial.giant_route = best_gs.giant_route;
        trial.giant_route[keep_idx] = se_whole;
        trial.giant_route.erase(trial.giant_route.begin() + del_idx);

        try {
            double tc = evaluate_gs(trial, inst);
            if (tc < best_cost - 1e-9) {
                best_cost = tc;
                best_gs = trial.copy();
                best_gs.cost_cache = tc;
            }
        } catch (...) {}
    }

    return best_gs;
}
