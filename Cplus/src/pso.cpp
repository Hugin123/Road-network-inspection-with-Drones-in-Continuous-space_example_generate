#include "functions.h"
#include "types.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>
#include <vector>
#include <iostream>

extern std::mt19937& get_rng();

// ALNS 算子和局部搜索函数已在 functions.h 中声明

// ============================================================
// 12. PSO 优化断点位置（二层遗传/断点存在性优化）
// ============================================================

GiantRouteSolution PSOBreakpointOptimizer::build_gs_from_mask(
    const GiantRouteSolution& gs,
    const Instance& inst,
    const std::vector<double>& bp_mask) const
{
    GiantRouteSolution new_gs = gs.copy();
    for (int ei = 0; ei < inst.num_edges; ++ei) {
        bool curr_has_bp = new_gs.breakpoints[ei].has_value();
        bool want_bp = (bp_mask[ei] > 0.5);

        if (curr_has_bp && !want_bp) {
            // 合并：找到 seg1/seg2，替换为整边
            int idx1 = -1, idx2 = -1;
            for (int gi = 0; gi < (int)new_gs.giant_route.size(); ++gi) {
                if (new_gs.giant_route[gi].origin_edge_idx == ei) {
                    if (new_gs.giant_route[gi].seg == 1) idx1 = gi;
                    else if (new_gs.giant_route[gi].seg == 2) idx2 = gi;
                }
            }
            if (idx1 >= 0 && idx2 >= 0) {
                auto [u_node, v_node] = inst.edges[ei];
                double ux = inst.x[u_node], uy = inst.y[u_node];
                double vx = inst.x[v_node], vy = inst.y[v_node];
                SubEdge se_whole{ei, 0, ux, uy, vx, vy};
                int keep_idx = std::min(idx1, idx2);
                int del_idx  = std::max(idx1, idx2);
                new_gs.giant_route[keep_idx] = se_whole;
                new_gs.giant_route.erase(new_gs.giant_route.begin() + del_idx);
                new_gs.breakpoints[ei] = std::nullopt;
            }
        } else if (!curr_has_bp && want_bp) {
            // 分裂：找到 seg=0，替换为 seg1+seg2（λ=0.5）
            int se_idx = -1;
            for (int gi = 0; gi < (int)new_gs.giant_route.size(); ++gi) {
                if (new_gs.giant_route[gi].origin_edge_idx == ei
                    && new_gs.giant_route[gi].seg == 0) {
                    se_idx = gi; break;
                }
            }
            if (se_idx >= 0) {
                auto [u_node, v_node] = inst.edges[ei];
                double ux = inst.x[u_node], uy = inst.y[u_node];
                double vx = inst.x[v_node], vy = inst.y[v_node];
                double lam = 0.5;
                double bpx = ux + lam * (vx - ux);
                double bpy = uy + lam * (vy - uy);
                SubEdge seg1_se{ei, 1, ux, uy, bpx, bpy};
                SubEdge seg2_se{ei, 2, bpx, bpy, vx, vy};
                // 替换 se_idx 处
                new_gs.giant_route[se_idx] = seg1_se;
                new_gs.giant_route.insert(
                    new_gs.giant_route.begin() + se_idx + 1, seg2_se);
                new_gs.breakpoints[ei] = lam;
            }
        }
    }
    new_gs.cost_cache = std::nullopt;
    return new_gs;
}

GiantRouteSolution PSOBreakpointOptimizer::optimize(
    const GiantRouteSolution& gs, const Instance& inst) const
{
    int E = inst.num_edges;
    GiantRouteSolution mutable_gs = gs;
    double current_cost = evaluate_gs(mutable_gs, inst);

    // 初始化种群（二值掩码）
    std::vector<std::vector<double>> population(num_particles, std::vector<double>(E));
    for (int e = 0; e < E; ++e)
        population[0][e] = gs.breakpoints[e].has_value() ? 1.0 : 0.0;

    std::uniform_real_distribution<double> uni_flip(0.0, 1.0);
    std::uniform_int_distribution<int> uni_e(0, E - 1);
    std::uniform_int_distribution<int> uni_nflip(1, std::max(1, std::min(3, E)));

    // 其余粒子随机翻转少量位
    for (int i = 1; i < num_particles; ++i) {
        population[i] = population[0];
        int n_flip = std::uniform_int_distribution<int>(
            1, std::max(1, E / 4))(get_rng());
        n_flip = std::min(n_flip, E);
        std::vector<int> idx_pool(E);
        for (int e = 0; e < E; ++e) idx_pool[e] = e;
        std::shuffle(idx_pool.begin(), idx_pool.end(), get_rng());
        for (int f = 0; f < n_flip; ++f)
            population[i][idx_pool[f]] = 1.0 - population[i][idx_pool[f]];
    }

    // 最后 1/5 粒子完全随机
    int n_rand = std::max(2, num_particles / 5);
    for (int i = num_particles - n_rand; i < num_particles; ++i) {
        for (int e = 0; e < E; ++e)
            population[i][e] = (uni_flip(get_rng()) < 0.3) ? 1.0 : 0.0;
    }

    // 评估初始适应度
    std::vector<std::vector<double>> pbest_mask = population;
    std::vector<double> pbest_cost(num_particles, std::numeric_limits<double>::infinity());
    std::vector<std::optional<GiantRouteSolution>> pbest_gs_list(num_particles);

    for (int i = 0; i < num_particles; ++i) {
        GiantRouteSolution trial_gs = build_gs_from_mask(gs, inst, population[i]);
        try {
            double cost_i = evaluate_gs(trial_gs, inst);
            pbest_cost[i] = cost_i;
            pbest_gs_list[i] = trial_gs;
        } catch (...) {}
    }

    int gbest_idx = (int)(std::min_element(pbest_cost.begin(), pbest_cost.end())
                          - pbest_cost.begin());
    std::vector<double> gbest_mask = pbest_mask[gbest_idx];
    double gbest_cost = pbest_cost[gbest_idx];
    std::optional<GiantRouteSolution> gbest_gs = pbest_gs_list[gbest_idx];

    // 外层遗传主循环
    for (int iter = 0; iter < max_iter; ++iter) {
        for (int i = 0; i < num_particles; ++i) {
            std::vector<double> new_mask = pbest_mask[i];

            // 随机翻转 1~3 位
            int n_flip = std::uniform_int_distribution<int>(
                1, std::max(1, std::min(3, E)))(get_rng());
            std::vector<int> idx_pool(E);
            for (int e = 0; e < E; ++e) idx_pool[e] = e;
            std::shuffle(idx_pool.begin(), idx_pool.end(), get_rng());
            for (int f = 0; f < n_flip; ++f)
                new_mask[idx_pool[f]] = 1.0 - new_mask[idx_pool[f]];

            // 30% 概率借用全局最优的一个位
            if (uni_flip(get_rng()) < 0.3 && E > 0) {
                int borrow_idx = uni_e(get_rng());
                new_mask[borrow_idx] = gbest_mask[borrow_idx];
            }

            GiantRouteSolution trial_gs = build_gs_from_mask(gs, inst, new_mask);
            try {
                double cost_i = evaluate_gs(trial_gs, inst);
                if (cost_i < pbest_cost[i]) {
                    pbest_cost[i] = cost_i;
                    pbest_mask[i] = new_mask;
                    pbest_gs_list[i] = trial_gs;
                    if (cost_i < gbest_cost) {
                        gbest_cost = cost_i;
                        gbest_mask = new_mask;
                        gbest_gs = trial_gs;
                    }
                }
            } catch (...) {}
        }
    }

    if (gbest_cost < current_cost - 1e-9 && gbest_gs.has_value()) {
        GiantRouteSolution result = *gbest_gs;
        result.cost_cache = gbest_cost;
        return result;
    }
    return gs;
}

// ============================================================
// 13. ALNS 主框架
// ============================================================

GiantRouteALNSSolver::GiantRouteALNSSolver(
    const Instance& inst,
    int max_iter,
    int segment_size,
    double removal_min,
    double removal_max,
    double sigma1,
    double sigma2,
    double sigma3,
    double decay,
    double sa_cooling,
    int pso_freq,
    int pso_particles,
    int pso_iter_n,
    int ls_freq,
    int no_improve_limit)
    : inst(inst),
      max_iter(max_iter),
      segment_size(segment_size),
      removal_min(removal_min),
      removal_max(removal_max),
      sigma1(sigma1),
      sigma2(sigma2),
      sigma3(sigma3),
      decay(decay),
      sa_cooling(sa_cooling),
      pso_freq(pso_freq),
      pso_particles(pso_particles),
      pso_iter(pso_iter_n),
      ls_freq(ls_freq),
      no_improve_limit(no_improve_limit),
      pso(pso_particles, pso_iter_n)
{
    // 3 个破坏算子，3 个修复算子
    destroy_weights = {1.0, 1.0, 1.0};
    destroy_scores  = {0.0, 0.0, 0.0};
    destroy_counts  = {0, 0, 0};
    repair_weights  = {1.0, 1.0, 1.0};
    repair_scores   = {0.0, 0.0, 0.0};
    repair_counts   = {0, 0, 0};

    stats_destroy_calls     = {0, 0, 0};
    stats_destroy_time      = {0.0, 0.0, 0.0};
    stats_destroy_impr_cur  = {0, 0, 0};
    stats_destroy_impr_best = {0, 0, 0};
    stats_repair_calls      = {0, 0, 0};
    stats_repair_time       = {0.0, 0.0, 0.0};
    stats_repair_impr_cur   = {0, 0, 0};
    stats_repair_impr_best  = {0, 0, 0};
}

int GiantRouteALNSSolver::roulette_select(const std::vector<double>& weights) const {
    double total = 0.0;
    for (double w : weights) total += w;
    if (total <= 0.0)
        return std::uniform_int_distribution<int>(0, (int)weights.size() - 1)(get_rng());
    std::uniform_real_distribution<double> uni(0.0, total);
    double r = uni(get_rng());
    double cumsum = 0.0;
    for (int i = 0; i < (int)weights.size(); ++i) {
        cumsum += weights[i];
        if (r <= cumsum) return i;
    }
    return (int)weights.size() - 1;
}

void GiantRouteALNSSolver::update_weights(int di, int ri, double score) {
    destroy_scores[di] += score;
    destroy_counts[di] += 1;
    repair_scores[ri] += score;
    repair_counts[ri] += 1;
}

void GiantRouteALNSSolver::normalize_weights() {
    for (int i = 0; i < (int)destroy_weights.size(); ++i) {
        if (destroy_counts[i] > 0) {
            double new_score = destroy_scores[i] / destroy_counts[i];
            destroy_weights[i] = (1 - decay) * destroy_weights[i] + decay * new_score;
        }
        destroy_weights[i] = std::max(0.1, destroy_weights[i]);
        destroy_scores[i] = 0.0;
        destroy_counts[i] = 0;
    }
    for (int i = 0; i < (int)repair_weights.size(); ++i) {
        if (repair_counts[i] > 0) {
            double new_score = repair_scores[i] / repair_counts[i];
            repair_weights[i] = (1 - decay) * repair_weights[i] + decay * new_score;
        }
        repair_weights[i] = std::max(0.1, repair_weights[i]);
        repair_scores[i] = 0.0;
        repair_counts[i] = 0;
    }
}

bool GiantRouteALNSSolver::sa_accept(double current_cost, double new_cost) {
    if (new_cost <= current_cost) return true;
    if (sa_temp <= 0) return false;
    double delta = new_cost - current_cost;
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    return uni(get_rng()) < std::exp(-delta / sa_temp);
}

// 获取精确时间（秒）
#include <chrono>
static double now_seconds() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

ALNSSolveResult GiantRouteALNSSolver::solve(bool verbose) {
    // ---- 初始解 ----
    if (verbose) std::cerr << "多启动策略生成初始 Giant Route 解...\n";
    GiantRouteSolution current_gs = multi_start_initial_gs(inst, 6);

    if (verbose) std::cerr << "对初始解执行断点邻域搜索...\n";
    current_gs = try_add_remove_breakpoints(current_gs, inst, false);

    double current_cost = evaluate_gs(current_gs, inst);
    GiantRouteSolution best_gs = current_gs.copy();
    double best_cost = current_cost;

    if (sa_temp < 0) {
        sa_temp = current_cost * 0.05;
        if (verbose) std::cerr << "SA 初始温度: " << sa_temp << "\n";
    }

    if (verbose) std::cerr << "初始解费用: " << current_cost << "\n";

    std::vector<double> cost_history = {current_cost};
    double start_time = now_seconds();
    int no_improve_count = 0;

    std::uniform_real_distribution<double> uni_rf(removal_min, removal_max);

    for (int iteration = 0; iteration < max_iter; ++iteration) {
        double removal_fraction = uni_rf(get_rng());

        // 选择算子
        int di_op = roulette_select(destroy_weights);
        int ri_op = roulette_select(repair_weights);

        double t0 = now_seconds();
        GiantRouteSolution destroyed_gs;
        std::vector<SubEdge> removed;

        // 破坏
        if (di_op == 0) {
            auto [d, r] = destroy_random_removal(current_gs, removal_fraction);
            destroyed_gs = d; removed = r;
        } else if (di_op == 1) {
            auto [d, r] = destroy_worst_removal(current_gs, inst, removal_fraction);
            destroyed_gs = d; removed = r;
        } else {
            auto [d, r] = destroy_segment_removal(current_gs, removal_fraction);
            destroyed_gs = d; removed = r;
        }

        double t_destroy = now_seconds();

        // 修复
        GiantRouteSolution new_gs;
        if (ri_op == 0) {
            new_gs = repair_greedy_insert(destroyed_gs, inst, removed);
        } else if (ri_op == 1) {
            new_gs = repair_random_insert(destroyed_gs, removed);
        } else {
            new_gs = repair_regret_insert(destroyed_gs, inst, removed);
        }

        double t_repair = now_seconds();

        stats_destroy_calls[di_op]++;
        stats_destroy_time[di_op] += t_destroy - t0;
        stats_repair_calls[ri_op]++;
        stats_repair_time[ri_op] += t_repair - t_destroy;
        stats_time_dr += t_repair - t0;

        // 局部搜索（or-opt，每 ls_freq 次）
        if ((iteration + 1) % ls_freq == 0) {
            double t_ls0 = now_seconds();
            new_gs = local_search_or_opt(new_gs, inst, {1, 2}, 3);
            stats_time_ls += now_seconds() - t_ls0;
        }

        // PSO（每 pso_freq 次）+ 断点协同搜索
        if ((iteration + 1) % pso_freq == 0) {
            double t_pso0 = now_seconds();
            new_gs = pso.optimize(new_gs, inst);
            stats_time_pso += now_seconds() - t_pso0;

            double t_ls1 = now_seconds();
            new_gs = try_add_remove_breakpoints_cooperative(new_gs, inst);
            new_gs = try_add_remove_breakpoints(new_gs, inst, false);
            stats_time_ls += now_seconds() - t_ls1;
        }

        double new_cost;
        bool eval_ok = true;
        try {
            new_cost = evaluate_gs(new_gs, inst);
        } catch (...) {
            eval_ok = false;
            if (verbose && (iteration + 1) % 50 == 0)
                std::cerr << "  Iter " << iteration + 1 << ": 评估失败，跳过\n";
        }

        if (!eval_ok) {
            cost_history.push_back(current_cost);
            continue;
        }

        double score = 0.0;
        bool impr_best = false, impr_cur = false;

        if (new_cost < best_cost - 1e-9) {
            best_gs = new_gs.copy();
            best_cost = new_cost;
            current_gs = new_gs.copy();
            current_cost = new_cost;
            score = sigma1;
            impr_best = impr_cur = true;
            no_improve_count = 0;
        } else if (new_cost < current_cost - 1e-9) {
            current_gs = new_gs.copy();
            current_cost = new_cost;
            score = sigma2;
            impr_cur = true;
            no_improve_count++;
        } else if (sa_accept(current_cost, new_cost)) {
            current_gs = new_gs.copy();
            current_cost = new_cost;
            score = sigma3;
            no_improve_count++;
        } else {
            no_improve_count++;
        }

        if (impr_best) {
            stats_destroy_impr_best[di_op]++;
            stats_repair_impr_best[ri_op]++;
        }
        if (impr_cur) {
            stats_destroy_impr_cur[di_op]++;
            stats_repair_impr_cur[ri_op]++;
        }

        update_weights(di_op, ri_op, score);
        sa_temp *= sa_cooling;

        if ((iteration + 1) % segment_size == 0) {
            normalize_weights();
            if (verbose) {
                double elapsed = now_seconds() - start_time;
                std::cerr << "  Iter " << iteration + 1 << "/" << max_iter
                          << " | current: " << current_cost
                          << " | best: " << best_cost
                          << " | T: " << sa_temp
                          << " | time: " << elapsed << "s\n";
            }
        }

        cost_history.push_back(best_cost);

        // 早停
        if (no_improve_limit > 0 && no_improve_count >= no_improve_limit) {
            if (verbose) {
                std::cerr << "  [早停] 连续 " << no_improve_count
                          << " 次迭代全局最优未改进，第 " << iteration + 1
                          << " 次迭代提前终止\n";
            }
            break;
        }
    }

    // ---- 最终精化 ----
    if (verbose) std::cerr << "最终精化：2-opt + or-opt + 断点邻域搜索 + PSO...\n";

    auto try_update = [&](GiantRouteSolution& candidate) {
        try {
            double fc = evaluate_gs(candidate, inst);
            if (fc < best_cost - 1e-9) {
                best_gs = candidate.copy();
                best_cost = fc;
            }
        } catch (...) {}
    };

    // 第一轮：2-opt + or-opt
    auto final1 = local_search_2opt(best_gs, inst, 2);
    final1 = local_search_or_opt(final1, inst, {1, 2, 3}, 3);
    try_update(final1);

    // 第二轮：断点邻域搜索
    auto final2 = try_add_remove_breakpoints(best_gs, inst, false);
    try_update(final2);

    // 第三轮：PSO 精化
    auto final3 = pso.optimize(best_gs, inst);
    try_update(final3);

    // 第四轮：断点协同搜索
    auto final4 = try_add_remove_breakpoints_cooperative(best_gs, inst);
    try_update(final4);

    // 第五轮：传统邻域搜索
    auto final5 = try_add_remove_breakpoints(best_gs, inst, false);
    try_update(final5);

    if (verbose) std::cerr << "求解完成！最优费用: " << best_cost << "\n";

    // 构建返回结果
    ALNSSolveResult result;
    Solution best_sol = gs_to_solution(best_gs, inst);
    best_sol.breakpoints = best_gs.breakpoints;
    result.best_sol = best_sol;
    result.best_cost = best_cost;
    result.cost_history = cost_history;

    result.stats.destroy_names = {"random", "worst", "segment"};
    result.stats.repair_names  = {"greedy", "random", "regret"};
    result.stats.destroy_calls      = stats_destroy_calls;
    result.stats.destroy_time       = stats_destroy_time;
    result.stats.destroy_impr_cur   = stats_destroy_impr_cur;
    result.stats.destroy_impr_best  = stats_destroy_impr_best;
    result.stats.repair_calls       = stats_repair_calls;
    result.stats.repair_time        = stats_repair_time;
    result.stats.repair_impr_cur    = stats_repair_impr_cur;
    result.stats.repair_impr_best   = stats_repair_impr_best;
    result.stats.time_dr  = stats_time_dr;
    result.stats.time_ls  = stats_time_ls;
    result.stats.time_pso = stats_time_pso;

    return result;
}
