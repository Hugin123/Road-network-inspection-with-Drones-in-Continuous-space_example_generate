/**
 * parallel.cpp
 * ============
 * 实现 SharedPool（线程安全精英解池）和 parallel_solve（异步并行求解）。
 * 与 Giant_Heuristic.py 的 SharedPool / parallel_solve 完全对应。
 *
 * 并行机制：
 *   - 同时启动 num_threads 个线程，每个线程独立运行 GiantRouteALNSSolver::solve_parallel_worker()
 *   - 每隔 push_freq 次迭代，线程将本地最优解推送到 SharedPool
 *   - 连续 stagnation_limit 次迭代未改进时，从 SharedPool 拉取多样化解重启
 *   - 所有线程完成后，取 SharedPool 全局最优作为最终解
 *   - cost_history 合并为每步取所有线程最小值（与 Python parallel_solve 一致）
 *   - per_thread_stats 每线程独立，供 RunBenchmark.py 按线程分析
 */

#include "functions.h"
#include "types.h"
#include <algorithm>
#include <numeric>
#include <thread>
#include <chrono>
#include <iostream>
#include <sstream>
#include <limits>
#include <random>
#include <set>

// 每线程独立随机引擎（在 initial_solution.cpp 中以 thread_local 定义）
extern std::mt19937& get_rng();

// 局部计时工具
static inline double now_sec() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

// ============================================================
// SharedPool 实现
// ============================================================

// 断点拓扑签名：收集所有有断点的边索引（已排序）
std::vector<int> SharedPool::gs_signature(const GiantRouteSolution& gs) {
    std::vector<int> sig;
    for (int i = 0; i < (int)gs.breakpoints.size(); ++i) {
        if (gs.breakpoints[i].has_value()) {
            sig.push_back(i);
        }
    }
    return sig;  // 已按 i 升序
}

// Jaccard 相似度（两个已排序整数集合）
double SharedPool::jaccard(const std::vector<int>& a, const std::vector<int>& b) {
    if (a.empty() && b.empty()) return 1.0;
    std::vector<int> inter, uni;
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                          std::back_inserter(inter));
    std::set_union(a.begin(), a.end(), b.begin(), b.end(),
                   std::back_inserter(uni));
    if (uni.empty()) return 1.0;
    return (double)inter.size() / (double)uni.size();
}

SharedPool::SharedPool(int capacity, double diversity_threshold)
    : _capacity(capacity), _diversity_threshold(diversity_threshold) {}

void SharedPool::push(const GiantRouteSolution& gs, double cost, int thread_id) {
    std::lock_guard<std::mutex> lk(_mutex);
    auto new_sig = gs_signature(gs);

    // 找池中结构最相似的解
    double best_sim = 0.0;
    int    best_sim_idx = -1;
    for (int i = 0; i < (int)_pool.size(); ++i) {
        double sim = jaccard(new_sig, gs_signature(_pool[i].gs));
        if (sim > best_sim) {
            best_sim     = sim;
            best_sim_idx = i;
        }
    }

    if (best_sim >= _diversity_threshold && best_sim_idx >= 0) {
        // 相似解：只在新解更好时替换
        if (cost < _pool[best_sim_idx].cost) {
            _pool[best_sim_idx] = {cost, gs, thread_id};
            std::sort(_pool.begin(), _pool.end(),
                      [](const PoolEntry& a, const PoolEntry& b){ return a.cost < b.cost; });
        }
    } else {
        // 无相似解：直接加入，淘汰最差
        _pool.push_back({cost, gs, thread_id});
        std::sort(_pool.begin(), _pool.end(),
                  [](const PoolEntry& a, const PoolEntry& b){ return a.cost < b.cost; });
        if ((int)_pool.size() > _capacity) {
            _pool.resize(_capacity);
        }
    }

    // 更新全局最优所属线程
    if (!_pool.empty()) {
        _best_thread_id = _pool[0].thread_id;
    }
    ++_total_pushes;
}

std::pair<GiantRouteSolution, double> SharedPool::pull_diverse(
    const GiantRouteSolution& current_gs)
{
    std::lock_guard<std::mutex> lk(_mutex);
    ++_total_pulls;
    if (_pool.empty()) {
        return {GiantRouteSolution{}, std::numeric_limits<double>::infinity()};
    }

    auto cur_sig = gs_signature(current_gs);

    // 找与 current_gs 断点结构差异最大（多样性最高）的解
    double best_div  = -1.0;
    int    best_idx  = 0;
    for (int i = 0; i < (int)_pool.size(); ++i) {
        double sim = jaccard(cur_sig, gs_signature(_pool[i].gs));
        double div = 1.0 - sim;
        if (div > best_div) {
            best_div = div;
            best_idx = i;
        }
    }
    // 若多样性改进不明显，回退到全局最优（idx=0）
    if (best_div < 0.2) {
        best_idx = 0;
    }
    return {_pool[best_idx].gs, _pool[best_idx].cost};
}

std::pair<GiantRouteSolution, double> SharedPool::global_best() const {
    std::lock_guard<std::mutex> lk(_mutex);
    if (_pool.empty()) {
        return {GiantRouteSolution{}, std::numeric_limits<double>::infinity()};
    }
    return {_pool[0].gs, _pool[0].cost};
}

int  SharedPool::total_pushes()   const { std::lock_guard<std::mutex> lk(_mutex); return _total_pushes; }
int  SharedPool::total_pulls()    const { std::lock_guard<std::mutex> lk(_mutex); return _total_pulls; }
int  SharedPool::best_thread_id() const { std::lock_guard<std::mutex> lk(_mutex); return _best_thread_id; }
bool SharedPool::empty()          const { std::lock_guard<std::mutex> lk(_mutex); return _pool.empty(); }
int  SharedPool::size()           const { std::lock_guard<std::mutex> lk(_mutex); return (int)_pool.size(); }

std::string SharedPool::stats_str() const {
    std::lock_guard<std::mutex> lk(_mutex);
    std::ostringstream oss;
    double best = _pool.empty() ? std::numeric_limits<double>::infinity() : _pool[0].cost;
    oss << "SharedPool: size=" << _pool.size() << "/" << _capacity
        << ", best=" << best
        << ", pushes=" << _total_pushes
        << ", pulls="  << _total_pulls
        << ", best_thread=" << _best_thread_id;
    return oss.str();
}


// ============================================================
// GiantRouteALNSSolver::solve_parallel_worker
// （在 parallel.cpp 中实现，逻辑与 Giant_Heuristic.py 完全对应）
// ============================================================
ALNSSolveResult GiantRouteALNSSolver::solve_parallel_worker(
    SharedPool& pool,
    int  thread_id,
    int  push_freq,
    int  stagnation_limit,
    bool verbose)
{
    // ---- 初始解 ----
    GiantRouteSolution current_gs = multi_start_initial_gs(inst, 6);
    current_gs = try_add_remove_breakpoints(current_gs, inst, false);

    double current_cost = evaluate_gs(current_gs, inst);
    GiantRouteSolution best_gs = current_gs.copy();
    double best_cost = current_cost;

    if (sa_temp < 0) {
        sa_temp = current_cost * 0.05;
    }

    std::vector<double> best_cost_history = {best_cost};

    int stagnation_count = 0;  // 连续未改进计数（用于停滞重启）
    int no_improve_count = 0;  // 早停计数

    if (verbose) {
        std::cerr << "  [T" << thread_id << "] 初始解费用: " << current_cost << "\n";
    }

    // 复用父类成员中的随机数器（每线程 solver 实例独立，rng 也独立）

    std::uniform_real_distribution<double> uni_rf(removal_min, removal_max);

    for (int iteration = 0; iteration < max_iter; ++iteration) {
        double removal_fraction = uni_rf(get_rng());

        int di_op = roulette_select(destroy_weights);
        int ri_op = roulette_select(repair_weights);

        double t0 = now_sec();
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

        double t_destroy = now_sec();

        // 修复
        GiantRouteSolution new_gs;
        if (ri_op == 0)      new_gs = repair_greedy_insert(destroyed_gs, inst, removed);
        else if (ri_op == 1) new_gs = repair_random_insert(destroyed_gs, removed);
        else                 new_gs = repair_regret_insert(destroyed_gs, inst, removed);

        double t_repair = now_sec();

        stats_destroy_calls[di_op]++;
        stats_destroy_time[di_op]  += t_destroy - t0;
        stats_repair_calls[ri_op]++;
        stats_repair_time[ri_op]   += t_repair - t_destroy;
        stats_time_dr += t_repair - t0;

        // 局部搜索（每 ls_freq 次）
        if ((iteration + 1) % ls_freq == 0) {
            double t_ls0 = now_sec();
            new_gs = local_search_or_opt(new_gs, inst, {1, 2}, 3);
            stats_time_ls += now_sec() - t_ls0;
        }

        // PSO（每 pso_freq 次）+ 断点协同搜索
        if ((iteration + 1) % pso_freq == 0) {
            double t_pso0 = now_sec();
            new_gs = pso.optimize(new_gs, inst);
            stats_time_pso += now_sec() - t_pso0;

            double t_ls1 = now_sec();
            new_gs = try_add_remove_breakpoints_cooperative(new_gs, inst);
            new_gs = try_add_remove_breakpoints(new_gs, inst, false);
            stats_time_ls += now_sec() - t_ls1;
        }

        double new_cost;
        bool eval_ok = true;
        try {
            new_cost = evaluate_gs(new_gs, inst);
        } catch (...) {
            eval_ok = false;
        }

        if (!eval_ok) {
            best_cost_history.push_back(best_cost);
            stagnation_count++;
            no_improve_count++;
            continue;
        }

        double score     = 0.0;
        bool impr_best   = false;
        bool impr_cur    = false;

        if (new_cost < best_cost - 1e-9) {
            best_gs        = new_gs.copy();
            best_cost      = new_cost;
            current_gs     = new_gs.copy();
            current_cost   = new_cost;
            score          = sigma1;
            stagnation_count = 0;
            no_improve_count = 0;
            impr_best = impr_cur = true;
        } else if (new_cost < current_cost - 1e-9) {
            current_gs   = new_gs.copy();
            current_cost = new_cost;
            score        = sigma2;
            stagnation_count++;
            no_improve_count++;
            impr_cur = true;
        } else if (sa_accept(current_cost, new_cost)) {
            current_gs   = new_gs.copy();
            current_cost = new_cost;
            score        = sigma3;
            stagnation_count++;
            no_improve_count++;
        } else {
            stagnation_count++;
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
                std::cerr << "  [T" << thread_id << "] Iter " << (iteration + 1)
                          << "/" << max_iter
                          << " | best: " << best_cost << "\n";
            }
        }

        best_cost_history.push_back(best_cost);

        // ---- 推送：每 push_freq 次迭代把本线程最优解推入公共池 ----
        if ((iteration + 1) % push_freq == 0) {
            pool.push(best_gs, best_cost, thread_id);
            if (verbose) {
                std::cerr << "  [T" << thread_id << "] Iter " << (iteration + 1)
                          << " push cost=" << best_cost << "\n";
            }
        }

        // ---- 拉取：停滞超过 stagnation_limit 次时从公共池拉取多样化解 ----
        if (stagnation_count >= stagnation_limit) {
            auto [pulled_gs, pulled_cost] = pool.pull_diverse(best_gs);
            if (pulled_cost < std::numeric_limits<double>::infinity()) {
                // 以拉取解做断点精化后重启
                try {
                    GiantRouteSolution restarted_gs =
                        try_add_remove_breakpoints(pulled_gs.copy(), inst, false);
                    double restarted_cost = evaluate_gs(restarted_gs, inst);
                    current_gs   = restarted_gs.copy();
                    current_cost = restarted_cost;
                    if (restarted_cost < best_cost - 1e-9) {
                        best_gs   = restarted_gs.copy();
                        best_cost = restarted_cost;
                        no_improve_count = 0;
                    }
                    if (verbose) {
                        std::cerr << "  [T" << thread_id << "] Iter " << (iteration + 1)
                                  << " pull_diverse → restart=" << current_cost << "\n";
                    }
                } catch (...) {}
            }
            stagnation_count = 0;
        }

        // ---- 早停检测 ----
        if (no_improve_limit > 0 && no_improve_count >= no_improve_limit) {
            if (verbose) {
                std::cerr << "  [T" << thread_id << "][早停] 连续 " << no_improve_count
                          << " 次迭代全局最优未改进，第 " << (iteration + 1)
                          << " 次迭代提前终止\n";
            }
            break;
        }
    }

    // ---- 最终精化 ----
    auto try_update = [&](GiantRouteSolution& candidate) {
        try {
            double fc = evaluate_gs(candidate, inst);
            if (fc < best_cost - 1e-9) {
                best_gs   = candidate.copy();
                best_cost = fc;
            }
        } catch (...) {}
    };

    auto final1 = local_search_2opt(best_gs, inst, 2);
    final1 = local_search_or_opt(final1, inst, {1, 2, 3}, 3);
    try_update(final1);

    auto final2 = try_add_remove_breakpoints(best_gs, inst, false);
    try_update(final2);

    auto final3 = pso.optimize(best_gs, inst);
    try_update(final3);

    auto final4 = try_add_remove_breakpoints_cooperative(best_gs, inst);
    try_update(final4);

    auto final5 = try_add_remove_breakpoints(best_gs, inst, false);
    try_update(final5);

    // 最终推送一次
    pool.push(best_gs, best_cost, thread_id);

    if (verbose) {
        std::cerr << "  [T" << thread_id << "] 完成，最优费用: " << best_cost << "\n";
    }

    // 构建返回结果
    ALNSSolveResult result;
    Solution best_sol = gs_to_solution(best_gs, inst);
    best_sol.breakpoints = best_gs.breakpoints;
    result.best_sol   = best_sol;
    result.best_cost  = best_cost;
    result.cost_history = best_cost_history;

    result.stats.destroy_names     = {"random", "worst", "segment"};
    result.stats.repair_names      = {"greedy", "random", "regret"};
    result.stats.destroy_calls     = stats_destroy_calls;
    result.stats.destroy_time      = stats_destroy_time;
    result.stats.destroy_impr_cur  = stats_destroy_impr_cur;
    result.stats.destroy_impr_best = stats_destroy_impr_best;
    result.stats.repair_calls      = stats_repair_calls;
    result.stats.repair_time       = stats_repair_time;
    result.stats.repair_impr_cur   = stats_repair_impr_cur;
    result.stats.repair_impr_best  = stats_repair_impr_best;
    result.stats.time_dr  = stats_time_dr;
    result.stats.time_ls  = stats_time_ls;
    result.stats.time_pso = stats_time_pso;

    return result;
}


// ============================================================
// parallel_solve：启动多线程并发求解，聚合结果
// 与 Giant_Heuristic.py::parallel_solve 完全对应
// ============================================================
ParallelSolveResult parallel_solve(
    const Instance& inst,
    int  num_threads,
    int  max_iter,
    int  push_freq,
    int  stagnation_limit,
    int  pool_capacity,
    int  pso_freq,
    int  pso_particles,
    int  pso_iter,
    int  ls_freq,
    int  no_improve_limit,
    bool verbose)
{
    if (verbose) {
        std::cerr << "\n============================================================\n"
                  << "异步并行 Giant Route ALNS+PSO 启动\n"
                  << "  线程数: " << num_threads
                  << "  迭代数/线程: " << max_iter << "\n"
                  << "  推送频率: 每 " << push_freq << " 次迭代"
                  << "  停滞阈值: " << stagnation_limit << " 次\n"
                  << "  公共池容量: " << pool_capacity << "\n"
                  << "============================================================\n";
    }

    SharedPool pool(pool_capacity);

    // 每个线程的结果存储
    std::vector<ALNSSolveResult> thread_results(num_threads);
    std::vector<std::thread>     threads;

    // 注意：每个线程需要独立的 solver 实例（不共享状态）
    // 用 unique_ptr 避免 solver 被拷贝
    std::vector<std::unique_ptr<GiantRouteALNSSolver>> solvers;
    solvers.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        solvers.push_back(std::make_unique<GiantRouteALNSSolver>(
            inst,
            max_iter,
            50,       // segment_size
            0.1, 0.4, // removal_min/max
            33.0, 9.0, 3.0, // sigma1/2/3
            0.8,      // decay
            0.998,    // sa_cooling
            pso_freq,
            pso_particles,
            pso_iter,
            ls_freq,
            no_improve_limit
        ));
    }

    // 启动所有线程
    for (int tid = 0; tid < num_threads; ++tid) {
        threads.emplace_back([&, tid]() {
            thread_results[tid] = solvers[tid]->solve_parallel_worker(
                pool, tid, push_freq, stagnation_limit, verbose);
            if (verbose) {
                std::cerr << "  [线程 " << (tid + 1) << "/" << num_threads
                          << "] 完成，本地最优: "
                          << thread_results[tid].best_cost << "\n";
            }
        });
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }

    if (verbose) {
        std::cerr << "\n" << pool.stats_str() << "\n";
    }

    // ---- 取全局最优解 ----
    auto [best_gs, best_cost] = pool.global_best();
    Solution best_sol;
    if (best_cost < std::numeric_limits<double>::infinity()) {
        best_sol = gs_to_solution(best_gs, inst);
        best_sol.breakpoints = best_gs.breakpoints;
    } else {
        // 回退到各线程本地最优（理论上不应发生）
        best_cost = std::numeric_limits<double>::infinity();
        for (int tid = 0; tid < num_threads; ++tid) {
            if (thread_results[tid].best_cost < best_cost) {
                best_cost = thread_results[tid].best_cost;
                best_sol  = thread_results[tid].best_sol;
            }
        }
    }

    // ---- 合并 cost_history：每步取所有线程的最小值 ----
    std::vector<std::vector<double>> all_hists;
    for (int tid = 0; tid < num_threads; ++tid) {
        if (!thread_results[tid].cost_history.empty()) {
            all_hists.push_back(thread_results[tid].cost_history);
        }
    }

    std::vector<double> merged_history;
    if (!all_hists.empty()) {
        size_t max_len = 0;
        for (auto& h : all_hists) max_len = std::max(max_len, h.size());

        // 末尾填充（补齐到最长长度）
        for (auto& h : all_hists) {
            while (h.size() < max_len) h.push_back(h.back());
        }

        merged_history.resize(max_len);
        for (size_t i = 0; i < max_len; ++i) {
            double mn = std::numeric_limits<double>::infinity();
            for (auto& h : all_hists) mn = std::min(mn, h[i]);
            merged_history[i] = mn;
        }
    }

    // ---- 收集每个线程的统计信息 ----
    std::vector<SolverStats> per_thread_stats;
    per_thread_stats.reserve(num_threads);
    for (int tid = 0; tid < num_threads; ++tid) {
        per_thread_stats.push_back(thread_results[tid].stats);
    }

    if (verbose) {
        std::cerr << "并行求解完成！全局最优费用: " << best_cost << "\n";
    }

    ParallelSolveResult result;
    result.best_sol             = best_sol;
    result.best_cost            = best_cost;
    result.merged_history       = merged_history;
    result.per_thread_stats     = per_thread_stats;
    result.pool_total_pushes    = pool.total_pushes();
    result.pool_total_pulls     = pool.total_pulls();
    result.pool_best_thread_id  = pool.best_thread_id();

    return result;
}
