#pragma once
#include "types.h"
#include <vector>
#include <optional>
#include <string>
#include <mutex>

// ---- 解析 ----
Instance parse_instance(const std::string& filepath);

// ---- 子边工具 ----
std::vector<SubEdge> build_sub_edges(
    const Instance& inst,
    const std::vector<std::optional<double>>& breakpoints);

GiantRouteSolution rebuild_sub_edges_in_giant_route(
    const GiantRouteSolution& gs,
    const Instance& inst,
    const std::vector<std::optional<double>>& new_breakpoints);

// ---- 贪婪方向 ----
std::vector<bool> assign_directions_greedy(
    const std::vector<SubEdge>& sub_edges,
    double depot_x, double depot_y);

// ---- 费用计算 ----
double compute_route_distance(const DroneRoute& route, const Instance& inst);
std::pair<double,double> compute_route_raw_distance(const DroneRoute& route, const Instance& inst);
double compute_route_cost(const DroneRoute& route, const Instance& inst);
double compute_cost(Solution& sol, const Instance& inst);
std::pair<double,double> compute_segment_cost(
    const std::vector<SubEdge>& sub_edges,
    const Instance& inst,
    double depot_x, double depot_y);

// ---- Split DP ----
Solution split_dp(
    const std::vector<SubEdge>& giant_route,
    const Instance& inst,
    int max_drones);

Solution gs_to_solution(const GiantRouteSolution& gs, const Instance& inst);
double evaluate_gs(GiantRouteSolution& gs, const Instance& inst);

// ---- 初始解生成 ----
GiantRouteSolution generate_initial_gs(
    const Instance& inst,
    const std::vector<std::optional<double>>& breakpoints,
    const std::string& strategy = "nearest_neighbor");

GiantRouteSolution multi_start_initial_gs(const Instance& inst, int n_starts = 6);

// ---- ALNS 破坏/修复算子 ----
std::pair<GiantRouteSolution, std::vector<SubEdge>>
destroy_random_removal(const GiantRouteSolution& gs, double removal_fraction);

std::pair<GiantRouteSolution, std::vector<SubEdge>>
destroy_worst_removal(const GiantRouteSolution& gs, const Instance& inst,
                      double removal_fraction);

std::pair<GiantRouteSolution, std::vector<SubEdge>>
destroy_segment_removal(const GiantRouteSolution& gs, double removal_fraction);

GiantRouteSolution repair_greedy_insert(
    const GiantRouteSolution& gs_in,
    const Instance& inst,
    std::vector<SubEdge> removed);

GiantRouteSolution repair_random_insert(
    const GiantRouteSolution& gs_in,
    std::vector<SubEdge> removed);

GiantRouteSolution repair_regret_insert(
    const GiantRouteSolution& gs_in,
    const Instance& inst,
    std::vector<SubEdge> removed);

// ---- 局部搜索 ----
GiantRouteSolution local_search_2opt(
    GiantRouteSolution gs, const Instance& inst, int max_no_improve = -1);

GiantRouteSolution local_search_or_opt(
    GiantRouteSolution gs, const Instance& inst,
    std::vector<int> segment_sizes = {1, 2, 3},
    int max_no_improve = -1);

// ---- 断点邻域搜索 ----
GiantRouteSolution try_add_remove_breakpoints(
    GiantRouteSolution gs, const Instance& inst, bool fast_mode = false);

GiantRouteSolution try_add_remove_breakpoints_cooperative(
    GiantRouteSolution gs, const Instance& inst);

// ---- PSO ----
struct PSOBreakpointOptimizer {
    int num_particles;
    int max_iter;
    double w, c1, c2;

    PSOBreakpointOptimizer(int np = 20, int mi = 30,
                           double w = 0.7, double c1 = 1.5, double c2 = 1.5)
        : num_particles(np), max_iter(mi), w(w), c1(c1), c2(c2) {}

    GiantRouteSolution build_gs_from_mask(
        const GiantRouteSolution& gs,
        const Instance& inst,
        const std::vector<double>& bp_mask) const;

    GiantRouteSolution optimize(
        const GiantRouteSolution& gs, const Instance& inst) const;
};

// ---- ALNS 主求解器 ----
struct SolverStats {
    std::vector<std::string> destroy_names;
    std::vector<std::string> repair_names;
    std::vector<int>    destroy_calls;
    std::vector<double> destroy_time;
    std::vector<int>    destroy_impr_cur;
    std::vector<int>    destroy_impr_best;
    std::vector<int>    repair_calls;
    std::vector<double> repair_time;
    std::vector<int>    repair_impr_cur;
    std::vector<int>    repair_impr_best;
    double time_dr  = 0.0;
    double time_ls  = 0.0;
    double time_pso = 0.0;
};

struct ALNSSolveResult {
    Solution best_sol;
    double best_cost = 0.0;
    std::vector<double> cost_history;
    SolverStats stats;
};

class GiantRouteALNSSolver {
public:
    const Instance& inst;
    int max_iter;
    int segment_size;
    double removal_min;
    double removal_max;
    double sigma1, sigma2, sigma3;
    double decay;
    double sa_cooling;
    int pso_freq;
    int pso_particles;
    int pso_iter;
    int ls_freq;
    int no_improve_limit;  // -1 表示不启用

    double sa_temp = -1.0;
    PSOBreakpointOptimizer pso;

    // 算子权重和统计
    std::vector<double> destroy_weights;
    std::vector<double> destroy_scores;
    std::vector<int>    destroy_counts;
    std::vector<double> repair_weights;
    std::vector<double> repair_scores;
    std::vector<int>    repair_counts;

    // 统计
    std::vector<int>    stats_destroy_calls;
    std::vector<double> stats_destroy_time;
    std::vector<int>    stats_destroy_impr_cur;
    std::vector<int>    stats_destroy_impr_best;
    std::vector<int>    stats_repair_calls;
    std::vector<double> stats_repair_time;
    std::vector<int>    stats_repair_impr_cur;
    std::vector<int>    stats_repair_impr_best;
    double stats_time_dr  = 0.0;
    double stats_time_ls  = 0.0;
    double stats_time_pso = 0.0;

    GiantRouteALNSSolver(
        const Instance& inst,
        int max_iter = 500,
        int segment_size = 50,
        double removal_min = 0.1,
        double removal_max = 0.4,
        double sigma1 = 33.0,
        double sigma2 = 9.0,
        double sigma3 = 3.0,
        double decay = 0.8,
        double sa_cooling = 0.998,
        int pso_freq = 50,
        int pso_particles = 20,
        int pso_iter_n = 30,
        int ls_freq = 25,
        int no_improve_limit = -1);

    int roulette_select(const std::vector<double>& weights) const;
    void update_weights(int di, int ri, double score);
    void normalize_weights();
    bool sa_accept(double current_cost, double new_cost);

    // 单线程求解（原有逻辑）
    ALNSSolveResult solve(bool verbose = true);

    // 并行工作线程求解（与 SharedPool 交互）
    // 返回本线程最优解和收敛历史
    ALNSSolveResult solve_parallel_worker(
        class SharedPool& pool,
        int  thread_id,
        int  push_freq       = 50,
        int  stagnation_limit = 100,
        bool verbose         = false);
};

// ============================================================
// SharedPool - 线程安全的公共精英解池（多样性保留）
// 与 Giant_Heuristic.py::SharedPool 完全对应
// ============================================================
class SharedPool {
public:
    explicit SharedPool(int capacity = 5, double diversity_threshold = 0.8);

    // 推入解（保持多样性 + 容量限制）
    void push(const GiantRouteSolution& gs, double cost, int thread_id);

    // 拉取与 current_gs 断点结构差异最大的解（用于停滞重启）
    // 返回 {gs_copy, cost}；池空时 cost = inf，gs 为默认构造
    std::pair<GiantRouteSolution, double> pull_diverse(
        const GiantRouteSolution& current_gs);

    // 获取全局最优（不计入 pull 统计）
    std::pair<GiantRouteSolution, double> global_best() const;

    int  total_pushes()    const;
    int  total_pulls()     const;
    int  best_thread_id()  const;
    bool empty()           const;
    int  size()            const;
    std::string stats_str() const;

private:
    struct PoolEntry {
        double             cost;
        GiantRouteSolution gs;
        int                thread_id;
    };

    mutable std::mutex          _mutex;
    std::vector<PoolEntry>      _pool;
    int    _capacity;
    double _diversity_threshold;
    int    _total_pushes    = 0;
    int    _total_pulls     = 0;
    int    _best_thread_id  = -1;

    // 断点拓扑签名（有断点的边索引集合）
    static std::vector<int> gs_signature(const GiantRouteSolution& gs);
    // Jaccard 相似度
    static double jaccard(const std::vector<int>& a, const std::vector<int>& b);
};

// ============================================================
// 并行求解结果（对应 Python parallel_solve 的返回值）
// ============================================================
struct ParallelSolveResult {
    Solution                 best_sol;
    double                   best_cost      = 0.0;
    std::vector<double>      merged_history;     // 每步取所有线程最小值
    std::vector<SolverStats> per_thread_stats;   // 每线程独立统计
    int    pool_total_pushes    = 0;
    int    pool_total_pulls     = 0;
    int    pool_best_thread_id  = -1;
};

// ============================================================
// 并行求解入口（对应 Giant_Heuristic.py::parallel_solve）
// ============================================================
ParallelSolveResult parallel_solve(
    const Instance& inst,
    int  num_threads      = 4,
    int  max_iter         = 500,
    int  push_freq        = 50,
    int  stagnation_limit = 100,
    int  pool_capacity    = 5,
    int  pso_freq         = 50,
    int  pso_particles    = 20,
    int  pso_iter         = 30,
    int  ls_freq          = 25,
    int  no_improve_limit = -1,
    bool verbose          = false);
