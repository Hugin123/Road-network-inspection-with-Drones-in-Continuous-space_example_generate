#include "types.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cctype>

// ============================================================
// 文件解析：从 txt 文件解析算例（对齐 Python parse_instance）
// ============================================================

static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

static bool is_edge_line(const std::string& s) {
    // 包含 ',' 或 '(' 的行视为边数据行
    return (s.find(',') != std::string::npos || s.find('(') != std::string::npos);
}

Instance parse_instance(const std::string& filepath) {
    std::ifstream f(filepath);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    std::vector<std::string> non_empty;
    std::string line;
    while (std::getline(f, line)) {
        std::string t = trim(line);
        if (!t.empty()) {
            non_empty.push_back(t);
        }
    }
    f.close();

    int idx = 0;
    auto next_line = [&]() -> std::string {
        if (idx >= (int)non_empty.size())
            throw std::runtime_error("Unexpected end of file");
        return non_empty[idx++];
    };

    Instance inst;
    inst.num_depots     = std::stoi(next_line());
    inst.num_road_nodes = std::stoi(next_line());
    inst.total_nodes    = std::stoi(next_line());
    inst.num_edges      = std::stoi(next_line());
    inst.num_drones     = std::stoi(next_line());
    inst.battery        = std::stod(next_line());
    inst.speed          = std::stod(next_line());
    inst.energy_cost    = std::stod(next_line());
    inst.call_cost      = std::stod(next_line());
    inst.inspect_coef   = std::stod(next_line());
    inst.transfer_coef  = std::stod(next_line());

    inst.big_m = 10000.0;

    // 读取可能的额外参数（类似 Python：非逗号/括号的浮点行）
    std::vector<double> extra_params;
    while (idx < (int)non_empty.size()) {
        const std::string& l = non_empty[idx];
        if (is_edge_line(l)) break;
        // 尝试解析为浮点数
        try {
            double val = std::stod(l);
            extra_params.push_back(val);
            idx++;
        } catch (...) {
            break;
        }
    }
    if (extra_params.size() == 1) {
        inst.big_m = extra_params[0];
    } else if (extra_params.size() >= 2) {
        inst.big_m = extra_params.back();
    }

    // 读取 x 坐标（逗号分隔）
    {
        std::string xline = next_line();
        std::stringstream ss(xline);
        std::string token;
        while (std::getline(ss, token, ',')) {
            inst.x.push_back(std::stod(trim(token)));
        }
    }

    // 读取 y 坐标（逗号分隔）
    {
        std::string yline = next_line();
        std::stringstream ss(yline);
        std::string token;
        while (std::getline(ss, token, ',')) {
            inst.y.push_back(std::stod(trim(token)));
        }
    }

    // 读取边（格式：(u,v) 或 u,v）
    while (idx < (int)non_empty.size()) {
        std::string token = non_empty[idx++];
        // 去除括号
        std::string clean;
        for (char c : token) {
            if (c != '(' && c != ')') clean += c;
        }
        clean = trim(clean);
        if (clean.empty()) continue;

        std::stringstream ss(clean);
        std::string part;
        std::vector<int> parts;
        while (std::getline(ss, part, ',')) {
            parts.push_back(std::stoi(trim(part)));
        }
        if (parts.size() >= 2) {
            int u = parts[0], v = parts[1];
            if (u == 0 && v == 0) continue;  // 跳过 (0,0)
            inst.edges.emplace_back(u, v);
        }
    }

    inst.num_edges = (int)inst.edges.size();
    inst.depot_idx = 0;

    return inst;
}
