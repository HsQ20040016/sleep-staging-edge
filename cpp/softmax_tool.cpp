// softmax_tool.cpp —— 迷你分类器输出层（练习骨架）
//
// 编译:  g++ -std=c++17 -O2 -Wall -o softmax_tool softmax_tool.cpp
// 运行:  ./softmax_tool 2.0 1.0 0.1
//        ./softmax_tool --k 2 3.0 1.0 0.5
//
// 你要做的：把下面 3 个标了 TODO 的函数写出来。其余部分不要改。
// 每个函数上方都写了规格和验收标准 —— 先读规格，再动手。
// （程序运行时的提示故意写成英文：中文串在 Windows 控制台可能变乱码。）

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// ==================== 1. 求最大值 ====================
// 规格: 返回 v 里最大的元素。调用方保证 v 非空。
// 验收: {2,1,0.1} -> 2 ；{-5,-3,-9} -> -3 ；{7} -> 7
// 提示: 手写循环。别用 std::max_element（那是第 18 章的内容）。
double find_max(const std::vector<double>& v) {
    size_t max_idx = 0;
    for (size_t i = 0; i < v.size(); i++) {
        if (v[i] > v[max_idx]) {
            max_idx = i;
        }
    }
    return v[max_idx];
}

// ==================== 2. softmax ====================
// 规格: 返回与 logits 等长的概率向量 p：
//         p[i] = exp(logits[i] - M) / Σ exp(logits[j] - M)，其中 M = 该向量的最大值。
//       M 必须自己调上面的 find_max 求出来 —— 「减去 M」是本题的重点，不是可选项。
// 验收:
//   a) softmax({2,1,0.1}) ≈ {0.659001, 0.242433, 0.098566}  误差 < 1e-6
//   b) 所有元素之和 ≈ 1.0                                   误差 < 1e-9
//   c) 每个元素都 > 0
//   d) 【关键】softmax({1000,999,998}) ≈ {0.665241, 0.244728, 0.090031}，
//      不能出现 inf 或 nan。先想清楚：exp(1000) 等于多少？减掉 M 之后变成了什么？
std::vector<double> softmax(const std::vector<double>& logits) {
    double max = find_max(logits);
    std::vector<double> p;
    for (double x : logits) {
        p.push_back(std::exp(x - max));
    }
    double sum = 0.0;
    for (double x : p) sum += x;
    for (double& x : p) x /= sum;
    return p;
}

// ==================== 3. top-k 索引 ====================
// 规格: 返回 p 中最大的 k 个元素的下标，按概率从大到小排列。
//       k 超过 p.size() 时返回全部下标；k <= 0 时返回空 vector。
//       不许用 std::sort（第 18 章的内容）—— 用「选 k 轮，每轮扫一遍，
//       找当前最大、且没被选过的那个」来做。
// 验收: topk({0.66,0.24,0.10}, 2) -> {0,1}
//       topk({0.10,0.70,0.20}, 3) -> {1,2,0}
//       返回的索引不能重复。
std::vector<int> topk(const std::vector<double>& p, int k) {
    std::vector<int> idx;
    if (k <= 0) return idx;
    if (k > (int)p.size()) k = (int)p.size();

    std::vector<bool> used(p.size(), false);

    for (int round = 0; round < k; ++round) {
        int best = -1;
        for (int i = 0; i < (int)p.size(); i++) {
            if (!used[i] && (best == -1 || p[i] > p[best])) {
                best = i;
            }
        }
        used[best] = true;
        idx.push_back(best);
    }
    return idx;
}

// ==================== 以下已给，不用改 ====================

bool g_args_ok = true;

std::vector<double> parse_logits(int argc, char** argv, int& k) {
    k = 3;
    std::vector<double> logits;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--k") {
            if (i + 1 >= argc) { g_args_ok = false; return logits; }
            k = std::stoi(argv[++i]);
        } else {
            logits.push_back(std::stod(a));
        }
    }
    return logits;
}

int main(int argc, char** argv) {
    int k = 3;
    std::vector<double> logits = parse_logits(argc, argv, k);
    if (!g_args_ok || logits.empty()) {
        std::cerr << "usage: softmax_tool [--k N] <logits...>\n"
                     "e.g.   softmax_tool 2.0 1.0 0.1\n";
        return 1;
    }

    std::vector<double> p = softmax(logits);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "logits :";
    for (double x : logits) std::cout << ' ' << x;
    std::cout << "\nprobs  :";
    for (double x : p) std::cout << ' ' << x;

    double sum = 0.0;
    for (double x : p) sum += x;
    std::cout << "\nsum    : " << sum
              << (std::fabs(sum - 1.0) < 1e-9 ? "   [OK]" : "   [FAIL]") << '\n';

    std::vector<int> idx = topk(p, k);
    std::cout << "top" << k << "   :";
    for (int i : idx) std::cout << " (" << i << ", " << p[i] << ")";
    std::cout << '\n';
    return 0;
}

// ==================== 写完自问三句 ====================
// 1. 为什么 softmax 里必须先减掉最大值？不减的话 logits = {1000,999,998} 会算出什么？
// 2. topk 里，「没被选过的那个」你是怎么判断的？如果 k == p.size()，你的循环会怎样？
// 3. find_max 的初值你取的是什么？如果图省事写成 double m = 0;，
//    当 logits 全是很大的负数（-1000 -999 -998）时会算出什么？
//    想一想：「减一个错的常数」在数学上为什么不影响结果，在浮点数上却会出事？
