/*
 * bench_policy.cpp — 策略自身的开销（GUIDE Step 5 的 Q4）
 *
 * Q4: "你的 Adaptive 策略的额外开销（计算热度、维护统计）有多大？
 *      是否超过了好处？"
 *
 * 这个量**必须**单独量，因为它和迁移开销是两笔账：
 *   - 迁移开销走 PCIe，每块几十微秒，是可以被"少换几次"省回来的；
 *   - 策略开销吃 CPU，每次换出决策一次，是**纯支出**。
 * 如果策略每次决策要 50 µs，而一次换出只要 52.4 µs（项目一实测），
 * 那就等于把换出的钱原样花在了"决定换谁"上 —— 白干。
 *
 * 量法：**不碰 CUDA**，直接构造驻留集喂给 select_victim。
 * 这样量到的就是纯策略开销，不含任何迁移噪声。
 *
 * 用法: ./build/bench_policy [驻留集大小...]
 *       默认 64 512 2048 8192
 */

#include "eviction_policy.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

/// 造一个"像那么回事"的驻留集：访问次数重尾（少数块很热），
/// 分配时间分散。全同的话 LFU/GDSF 的并列分支会被反复走，
/// 量出来的不是真实分布下的开销。
std::vector<BlockStats> make_resident(size_t n, int64_t now, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::lognormal_distribution<double> heavy(1.0, 1.2);
    std::vector<BlockStats> v;
    v.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        BlockStats b{};
        b.block_id = static_cast<int64_t>(i);
        b.size = 2u << 20;
        b.in_gpu = true;
        b.now = now;
        b.alloc_time = now - static_cast<int64_t>(rng() % 20000);
        b.last_access = now - static_cast<int64_t>(rng() % 5000);
        b.access_count = static_cast<int64_t>(heavy(rng));
        v.push_back(b);
    }
    return v;
}

struct Row {
    const char* name;
    double us_per_decision;
    double us_per_block;
};

Row bench(IEvictionPolicy& p, const std::vector<BlockStats>& resident,
          int iters) {
    // 预热：让 nth_element 的分位、GDSF 的 base_ 表都就位
    for (int i = 0; i < 32; ++i) p.select_victim(resident);

    // 时间要摊在**决策次数**上。每次决策扫一遍驻留集，
    // 所以 per-block 那列才是和"驻留集多大"无关的可比量。
    //
    // ⚠️ 必须消费返回值，否则编译器会把整个循环删掉：LRUPolicy/LFUPolicy
    // 是**纯函数**（无成员、无副作用），编译器看得一清二楚。实测撞上过 ——
    // 第一版量出来 LRU/LFU 是 0.000 us，而 GDSF/Adaptive 有可变成员留了下来，
    // 于是表格里"最简单的策略最快"这个结论完全是假的。
    int64_t sink = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        sink += p.select_victim(resident);
        // 顺带让每次喂进去的驻留集略有不同，避免别的 CSE 把循环压扁
        if ((i & 1023) == 0) const_cast<BlockStats&>(resident[i % resident.size()])
                                .access_count += 1;
    }
    auto t1 = std::chrono::steady_clock::now();
    if (sink == 0x7fffffff) std::printf("(不可能发生，只为阻止优化)");

    double total_us =
        std::chrono::duration<double, std::micro>(t1 - t0).count();
    double per_dec = total_us / iters;
    return Row{"", per_dec, per_dec / static_cast<double>(resident.size())};
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<size_t> sizes;
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            sizes.push_back(static_cast<size_t>(std::atoll(argv[i])));
        }
    } else {
        sizes = {64, 512, 2048, 8192};
    }

    const int64_t now = 100000;
    const int iters = 20000;

    std::printf("=== 策略自身开销（纯 CPU，不含任何迁移）===\n");
    std::printf("每次决策 = 一次 select_victim；per-block 列 = 每次决策 / 驻留块数\n");
    std::printf("参照：项目一实测换出一个块 52.4 us、重算一个块 1620 us\n\n");

    std::printf("%-12s", "驻留块数");
    const char* names[] = {"LRU", "LFU", "GDSF", "Adaptive"};
    for (const char* n : names) std::printf("%22s", n);
    std::printf("\n");

    std::vector<std::vector<Row>> table;
    for (size_t n : sizes) {
        auto resident = make_resident(n, now, 12345 + n);
        std::vector<Row> row;
        {
            LRUPolicy p;
            Row r = bench(p, resident, iters);
            r.name = "LRU";
            row.push_back(r);
        }
        {
            LFUPolicy p;
            Row r = bench(p, resident, iters);
            r.name = "LFU";
            row.push_back(r);
        }
        {
            GDSFPolicy p;
            Row r = bench(p, resident, iters);
            r.name = "GDSF";
            row.push_back(r);
        }
        {
            AdaptivePolicy p;
            Row r = bench(p, resident, iters);
            r.name = "Adaptive";
            row.push_back(r);
        }
        table.push_back(row);
    }

    std::printf("%-12s", "（us/次）");
    for (size_t k = 0; k < 4; ++k) std::printf("%22s", names[k]);
    std::printf("\n");
    std::printf("%s\n", std::string(12 + 22 * 4, '-').c_str());
    for (size_t i = 0; i < sizes.size(); ++i) {
        std::printf("%-12zu", sizes[i]);
        for (const auto& r : table[i]) std::printf("%22.3f", r.us_per_decision);
        std::printf("\n");
    }
    std::printf("\n%-12s", "（ns/块）");
    for (size_t k = 0; k < 4; ++k) std::printf("%22s", names[k]);
    std::printf("\n");
    std::printf("%s\n", std::string(12 + 22 * 4, '-').c_str());
    for (size_t i = 0; i < sizes.size(); ++i) {
        std::printf("%-12zu", sizes[i]);
        for (const auto& r : table[i]) {
            std::printf("%22.1f", r.us_per_block * 1000.0);
        }
        std::printf("\n");
    }

    // 结论行：把开销折成"每次换出多花了百分之几"
    std::printf("\n相对一次换出（52.4 us）的比例：\n");
    for (size_t i = 0; i < sizes.size(); ++i) {
        std::printf("  %5zu 块: ", sizes[i]);
        for (const auto& r : table[i]) {
            std::printf("%s=%.2f%%  ", r.name, r.us_per_decision / 52.4 * 100.0);
        }
        std::printf("\n");
    }
    return 0;
}
