/*
 * eviction_policy.h — 四种换出策略实现
 *
 * 策略 A: LRU      — 选最久没被访问的 block
 * 策略 B: LFU      — 选访问次数最少的 block
 * 策略 C: Adaptive — 数据驱动的混合策略（见下）
 * 策略 D: GDSF     — 代价加权的频率密度（Adaptive 的排序内核，单独可跑作对照）
 *
 * 关于 Adaptive：为什么它长成现在这样
 * ------------------------------------------------------------
 * 骨架版本是这样写的：
 *
 *     if (b->access_count > 50) continue;      // 跳过"热"块
 *     ... 在剩下的里面挑 last_access 最小的
 *
 * 这个写法有三个问题，而且**都实测复现过**：
 *
 * 1. `50` 是拍脑袋的占位符，而且**任何单一绝对阈值都不可迁移**。项目一的
 *    trace 里，`>50` 在 long-summarize 上保护 98.9% 的块，在 legacy-c1 上
 *    只保护 27.0% —— 同一个常数，候选集大小差一个量级。详见
 *    policy_thresholds.h（那个头文件是脚本从 trace 现算的，不是手抄的）。
 *
 * 2. 保护率一高，候选集就空，于是掉进 `fallback` 分支"随便选一个" ——
 *    策略**静默退化**成 LRU，而外部完全看不出来。项目一栽过一次
 *    （tau 取 63ms 时 adaptive 逐位等于 LRU），gpu-swap 自己的比较脚本里
 *    又栽过一次（hot_threshold=50 保护 84.2% 的块 → adaptive ≡ lru，
 *    miss 率逐位相同到小数点后两位）。
 *
 * 3. "访问次数多"和"换出去代价大"被混成了一个数。次数多只说明过去被读过，
 *    不说明未来还会被读；而代价是可以实测的（重算 1620us vs 换回 52.4us）。
 *
 * 所以现在的 Adaptive = **代价加权排序（GDSF）+ 一层会自己松手的保护**：
 *
 * - **排序**：GDSF 打分，取分数最低的换出。分数是连续量，**永远存在最小值**，
 *   所以"选不出候选"在构造上不可能发生 —— 不可能静默退化。
 * - **保护**：年龄规则（数据驱动）+ 热块规则（绝对阈值 + 夹断）。
 *   候选集被保护层清空时**逐级松手**并计数，而不是掉进一个装作 LRU 的
 *   fallback。
 * - **上报**：保护率按决策点分桶报（首/末四分位 + min/mean/max）。
 *   两阶段行为（早期纯打分、后期过度保护）不做分桶就互相抵消看不见。
 * - **确定性**：并列按 (last_access, block_id) 字典序打破。不加这一条，
 *   模拟时钟下大量 block 同刻，谁赢由 unordered_map 的桶序决定，
 *   同一 workload 跑两遍能得到不同的逐出序列（项目一实测 12 次运行
 *   12 个指纹）。
 */

#include "gpu_swap.h"
#include "policy_thresholds.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace gsm_policy_detail {

/// 候选排序的确定性打破方式：先比主键，主键相等再比 (last_access, block_id)。
/// **每个策略都必须用它**，否则"同一 workload 跑两遍"给不出同一个逐出序列，
/// 而逐出序列的可复现性正是本项目用来证明"策略真的做了决策"的证据。
inline bool tie_break_less(const BlockStats& a, const BlockStats& b) {
    if (a.last_access != b.last_access) return a.last_access < b.last_access;
    return a.block_id < b.block_id;
}

}  // namespace gsm_policy_detail


// ---- 策略 A: LRU ----

class LRUPolicy : public IEvictionPolicy {
public:
    int64_t select_victim(const std::vector<BlockStats>& gpu_blocks) override {
        if (gpu_blocks.empty()) return -1;
        const BlockStats* victim = &gpu_blocks[0];
        for (const auto& b : gpu_blocks) {
            if (gsm_policy_detail::tie_break_less(b, *victim)) victim = &b;
        }
        return victim->block_id;
    }
};


// ---- 策略 B: LFU ----

class LFUPolicy : public IEvictionPolicy {
public:
    int64_t select_victim(const std::vector<BlockStats>& gpu_blocks) override {
        if (gpu_blocks.empty()) return -1;
        const BlockStats* victim = &gpu_blocks[0];
        for (const auto& b : gpu_blocks) {
            if (b.access_count < victim->access_count) {
                victim = &b;
            } else if (b.access_count == victim->access_count &&
                       gsm_policy_detail::tie_break_less(b, *victim)) {
                victim = &b;
            }
        }
        return victim->block_id;
    }
};


// ---- 策略 D: GDSF（代价加权的频率密度）----
//
// 出自 Squid 的 GDSF（Cherkasova 的 GD-Size-Frequency）：
//
//     score(b) = (base(b) - origin) + cost_per_hit * hits(b) / size(b)
//
// `hits` 用 BlockStats 的 access_count（管理器是唯一权威来源，策略不自己数）。
// `base` 是块**进入热池那一刻**的通胀水位 `L`，此后冻结；每次逐出把 L 更新到
// 被逐出者的分数（GreedyDual 原文的 L ← H(victim)，是**赋值**不是取最大）。
//
// ⚠️ 为什么要有 origin 这一项 —— 这里踩过一个实测的坑，留痕：
//
// L 是**单调递增**的：新块以 L 为起点，而 L 又只被抬不被压，于是 L 会随时间
// 无界增长。而频率项的尺度是 `cost_per_hit * hits`（本 workload 下 ≈ 0.77/hit），
// 一个跑了上百步的块也就 100 上下。实测（60 请求 / 254 槽，模拟器）：
//
//     请求数   L      miss    LFU 的 miss
//      20    64.3    34.7%      33.5%
//      40    99.5    54.6%      40.1%
//      80   214.3    74.0%      45.8%
//
// L 涨到 214 之后，新块（base = L、0 次命中）的分数**高于**一个有 200 次命中
// 的老块（base ≈ 64 + 153）。排序于是退化成"逐出最老的" —— 而 LRU 在 80 请求
// 那档是 90.7%。**GDSF 悄悄变成了 FIFO/LRU**，正是本项目最想避免的那种静默退化。
//
// 修法是给通胀加一条**不变量**，而不是调参：
//
//     L - origin ≤ max(频率项 over 驻留块)
//
// 即"通胀的幅度不许超过**最热驻留块**的频率项"。通胀的用途是给冷启动垫一脚
// （新块计数低只是还没轮到被读），不是压倒真实热度；一旦它压过了，就把它
// 整体平移到 0（`origin = L`，O(1) 重基准，不改任何两个块之间的相对次序）。
// 重基准次数计入报告 —— 退化因此可见，不是靠人盯。
//
// 修完：80 请求那档 74.0% → 60.0%。
//
// 与此独立的另一条实测结论：**本 workload 上 GDSF 仍然输给纯 LFU**
// （80 请求：GDSF 60.0% vs LFU 45.8%）。原因是所有块的 size 相同，
// `cost/size` 提供不了频率之外的任何信号，于是 GDSF 与 LFU 的**唯一**差别
// 就是这个通胀 —— 而通胀对 KV cache 的前提不成立：Web 代理里新抓回来的对象
// 常常再也无人访问，所以要挡住它；KV cache 里刚分配的块**几乎必然**马上会被读。
// 详见 README 的对比表。
class GDSFPolicy : public IEvictionPolicy {
public:
    explicit GDSFPolicy(double cost_per_hit_us =
                            GSM_RECOMPUTE_US_PER_BLOCK -
                            GSM_SWAP_IN_US_PER_BLOCK)
        : cost_per_hit_us_(cost_per_hit_us) {}

    void on_alloc(int64_t block_id, size_t) override {
        base_[block_id] = inflation_;
    }
    void on_free(int64_t block_id) override { base_.erase(block_id); }

    /// 频率项（代价加权的命中密度）。通胀是加在它上面的**偏移**。
    double freq_term(const BlockStats& b) const {
        double size_kb = std::max<size_t>(b.size, 1) / 1024.0;
        return cost_per_hit_us_ * static_cast<double>(b.access_count) / size_kb;
    }

    /// 进入排序的分数。base 存的是绝对值，减掉 origin 才是当前基准下的相对位置。
    double score(const BlockStats& b) const {
        auto it = base_.find(b.block_id);
        double base = (it == base_.end()) ? 0.0 : it->second;
        return (base - origin_) + freq_term(b);
    }

    int64_t select_victim(const std::vector<BlockStats>& gpu_blocks) override {
        if (gpu_blocks.empty()) return -1;
        const BlockStats* victim = &gpu_blocks[0];
        double best = score(*victim);
        double max_freq = freq_term(*victim);
        for (const auto& b : gpu_blocks) {
            double s = score(b);
            if (s < best || (s == best && gsm_policy_detail::tie_break_less(b, *victim))) {
                best = s;
                victim = &b;
            }
            max_freq = std::max(max_freq, freq_term(b));
        }
        // GreedyDual 原文：L ← H(被逐出者)，**赋值**（取最大会把它变成只涨不落的棘轮）
        inflation_ = best + origin_;
        // 不变量：通胀幅度不许超过最热驻留块的频率项，超了就 O(1) 平移到 0
        if (inflation_ - origin_ > max_freq) {
            origin_ = inflation_;
            ++rebases_;
        }
        return victim->block_id;
    }

    /// 当前生效的通胀幅度（相对基准）。重基准之后归零，所以它是**有界**的。
    double inflation() const { return inflation_ - origin_; }
    /// 通胀被迫重基准的次数。>0 就说明 L 曾涨到压过真实热度 —— 退化可见。
    int64_t rebases() const { return rebases_; }

private:
    double cost_per_hit_us_;
    double inflation_ = 0.0;      // 绝对值
    double origin_ = 0.0;         // 基准；effective L = inflation_ - origin_
    int64_t rebases_ = 0;
    std::unordered_map<int64_t, double> base_;
};


// ---- 策略 C: Adaptive ----

class AdaptivePolicy : public IEvictionPolicy {
public:
    struct Config {
        /// 保护层最多允许挡掉这么大比例的驻留块。超了就把热块阈值往上抬
        /// （见下面 cutoff 的计算）。这个值同时是"退化可见"的闸门：
        /// 夹断次数会被计数并上报。
        double protect_max_rate = GSM_MAX_PROTECTION_RATE;

        /// 绝对热块阈值。**只作次级保险** —— 它在不同 workload 上保护率
        /// 从 27% 到 99%，单独用必然在某个 workload 上退化。
        int64_t hot_access_threshold = GSM_HOT_ACCESS_THRESHOLD;

        /// 年龄规则："刚分配 = 马上还会被读"。见 policy_thresholds.h。
        int64_t young_age_ms = GSM_YOUNG_AGE_MS;

        /// 每次命中省下的重算时间（微秒）—— GDSF 打分的 cost 项，31x 代价比。
        double cost_per_hit_us = GSM_RECOMPUTE_US_PER_BLOCK - GSM_SWAP_IN_US_PER_BLOCK;

        /// 每隔多少次决策重算一次热块分位。每决策点都算全体分位是 O(n log n)，
        /// 驻留集上千时开销不可忽略（GUIDE Step 5 Q4 要量的就是它）。
        int cutoff_period = 8;
    };

    struct Report {
        int64_t decisions = 0;         // select_victim 调用次数
        int64_t victims = 0;           // 给出候选的次数
        int64_t clamped = 0;           // 绝对阈值被夹断（保护率超线）的次数
        int64_t relaxed_hot = 0;       // 连热块规则也放开的次数（最后一道；此时两条都已放开）
        int64_t relaxed_young = 0;     // 保护层清空候选、被迫放开年龄规则的次数（第一道）
        int64_t young_block_hits = 0;  // 累计被年龄规则挡下的 (块 × 决策)
        int64_t hot_block_hits = 0;    // 累计被热块规则挡下的 (块 × 决策)
        int64_t resident_seen = 0;     // 累计见过的驻留块数（算保护率的分母）

        double protect_min = 0.0;
        double protect_mean = 0.0;
        double protect_max = 0.0;
        // 按决策点分桶：只看 min/mean/max 会让"早期纯打分、后期过度保护"
        // 这两个阶段互相抵消。首/末四分位能直接看出漂移。
        double protect_first_quartile = 0.0;
        double protect_last_quartile = 0.0;

        int64_t cutoff_last = 0;       // 最近一次生效的热块阈值
        int64_t cutoff_clamped_last = 0;  // 夹断后实际用的阈值
        double inflation = 0.0;        // 排序内核当前的通胀幅度
        int64_t inflation_rebases = 0; // 通胀被迫重基准的次数（>0 = 退化过）
    };

    // 不能写成 `Config cfg = Config()`：默认实参在外层类还没完整时就要解析，
    // 而 Config 的成员默认值是在外层类结束时才就绪的（GCC 会直接报错）。
    AdaptivePolicy() : since_cutoff_(cfg_.cutoff_period) {}
    explicit AdaptivePolicy(Config cfg)
        : cfg_(cfg), since_cutoff_(cfg.cutoff_period) {}

    const Config& config() const { return cfg_; }

    void on_alloc(int64_t block_id, size_t) override {
        core_.on_alloc(block_id, 0);
    }
    void on_free(int64_t block_id) override { core_.on_free(block_id); }

    int64_t select_victim(const std::vector<BlockStats>& gpu_blocks) override {
        if (gpu_blocks.empty()) return -1;

        const int64_t now = gpu_blocks[0].now;
        const size_t n = gpu_blocks.size();

        // ---- 热块阈值：绝对阈值 + 夹断 ----
        //
        // 目标是"保护 access_count > cutoff 的块不超过 protect_max_rate"。
        // cutoff 越大保护越少，所以夹断 = 把 cutoff 抬到相应分位上。
        // 抬到的那条线是 O(n) 的 nth_element，不是全排序。
        if (++since_cutoff_ >= cfg_.cutoff_period) refresh_cutoff(gpu_blocks);

        int64_t cutoff = cfg_.hot_access_threshold;
        size_t over = 0;
        for (const auto& b : gpu_blocks) {
            if (b.access_count > cutoff) ++over;
        }
        if (static_cast<double>(over) > cfg_.protect_max_rate * n) {
            cutoff = clamped_cutoff_;
            ++stats_.clamped;
        }
        stats_.cutoff_last = cfg_.hot_access_threshold;
        stats_.cutoff_clamped_last = cutoff;

        // ---- 逐块分类 ----
        std::vector<const BlockStats*> fresh;   // 只过了"非热"这一关
        fresh.reserve(n);
        size_t young_hits = 0, hot_hits = 0;
        for (const auto& b : gpu_blocks) {
            if (b.access_count > cutoff) {
                ++hot_hits;
                continue;
            }
            fresh.push_back(&b);
        }
        std::vector<const BlockStats*> candidates;
        candidates.reserve(fresh.size());
        for (const auto* b : fresh) {
            if (now - b->alloc_time < cfg_.young_age_ms) {
                ++young_hits;
                continue;
            }
            candidates.push_back(b);
        }

        // ---- 保护层逐级松手（**不是**掉进一个装作 LRU 的 fallback）----
        if (candidates.empty()) {
            ++stats_.relaxed_young;
            candidates = fresh;                       // 放开年龄规则（fresh 仍不含热块）
        }
        if (candidates.empty()) {
            ++stats_.relaxed_hot;                     // 连热块也放开 → 两条规则都已放开
            candidates.reserve(n);
            for (const auto& b : gpu_blocks) candidates.push_back(&b);
        }

        // ---- 保护率上报 ----
        std::vector<BlockStats> scoped;
        scoped.reserve(candidates.size());
        for (const auto* b : candidates) scoped.push_back(*b);
        int64_t victim = core_.select_victim(scoped);

        ++stats_.decisions;
        stats_.resident_seen += static_cast<int64_t>(n);
        stats_.young_block_hits += static_cast<int64_t>(young_hits);
        stats_.hot_block_hits += static_cast<int64_t>(hot_hits);
        rates_.push_back(static_cast<double>(young_hits + hot_hits) /
                         static_cast<double>(n));
        if (victim >= 0) ++stats_.victims;
        return victim;
    }

    Report report() const {
        Report r = stats_;
        r.inflation = core_.inflation();
        r.inflation_rebases = core_.rebases();
        if (!rates_.empty()) {
            r.protect_min = *std::min_element(rates_.begin(), rates_.end());
            r.protect_max = *std::max_element(rates_.begin(), rates_.end());
            double sum = 0;
            for (double v : rates_) sum += v;
            r.protect_mean = sum / rates_.size();
            r.protect_first_quartile = quartile_mean(0.00, 0.25);
            r.protect_last_quartile = quartile_mean(0.75, 1.00);
        }
        return r;
    }

    void reset_stats() {
        stats_ = Report{};
        rates_.clear();
        since_cutoff_ = cfg_.cutoff_period;   // 强制下次重算
    }

private:
    double quartile_mean(double lo, double hi) const {
        if (rates_.empty()) return 0.0;
        size_t a = static_cast<size_t>(lo * rates_.size());
        size_t b = std::max(a + 1, static_cast<size_t>(hi * rates_.size()));
        b = std::min(b, rates_.size());
        double sum = 0;
        for (size_t i = a; i < b; ++i) sum += rates_[i];
        return sum / static_cast<double>(b - a);
    }

    void refresh_cutoff(const std::vector<BlockStats>& gpu_blocks) {
        since_cutoff_ = 0;
        // 想要的是"保护不超过 protect_max_rate"，即取 (1 - q) 分位做阈值。
        // protect_max_rate = 0.30 时，阈值 = 计数分布的第 70 百分位。
        double q = 1.0 - cfg_.protect_max_rate;
        std::vector<int64_t> counts;
        counts.reserve(gpu_blocks.size());
        for (const auto& b : gpu_blocks) counts.push_back(b.access_count);
        if (counts.empty()) return;
        size_t k = std::min(static_cast<size_t>(q * counts.size()),
                            counts.size() - 1);
        std::nth_element(counts.begin(), counts.begin() + k, counts.end());
        clamped_cutoff_ = counts[k];
    }

    Config cfg_;
    GDSFPolicy core_;
    std::vector<double> rates_;
    Report stats_;
    // 初值 = period，让**第一次决策**就先把分位算出来。留成 0 的话第一次
    // 决策会拿没算过的 clamped_cutoff_（= 0）当阈值 → 什么都被保护 →
    // 候选集空 → 保护率报 100%。而第一次恰恰是最该被看见的那一次。
    int since_cutoff_ = 0;
    int64_t clamped_cutoff_ = 0;
};
