/*
 * eviction_policy.h — 三种换出策略实现
 *
 * 策略 A: LRU   — 选最久没被访问的 block
 * 策略 B: LFU   — 选访问次数最少的 block
 * 策略 C: Adaptive — 基于项目一 trace 数据的混合策略：
 *         - 跳过"短命"block（生命周期 < 阈值）
 *         - 跳过"热"block（访问次数 > 阈值）
 *         - 在剩余候选中用 LRU 排序
 *
 * 关键思考：为什么 Adaptive 策略需要项目一的数据？
 *   因为"短命"和"热"的阈值不是拍脑袋定的——
 *   需要从真实的 KV Cache trace 里分析出来。
 *   这正是 kvcache-tracer 和数据驱动的系统设计的价值。
 */

#include "gpu_swap.h"
#include <vector>
#include <algorithm>
#include <climits>
#include <unordered_set>

// ---- LRU 策略 ----

class LRUPolicy : public IEvictionPolicy {
public:
    int64_t select_victim(const std::vector<BlockStats>& gpu_blocks) override {
        if (gpu_blocks.empty()) return -1;

        int64_t victim = -1;
        int64_t oldest = INT64_MAX;
        for (const auto& b : gpu_blocks) {
            if (b.last_access < oldest) {
                oldest = b.last_access;
                victim = b.block_id;
            }
        }
        return victim;
    }
};


// ---- LFU 策略 ----

class LFUPolicy : public IEvictionPolicy {
public:
    int64_t select_victim(const std::vector<BlockStats>& gpu_blocks) override {
        if (gpu_blocks.empty()) return -1;

        int64_t victim = -1;
        int64_t min_count = INT64_MAX;
        for (const auto& b : gpu_blocks) {
            if (b.access_count < min_count) {
                min_count = b.access_count;
                victim = b.block_id;
            }
        }
        return victim;
    }
};


// ---- Adaptive 策略 ----

class AdaptivePolicy : public IEvictionPolicy {
private:
    // 保护不换出的 block id 集合
    std::unordered_set<int64_t> pinned_;

    // 阈值——这些值应该从项目一的 trace 数据中推导
    // 默认值只是占位符，你需要用自己的数据替换
    double short_lifetime_threshold_s_ = 1.0;   // <1s 的 block 视为"短命"
    int    hot_access_threshold_ = 50;            // >50 次访问视为"热块"

    // 每个 block 的 alloc 时间（用于判断"短命"）
    // 在真实系统中，生命周期是从 trace 中观测的，不是实时的
    // 这里用一个简化方案：记录 alloc 时的时间戳

public:
    /// 标记一个 block 为"受保护"（不会被选中换出）
    void pin(int64_t block_id)   { pinned_.insert(block_id); }
    void unpin(int64_t block_id) { pinned_.erase(block_id); }

    /**
     * 设置阈值。
     *
     * 典型用法（数据驱动）：
     *   1. 运行 kvcache-tracer，得到 trace
     *   2. 用 analyze.py 分析 trace，得到：
     *      - P50 生命周期 = 0.8s  → 设 short_lifetime = 1.0s
     *      - P90 访问次数 = 45   → 设 hot_access = 50
     *   3. 将这些值传入 AdaptivePolicy
     */
    void set_short_lifetime_threshold(double seconds) {
        short_lifetime_threshold_s_ = seconds;
    }
    void set_hot_access_threshold(int count) {
        hot_access_threshold_ = count;
    }

    int64_t select_victim(const std::vector<BlockStats>& gpu_blocks) override {
        // 第一轮：跳过受保护的 block，收集候选项
        std::vector<const BlockStats*> candidates;
        for (const auto& b : gpu_blocks) {
            if (pinned_.count(b.block_id)) continue;
            candidates.push_back(&b);
        }

        if (candidates.empty()) {
            // 所有 block 都被保护——只能随便选一个（fallback 到 LRU）
            // 这种情况说明热池设得太小了
            int64_t victim = -1;
            int64_t oldest = INT64_MAX;
            for (const auto& b : gpu_blocks) {
                if (b.last_access < oldest) {
                    oldest = b.last_access;
                    victim = b.block_id;
                }
            }
            return victim;
        }

        // 第二轮：在候选中用 LRU 排序
        int64_t victim = -1;
        int64_t oldest = INT64_MAX;
        for (const auto* b : candidates) {
            // TODO: 这里可以加入"短命 block 跳过"的逻辑
            // 但生命周期信息需要从外部传入（历史 trace）
            // 简化处理：只跳过高频访问的 block
            if (b->access_count > hot_access_threshold_) continue;

            if (b->last_access < oldest) {
                oldest = b->last_access;
                victim = b->block_id;
            }
        }

        // 如果所有候选都是"热块"，退化成普通 LRU
        if (victim < 0) {
            for (const auto* b : candidates) {
                if (b->last_access < oldest) {
                    oldest = b->last_access;
                    victim = b->block_id;
                }
            }
        }

        return victim;
    }
};
