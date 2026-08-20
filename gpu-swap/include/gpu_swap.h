/*
 * gpu_swap.h — GPU 显存到 CPU 内存的透明迁移层
 *
 * 核心职责：管理"热池"(GPU显存)和"冷池"(CPU内存)之间的数据迁移。
 * 策略层独立——通过 IEvictionPolicy 接口插入不同的替换算法。
 *
 * 与 CXL 的关系：
 *   CXL 在硬件层做同样的事——让 GPU 透明访问外部内存池。
 *   本项目的"外部内存池"是 CPU DDR，通过 PCIe 访问。
 *   管理逻辑是通用的——无论是 CXL (200ns) 还是 PCIe (10us)。
 */

#pragma once

#include <cstddef>
#include <cstdint>

// ---- 数据结构 ----

struct BlockDesc {
    int64_t block_id;          // 全局唯一标识
    size_t  size;              // 字节数
    void*   gpu_ptr;           // GPU 显存地址（如果当前在 GPU）
    void*   cpu_ptr;           // CPU pinned memory 地址（如果当前在 CPU）
    bool    in_gpu;            // 当前在 GPU 热池？
    int64_t last_access_time;  // 用于 LRU（单调递增的时间戳）
    int64_t access_count;      // 用于 LFU
};

enum class MigrateResult {
    OK = 0,
    ERR_GPU_OOM,           // 热池满 + 无法腾出空间
    ERR_CPU_OOM,           // 冷池满
    ERR_BLOCK_NOT_FOUND,   // block_id 不存在
    ERR_IN_PROGRESS,       // block 正在迁移中
    ERR_INVALID_STATE,     // block 状态不一致
};

// ---- 策略接口 ----

struct BlockStats {
    int64_t block_id;
    int64_t last_access;
    int64_t access_count;
    bool    in_gpu;
    size_t  size;
};

/**
 * 换出策略接口
 *
 * 实现时需要考虑的问题（来自项目一的 trace 分析）：
 * - block 的生命周期分布：短命 block 不值得换出
 * - block 的访问频率：热 block 换出代价大（频繁换回）
 * - 请求类型差异：短问答 vs 长文档的 block 模式完全不同
 */
class IEvictionPolicy {
public:
    virtual ~IEvictionPolicy() = default;

    /// 通知：block 被分配
    virtual void on_alloc(int64_t block_id, size_t size) {}

    /// 通知：block 被访问
    virtual void on_access(int64_t block_id) {}

    /// 通知：block 被释放
    virtual void on_free(int64_t block_id) {}

    /**
     * 选择一个换出候选
     * @param gpu_blocks  当前在 GPU 热池中的所有 block 的统计信息
     * @return block_id，或 -1 表示"不需要换出"
     *
     * 这个函数在每个换出决策点最多调用一次——不要在内部做 O(n²) 的事
     */
    virtual int64_t select_victim(
        const std::vector<BlockStats>& gpu_blocks) = 0;
};


// ---- 核心管理器 ----

class GpuSwapManager {
private:
    // TODO: 你来实现 pimpl
    // - std::unordered_map<int64_t, BlockDesc> blocks_
    // - size_t hot_capacity_, cold_capacity_
    // - size_t hot_used_, cold_used_
    // - cudaStream_t swap_stream_  （专用 stream）
    // - IEvictionPolicy* policy_
    //
    // 实现细节参考 GUIDE.md Step 2

public:
    /**
     * @param hot_pool_bytes   GPU 显存中给 KV Cache 用的预算
     * @param cold_pool_bytes  CPU pinned memory 预算
     */
    GpuSwapManager(size_t hot_pool_bytes, size_t cold_pool_bytes);
    ~GpuSwapManager();

    // 禁止拷贝（管理 GPU 资源）
    GpuSwapManager(const GpuSwapManager&) = delete;
    GpuSwapManager& operator=(const GpuSwapManager&) = delete;

    /**
     * 在热池中分配一个 block
     * 如果热池满，触发换出（通过 policy_->select_victim）
     * @return 分配的 BlockDesc（保证 in_gpu == true），或 nullptr
     */
    BlockDesc* allocate(int64_t block_id, size_t size);

    /// 释放 block（从热池或冷池）
    void free(int64_t block_id);

    /**
     * 换出：GPU 热池 → CPU 冷池
     * 异步执行——调用立刻返回，实际拷贝在 swap_stream_ 上排队
     */
    MigrateResult evict_to_cold(int64_t block_id);

    /**
     * 换回：CPU 冷池 → GPU 热池
     * 异步执行。如果热池满会先腾空间。
     */
    MigrateResult prefetch_to_hot(int64_t block_id);

    /// 等待所有正在进行的异步迁移完成
    void sync();

    /// 标记 block 被访问（更新 last_access_time 和 access_count）
    void touch(int64_t block_id);

    /// 设置策略
    void set_policy(IEvictionPolicy* policy);

    // ---- 统计 ----
    size_t hot_used_bytes() const;
    size_t cold_used_bytes() const;
    size_t hot_capacity() const;
    size_t cold_capacity() const;
    int    num_blocks() const;
    int    num_blocks_in_gpu() const;
};
