/*
 * gpu_swap.h — GPU 显存到 CPU 内存的透明迁移层
 *
 * 核心职责：管理"热池"(GPU显存)和"冷池"(CPU pinned内存)之间的数据迁移。
 * 策略层独立——通过 IEvictionPolicy 接口插入不同的替换算法。
 *
 * 与 CXL 的关系：
 *   CXL 在硬件层做同样的事——让 GPU 透明访问外部内存池。
 *   本项目的"外部内存池"是 CPU DDR，通过 PCIe 访问。
 *   管理逻辑是通用的——无论是 CXL (200ns) 还是 PCIe (10us)。
 *
 * 线程亲和：CUDA 上下文是 per-thread 的。本类的所有方法都必须从
 * **创建它的那个线程**调用，且**不是线程安全的**。
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

// ---- 数据结构 ----

struct BlockDesc {
    int64_t block_id;          // 全局唯一标识
    size_t  size;              // 字节数
    void*   gpu_ptr;           // GPU 显存地址（如果当前在 GPU）
    void*   cpu_ptr;           // CPU pinned memory 地址（如果当前在 CPU）
    bool    in_gpu;            // 当前在 GPU 热池？
    int64_t last_access_time;  // 用于 LRU（管理器的单调时钟，单位毫秒）
    int64_t access_count;      // 用于 LFU
    int64_t alloc_time;        // 最近一次分配时刻（毫秒），用于"年龄"判断

    // ⚠️ in_gpu 只是"数据有效性"这一层的投影，而且**只在迁移收割之后才翻转**。
    // 迁移在途时它保持原值——那正是异步的语义。想知道有没有在途操作，
    // 看 GpuSwapManager::pending_migrations() 或 is_migrating()。
};

enum class MigrateResult {
    OK = 0,
    ERR_GPU_OOM,           // 热池满 + 无法腾出空间
    ERR_CPU_OOM,           // 冷池满
    ERR_BLOCK_NOT_FOUND,   // block_id 不存在
    ERR_IN_PROGRESS,       // block 正在迁移中
    ERR_INVALID_STATE,     // block 状态不一致
    ERR_PINNED,            // block 被 pin 住（正在被推理读），本轮不可动
    ERR_CUDA,              // CUDA 调用失败，细节见 last_error()
};

const char* to_string(MigrateResult r);

// ---- 策略接口 ----

struct BlockStats {
    int64_t block_id;
    int64_t last_access;   // 管理器时钟（毫秒）
    int64_t access_count;
    bool    in_gpu;
    size_t  size;
    int64_t alloc_time;    // 最近一次分配时刻（毫秒）
    int64_t now;           // 本次决策时刻（毫秒）——每条记录里都是同一个值
};

/**
 * 换出策略接口
 *
 * 实现时需要考虑的问题（来自项目一的 trace 分析）：
 * - block 的年龄分布：刚分配的 block 不值得换出（22.8% 的 block 活不过 1 秒）
 * - block 的访问频率：热 block 换出代价大（频繁换回）
 * - 请求类型差异：短问答 vs 长文档的 block 模式完全不同
 *
 * ⚠️ 契约（两条，都必须遵守）：
 * 1. `gpu_blocks` 是**已经剔除掉 pinned / 在途 / 不可换出**的集合。
 *    策略不需要、也不应该自己再维护一份保护名单——那是管理器的职责。
 * 2. `select_victim` 在每个换出决策点最多调用一次，且**实现必须是 O(n) 或更好**。
 *    全体排序是 O(n log n)，n 上千时要在 Step 5 Q4（策略自身开销）里量它。
 */
class IEvictionPolicy {
public:
    virtual ~IEvictionPolicy() = default;

    /// 通知：block 被分配
    virtual void on_alloc(int64_t /*block_id*/, size_t /*size*/) {}
    /// 通知：block 被访问
    virtual void on_access(int64_t /*block_id*/) {}
    /// 通知：block 被释放
    virtual void on_free(int64_t /*block_id*/) {}

    /**
     * 选择一个换出候选
     * @param gpu_blocks  当前**可换出**的 block 统计（已排除 pinned / 在途）
     * @return block_id，或 -1 表示"没有合适的候选"
     */
    virtual int64_t select_victim(
        const std::vector<BlockStats>& gpu_blocks) = 0;
};


/// 一个 block 的只读快照，给测试和 C 接口用。
struct BlockView {
    BlockDesc desc;
    bool valid;       // 数据可读（在热池且已落地，或在冷池）
    bool has_cold;    // 在冷池持有槽位
    bool migrating;   // 有在途迁移
    bool pinned;      // 被 pin（至少一个读者事件未解除）
};

// ---- 核心管理器 ----

class GpuSwapManager {
public:
    /**
     * @param hot_pool_bytes   GPU 显存中给 KV Cache 用的预算
     * @param cold_pool_bytes  CPU pinned memory 预算
     *
     * 构造时按 hot_pool_bytes 一次性 cudaMalloc 出整块 arena，之后所有热池
     * 子分配都在 arena 内部切分——**迁移路径上不出现 cudaMalloc/cudaFree**。
     * 理由：驱动文档明写非 cudaMallocAsync 指针的 free 可能做隐式同步，
     * 那会让整个设备静默串行化，异步就变成了假的（数据还是对的，只是不重叠）。
     */
    GpuSwapManager(size_t hot_pool_bytes, size_t cold_pool_bytes);
    ~GpuSwapManager();

    GpuSwapManager(const GpuSwapManager&) = delete;
    GpuSwapManager& operator=(const GpuSwapManager&) = delete;

    /**
     * 在热池中分配一个 block。
     * 如果热池满，按策略触发换出；腾不出空间则返回 nullptr（ERR_GPU_OOM）。
     *
     * @return 指向管理器内部记录的指针，保证 in_gpu == true；
     *         **失效条件**：同一 block_id 被 free/重新 allocate，或对象析构。
     *         在途迁移收割时会改写它的字段，但指针本身不会失效
     *         （底层是 unordered_map，节点稳定）。
     */
    BlockDesc* allocate(int64_t block_id, size_t size);

    /// 释放 block（从热池或冷池）。在途迁移中的 block 会拒绝释放。
    MigrateResult free(int64_t block_id);

    /**
     * 换出：GPU 热池 → CPU 冷池，异步。
     *
     * 契约：**立即返回**，实际拷贝排在 swap_stream_ 上。
     * 返回 OK 时：block 仍在热池计费（显存还没还），in_gpu 仍为 true，
     * pending_migrations() 增加。要等它真正落地，用 sync() 或 wait_block()。
     */
    MigrateResult evict_to_cold(int64_t block_id);

    /**
     * 换回：CPU 冷池 → GPU 热池，异步。
     *
     * 如果热池满，会先换出 victim，并在**同一条 stream 上**紧接着发 H2D——
     * 同流顺序保证 H2D 执行时 victim 的槽位一定已经空出来，所以发起时刻
     * 就能把槽位过户给目标，host 不用等。
     */
    MigrateResult prefetch_to_hot(int64_t block_id);

    /**
     * 预热冷池：把 n 个槽位的 pinned 内存先分配好。
     *
     * **为什么必须有这个接口。** 冷池是惰性按 chunk cudaMallocHost 的，而
     * `cudaMallocHost` 是驱动文档点名的**隐式同步点** —— 它会让整个设备
     * 同步（"页锁定主机内存分配"本身就跨 stream 同步）。所以在第一次换出
     * 才去分配冷池时，那次换出会**等前面所有 kernel 跑完**，异步性恰好在
     * 最需要它的那一刻失效。
     *
     * 这不是推理出来的，是实测撞上的：不加预热，一个 250ms 的自旋 kernel
     * 能把紧随其后的一次 2 MB 换出整整挡住（冷池分配在等设备空闲），
     * 于是"没加 pin 的负对照"也会通过 —— 测试因此假绿。
     *
     * 调用时机：在任何 kernel 之前，或者在最后一段 kernel 之后。
     */
    MigrateResult reserve_cold(size_t n_slots);

    /// 等待所有正在进行的异步迁移完成，返回聚合结果
    MigrateResult sync();

    /// 等待单个 block 的在途迁移完成（没有在途则立即返回）
    MigrateResult wait_block(int64_t block_id);

    /**
     * 钉住 block：让换出**排在 reader 后面**。
     *
     * reader_done 是"读这个 block 的推理算力"所在的流上记录的事件。
     * 管理器把它接到 swap_stream_ 上（`cudaStreamWaitEvent`），
     * **不是**靠宿主互斥 —— 只加宿主锁的话，D2H 会和正在写这块显存的
     * kernel 并发，拷贝引擎读到的就是半成品。
     *
     * 语义是**顺序约束，不是否决权**（这一点容易搞反）：
     *   - 自动挑 victim（热池满时）会**绕开** pinned 的块 —— 有更好的候选。
     *   - 显式 `evict_to_cold` 照做，只是 D2H 会等 reader 跑完。
     *   - `free()` **拒绝** pinned 的块（ERR_PINNED）：换出能靠 stream 顺序
     *     做安全，释放不能 —— 释放后显存分给别人而读者还在读，是
     *     use-after-free。
     *
     * @param reader_done 必须**已经 record 过**（cudaStreamWaitEvent 在调用
     *        时对事件状态做快照，之后再 record 不会追溯）。传 nullptr 表示
     *        纯宿主侧保护，没有在途的 GPU 读。
     *        **事件所有权转给管理器**：pin 之后调用者不要自己 destroy，
     *        由 unpin / free / 析构负责。可嵌套，栈式（每次 pin 追加一个）。
     */
    MigrateResult pin(int64_t block_id, cudaEvent_t reader_done);
    MigrateResult unpin(int64_t block_id);

    /// 标记 block 被访问（更新 last_access_time 和 access_count）
    MigrateResult touch(int64_t block_id);

    /// 设置策略（不接管所有权）
    void set_policy(IEvictionPolicy* policy);

    /// 推进管理器的模拟时钟（毫秒）。只喂策略的时间戳，不做超时/统计。
    void set_clock(int64_t now_ms);
    int64_t clock() const;

    /**
     * 把所有在途迁移**强制完成**（内部 sync），然后按当前驻留状态重建
     * 策略需要的统计。用在校准前后需要一致视图的场合。
     */
    MigrateResult settle();

    // ---- 统计 ----
    size_t hot_used_bytes() const;      // 含在途 evict（显存还没还），不含在途 prefetch 的目标？
                                        // 不——见 .cpp 的详细口径说明：两者都算在热池
    size_t hot_valid_bytes() const;     // 只有已落地可读的部分
    size_t cold_used_bytes() const;
    size_t hot_capacity() const;
    size_t cold_capacity() const;
    size_t slot_bytes() const;          // 定长槽位的粒度（默认 2 MB）
    /// 内部 swap stream 的裸句柄（cudaStream_t），给"必须和迁移排同一条队"
    /// 的辅助函数用。返回 void* 是为了不把 cudaStream_t 带进纯 C 的调用者。
    void*  raw_swap_stream() const;
    size_t hot_peak_bytes() const;      // 峰值，报告用（结束时刻的瞬时值恒为 0）
    int    num_blocks() const;
    int    num_blocks_in_gpu() const;
    int    pending_migrations() const;
    bool   is_migrating(int64_t block_id) const;
    bool   is_pinned(int64_t block_id) const;

    /// 只读快照，**不推进状态机**（不会顺带收割在途迁移）。
    /// 找不到该 block 返回 false，此时 out 不动。
    bool   inspect(int64_t block_id, BlockView* out) const;

    /// 逐出序列指纹：对每次逐出按顺序折进一个 crc32。
    /// 聚合计数分辨不出"两个策略是否做了同样的决策"，顺序敏感指纹可以——
    /// 前提是实现是确定性的（本项目有模拟时钟 + 单线程，应当如此）。
    uint32_t eviction_fingerprint() const;
    int64_t  eviction_count() const;

    /// 最近一次异步失败的描述（空字符串表示没有）
    const char* last_error() const;

private:
    struct Impl;
    Impl* impl_;
};
