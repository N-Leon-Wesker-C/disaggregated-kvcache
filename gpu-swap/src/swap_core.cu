/*
 * swap_core.cu — GpuSwapManager 的实现
 *
 * 这个文件里几乎每一处"看起来绕"的地方都是被一个具体的坑逼出来的。
 * 六个关键决定，逐条给理由：
 *
 * 1. **热池是 arena，不是逐块 cudaMalloc。**
 *    `cuda_runtime_api.h:4826-4831` 明写：只有 cudaMallocAsync 出来的指针
 *    不做隐式同步，"For all other pointers, this API may perform implicit
 *    synchronization"。也就是说"等事件完成 → cudaFree"这条路会把**整个
 *    设备**在释放那一刻串行化。而异步测试照样会通过（数据是对的），
 *    只是 host 被拖住。所以 ctor 里一次 cudaMalloc 出整个热池，迁移路径上
 *    **一个 cudaMalloc / cudaFree 都没有**，reap 变成纯记账。
 *
 * 2. **定长槽位（slot）。** 热池切成 n_slots 个 slot_bytes 的槽。
 *    KV cache 的 block 本来就是定长的（vLLM 的 block_size 是固定配置），
 *    定长让"槽位过户"成为可能（见第 3 条），也让预算与实际占用严格相等。
 *    size > slot_bytes 的请求直接拒绝（ERR_INVALID_STATE），不留半吊子支持。
 *
 * 3. **腾空间用"单流链式过户"，不 host 等待。**
 *    热池满时要给 prefetch 腾地方：先在**同一条 stream** 上排 D2H(victim)，
 *    紧接着排 H2D(target)。H2D 执行时 victim 的 D2H 一定已经读完，
 *    所以**发起时刻**就能把 victim 的槽位过户给 target，host 零等待。
 *    成立的前提是三条不变量：(i) 一切迁移只在这条 stream 上；
 *    (ii) 过户只在"释放者的拷贝在入队序上先于使用者的拷贝"时进行；
 *    (iii) 别的 stream 不许碰这块显存（靠 pin + event，见第 4 条）。
 *    副作用是账目更准：同一块物理显存被复用，占用恒为 1 个槽，
 *    不需要按"两个都在途都算"来保守多计。
 *
 * 4. **pin 是 CUDA 侧的顺序约束，不是宿主互斥。**
 *    只加宿主锁是**假锁**：推理 kernel 在 infer stream 上排队 → 调用者
 *    unpin → 我们立刻在 swap_stream_ 上发 D2H，两者之间没有任何 CUDA
 *    依赖，拷贝引擎读的正是那个 kernel 正在写的显存。做法是 pin 时把
 *    reader 事件存下来，真正要发 D2H 时 `cudaStreamWaitEvent(swap_stream_)`。
 *    （仓库自己在 kvcache-tracer/GUIDE.md:349-351 写过正确答案：
 *     vLLM 的迁移层是 `stream.wait_stream(current_stream())`。）
 *
 * 5. **三个正交的量，不是一个 bool。** 参考骨架用一个 `in_gpu` 硬塞了三件事：
 *      - **槽位归属**（slot）：这块显存地址归谁
 *      - **数据有效性**（valid）：能不能读
 *      - **预算占用**：= slot 数（定长槽位下二者等价）
 *    `in_gpu` 只在 reap 时翻转 —— 迁移在途时它保持原值，那正是异步的语义。
 *    骨架的真 bug 就是在这里把 `in_gpu = true` 提前设了，把"拷贝还没落地"
 *    的块公开成可读。（骨架另外那个"事件建好就销毁"其实不是 bug：
 *     `cuda_runtime_api.h:2976-2979` 明许可以提前销毁。它的问题是**那个事件
 *     从来没被 query 过**，建好、记录、销毁，中间没有消费者，状态机永远不动。）
 *
 * 6. **错误只在 reap 时才知道**，所以：粘性 last_error_ + 逐方向回滚 +
 *    sync() 有返回码。发起失败（参数非法等）必须**回滚预留**，否则预算
 *    被永久吃掉一块。
 */

#include "gpu_swap.h"
#include "eviction_policy.h"

#include <algorithm>
#include <cstring>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr size_t kAlign = 2u << 20;   // 2 MB：DMA 效率与实测口径对齐

size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

/// 顺序敏感的 crc32（反射多项式 0xEDB88320），逐位算不查表。
/// 逐出次数是几千的量级，位运算完全够，而"顺序敏感"是这里唯一重要的性质：
/// 项目一实测过 12 次运行 12 个互不相同的指纹 —— 正因为如此，
/// **指纹相同**（而不是不同）才是有信息量的证据。
uint32_t crc32_step(uint32_t crc, uint64_t v) {
    crc = ~crc;
    for (int i = 0; i < 8; ++i) {
        crc ^= static_cast<uint32_t>(v & 0xff);
        v >>= 8;
        for (int b = 0; b < 8; ++b) {
            crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
        }
    }
    return ~crc;
}

}  // namespace

const char* to_string(MigrateResult r) {
    switch (r) {
        case MigrateResult::OK: return "OK";
        case MigrateResult::ERR_GPU_OOM: return "ERR_GPU_OOM";
        case MigrateResult::ERR_CPU_OOM: return "ERR_CPU_OOM";
        case MigrateResult::ERR_BLOCK_NOT_FOUND: return "ERR_BLOCK_NOT_FOUND";
        case MigrateResult::ERR_IN_PROGRESS: return "ERR_IN_PROGRESS";
        case MigrateResult::ERR_INVALID_STATE: return "ERR_INVALID_STATE";
        case MigrateResult::ERR_PINNED: return "ERR_PINNED";
        case MigrateResult::ERR_CUDA: return "ERR_CUDA";
    }
    return "?";
}


struct GpuSwapManager::Impl {
    // ---- 配置 ----
    size_t hot_capacity = 0;      // = n_slots * slot_bytes
    size_t cold_capacity = 0;
    size_t slot_bytes = 0;
    size_t n_slots = 0;

    // ---- 显存 ----
    char* arena = nullptr;        // 对齐后的基址，子分配都从这里切
    void* arena_raw_ptr = nullptr;  // cudaMalloc 的原始指针（cudaFree 必须还它）
    size_t arena_raw = 0;

    // ---- 冷池：按 chunk 惰性 cudaMallocHost，只增不减 ----
    static constexpr size_t kColdChunkSlots = 32;
    std::vector<char*> cold_chunks;      // 按 chunk 序号懒分配
    size_t cold_slots_total = 0;         // n_cold_slots
    size_t cold_slots_used = 0;          // 预算占用（含在途）

    std::vector<int> free_slots;         // 热池空闲槽（后进先出）
    std::vector<int> free_cold;          // 冷池空闲槽

    cudaStream_t swap_stream = nullptr;
    IEvictionPolicy* policy = nullptr;
    LRUPolicy default_policy;
    bool own_policy = false;

    int64_t clock_ms = 0;

    size_t hot_valid_bytes = 0;          // 只有"已落地可读"的部分
    size_t hot_peak_slots = 0;
    int64_t evictions = 0;
    uint32_t fingerprint = 0;
    std::string last_error;              // 粘性

    struct Block {
        BlockDesc d{};
        int slot = -1;                   // 热池槽位，-1 = 不持有
        int cold = -1;                   // 冷池槽位，-1 = 不持有
        bool valid = false;              // 数据可读
        bool migrating = false;
        bool evict_dir = false;          // true = D2H（换出）, false = H2D（换回）
        cudaEvent_t done = nullptr;
        std::vector<cudaEvent_t> reader_events;   // pin 的读者事件，栈式
    };
    std::unordered_map<int64_t, Block> blocks;
    // 在途迁移的 FIFO。**顺序即入队序** —— 单条 stream 上事件按入队序完成，
    // 所以 poll 到第一个 not-ready 就可以停，不必把 N 个事件全 query 一遍。
    std::deque<int64_t> inflight;

    // ---- 工具 ----

    void note(const std::string& msg) { last_error = msg; }

    char* slot_ptr(int i) const {
        return arena + static_cast<size_t>(i) * slot_bytes;
    }

    bool ensure_cold_chunk(size_t chunk_idx) {
        if (chunk_idx < cold_chunks.size() && cold_chunks[chunk_idx]) return true;
        if (chunk_idx >= cold_chunks.size()) cold_chunks.resize(chunk_idx + 1, nullptr);
        size_t bytes = kColdChunkSlots * slot_bytes;
        void* p = nullptr;
        cudaError_t st = cudaMallocHost(&p, bytes);
        if (st != cudaSuccess) {
            note(std::string("cudaMallocHost(") + std::to_string(bytes) +
                 ") 失败: " + cudaGetErrorString(st));
            return false;
        }
        cold_chunks[chunk_idx] = static_cast<char*>(p);
        return true;
    }

    char* cold_ptr(int i) {
        size_t chunk = static_cast<size_t>(i) / kColdChunkSlots;
        if (!ensure_cold_chunk(chunk)) return nullptr;
        return cold_chunks[chunk] +
               (static_cast<size_t>(i) % kColdChunkSlots) * slot_bytes;
    }

    MigrateResult pop_cold_slot(int* out) {
        if (!free_cold.empty()) {
            *out = free_cold.back();
            free_cold.pop_back();
            cold_slots_used++;
            return MigrateResult::OK;
        }
        // 懒增长：只增不减，析构才 free。cudaMallocHost 的文档提醒
        // "Page-locking excessive amounts of memory ... may degrade system
        // performance ... best used sparingly"，所以绝不能一上来就 pin 满。
        size_t next = cold_slots_used + free_cold.size();
        if (next >= cold_slots_total) {
            note("冷池预算已满（" + std::to_string(cold_slots_total) + " 槽）");
            return MigrateResult::ERR_CPU_OOM;
        }
        if (!ensure_cold_chunk(next / kColdChunkSlots)) return MigrateResult::ERR_CPU_OOM;
        *out = static_cast<int>(next);
        cold_slots_used++;
        return MigrateResult::OK;
    }

    void push_cold_slot(int i) {
        free_cold.push_back(i);
        if (cold_slots_used > 0) cold_slots_used--;
    }

    std::vector<BlockStats> evictable(int64_t exclude) const {
        std::vector<BlockStats> out;
        out.reserve(blocks.size());
        for (const auto& kv : blocks) {
            const Block& b = kv.second;
            // 管理器的唯一职责：**这里就是那一道筛子**。
            // 传给策略的集合已经剔除了在途 / 无槽 / 不可读 / pinned 的块，
            // 所以策略不可能选中 pinned 的块，不需要自己再维护保护名单。
            if (!b.d.in_gpu) continue;
            if (b.slot < 0) continue;
            if (b.migrating) continue;
            if (!b.reader_events.empty()) continue;
            if (kv.first == exclude) continue;
            out.push_back(stats_of(kv.first, b));
        }
        return out;
    }

    BlockStats stats_of(int64_t id, const Block& b) const {
        BlockStats s{};
        s.block_id = id;
        s.last_access = b.d.last_access_time;
        s.access_count = b.d.access_count;
        s.in_gpu = b.d.in_gpu;
        s.size = b.d.size;
        s.alloc_time = b.d.alloc_time;
        s.now = clock_ms;
        return s;
    }

    int64_t pick_victim(int64_t exclude) {
        auto cands = evictable(exclude);
        if (cands.empty()) return -1;
        return policy->select_victim(cands);
    }

    // ---- 收割 ----

    /// 收割已完成的迁移。**只在这里翻转状态。**
    /// 不变量：访问器（hot_used_bytes 等）**不调用本函数** —— 读一个统计量
    /// 不该隐式推进状态机，否则"读两次统计量"和"读一次"会给出不同答案。
    void poll() {
        while (!inflight.empty()) {
            int64_t id = inflight.front();
            auto it = blocks.find(id);
            if (it == blocks.end() || !it->second.migrating) {
                inflight.pop_front();      // 不该发生，但别让它卡住队列
                continue;
            }
            Block& b = it->second;
            cudaError_t st = cudaEventQuery(b.done);
            if (st == cudaErrorNotReady) break;   // 入队序即完成序，后面都不用看
            inflight.pop_front();
            if (st == cudaSuccess) {
                finish_ok(b);
            } else {
                note(std::string("异步迁移失败(block ") + std::to_string(id) +
                     "): " + cudaGetErrorString(st));
                finish_err(b);
            }
        }
    }

    void finish_ok(Block& b) {
        if (b.evict_dir) {
            // D2H 落地：数据现在在冷池，热池槽位可以还了。
            // 注意此时把块公开成"不在 GPU"**才是对的** —— 发起的时刻还不能。
            if (b.slot >= 0) {
                free_slots.push_back(b.slot);
                b.slot = -1;
            }
            b.d.gpu_ptr = nullptr;
            b.d.in_gpu = false;
            b.valid = true;               // 冷池副本可用
            if (hot_valid_bytes >= b.d.size) hot_valid_bytes -= b.d.size;
        } else {
            // H2D 落地：数据回到热池。
            b.d.in_gpu = true;
            b.valid = true;
            hot_valid_bytes += b.d.size;
            if (b.cold >= 0) {
                push_cold_slot(b.cold);   // 必须先还，因为 H2D 已经读完
                b.cold = -1;
            }
            b.d.cpu_ptr = nullptr;
        }
        b.migrating = false;
        if (b.done) {
            cudaEventDestroy(b.done);
            b.done = nullptr;
        }
    }

    /// 异步失败的回滚。方向分明：
    ///   换出失败 → 块留在热池，还掉冷槽（数据仍在显存，无损）
    ///   换回失败 → 块留在冷池，还掉热槽（数据仍在 pinned 内存，无损）
    /// 链式过户里 target 的槽位是从 victim 手里接过来的，回滚时要还回去；
    /// 但 victim 的 D2H 排在前面、先被 reap，所以"victim 失败"时那个槽位
    /// 已经物理上属于 target 了 —— 这时 victim 的显存副本**已经丢了**
    /// （target 的 H2D 会覆盖它）。这条路径只有注入故障才走得到，
    /// 真出现就是真的数据丢失，所以标记 valid=false 并留在 last_error 里。
    void finish_err(Block& b) {
        if (b.evict_dir) {
            if (b.cold >= 0) {
                push_cold_slot(b.cold);
                b.cold = -1;
            }
            b.valid = b.slot >= 0;        // 槽位还在就还能读；不在就是真丢了
            b.d.in_gpu = b.valid;
            if (!b.valid) b.d.gpu_ptr = nullptr;
        } else {
            if (b.slot >= 0) {
                free_slots.push_back(b.slot);
                b.slot = -1;
                b.d.gpu_ptr = nullptr;
            }
            b.d.in_gpu = false;
            b.valid = b.cold >= 0;
        }
        b.migrating = false;
        if (b.done) {
            cudaEventDestroy(b.done);
            b.done = nullptr;
        }
    }

    void sync_all() {
        cudaError_t st = cudaStreamSynchronize(swap_stream);
        if (st != cudaSuccess) {
            note(std::string("cudaStreamSynchronize 失败: ") +
                 cudaGetErrorString(st));
        }
        poll();
    }

    int take_free_slot() {
        if (free_slots.empty()) return -1;
        int s = free_slots.back();
        free_slots.pop_back();
        return s;
    }
};


// ---- 生涯 ----

GpuSwapManager::GpuSwapManager(size_t hot_pool_bytes, size_t cold_pool_bytes)
    : impl_(new Impl()) {
    impl_->slot_bytes = kAlign;                     // 默认 2 MB 槽
    impl_->hot_capacity = align_up(hot_pool_bytes, impl_->slot_bytes);
    impl_->n_slots = impl_->hot_capacity / impl_->slot_bytes;
    if (impl_->n_slots == 0) impl_->n_slots = 1;
    impl_->hot_capacity = impl_->n_slots * impl_->slot_bytes;
    impl_->cold_capacity = align_up(cold_pool_bytes, impl_->slot_bytes);
    impl_->cold_slots_total = impl_->cold_capacity / impl_->slot_bytes;

    // 一次 cudaMalloc；多要一个槽用于对齐，之后所有子分配都在里面切。
    impl_->arena_raw = impl_->hot_capacity + impl_->slot_bytes;
    void* raw = nullptr;
    cudaError_t st = cudaMalloc(&raw, impl_->arena_raw);
    if (st != cudaSuccess) {
        impl_->note(std::string("cudaMalloc(") +
                    std::to_string(impl_->arena_raw) +
                    ") 失败: " + cudaGetErrorString(st));
        impl_->arena = nullptr;
    } else {
        uintptr_t p = reinterpret_cast<uintptr_t>(raw);
        uintptr_t a = (p + kAlign - 1) & ~(static_cast<uintptr_t>(kAlign) - 1);
        impl_->arena = reinterpret_cast<char*>(a);
        // cudaFree 必须还原始指针，所以单独记着（见析构）
        impl_->arena_raw_ptr = raw;
    }

    // swap stream 必须是 non-blocking：legacy 默认流会和其它流互相串行，
    // 那样异步就是假的（数据还是对的，只是不重叠）。
    cudaStreamCreateWithFlags(&impl_->swap_stream, cudaStreamNonBlocking);

    impl_->free_slots.reserve(impl_->n_slots);
    for (size_t i = impl_->n_slots; i-- > 0;) {
        impl_->free_slots.push_back(static_cast<int>(i));
    }
    impl_->policy = &impl_->default_policy;
}

GpuSwapManager::~GpuSwapManager() {
    Impl* im = impl_;
    if (!im) return;
    // 顺序有讲究：先等所有迁移落地（别指望 cudaStreamDestroy 帮你等），
    // 再销毁事件，再销毁流，最后才动 arena 和 pinned 内存。
    if (im->swap_stream) {
        cudaStreamSynchronize(im->swap_stream);
        im->poll();
    }
    for (auto& kv : im->blocks) {
        for (cudaEvent_t e : kv.second.reader_events) {
            if (e) cudaEventDestroy(e);
        }
        if (kv.second.done) cudaEventDestroy(kv.second.done);
    }
    if (im->swap_stream) cudaStreamDestroy(im->swap_stream);
    if (im->own_policy) delete im->policy;
    if (im->arena) cudaFree(im->arena_raw_ptr);
    for (char* c : im->cold_chunks) {
        if (c) cudaFreeHost(c);
    }
    delete im;
    impl_ = nullptr;
}


// ---- 分配与释放 ----

BlockDesc* GpuSwapManager::allocate(int64_t block_id, size_t size) {
    Impl* im = impl_;
    im->poll();

    if (size == 0) {
        im->note("allocate: size 不能为 0");
        return nullptr;
    }
    if (size > im->slot_bytes) {
        im->note("allocate: size " + std::to_string(size) +
                 " 超过槽位 " + std::to_string(im->slot_bytes) +
                 "（本管理器只支持定长槽位）");
        return nullptr;
    }
    if (im->blocks.count(block_id)) {
        // 静默覆盖会踩到项目一记过的坑：物理 block_id 会被池子反复复用，
        // 同一个 id 在不同时刻是完全不同的内容。覆盖等于把别人的数据
        // 从账上抹掉，但槽位还占着。
        im->note("allocate: block " + std::to_string(block_id) + " 已存在");
        return nullptr;
    }
    if (!im->arena) {
        im->note("allocate: arena 未建立（构造函数里 cudaMalloc 失败过）");
        return nullptr;
    }

    // 热池满 → 按策略换出，直到有槽位。
    //
    // **这里会阻塞**：allocate 交出去的指针是给**宿主**写的，宿主的写没有
    // stream 顺序，所以不能像 prefetch 那样靠链式过户把槽位提前拿走，
    // 只能等真实槽位空出来。这是有意的取舍 —— 异步路径是 evict/prefetch，
    // 不是 allocate；真实系统里分配发生在调度器里，不在热路径上。
    int s = im->take_free_slot();
    for (size_t guard = 0; s < 0 && guard <= im->blocks.size(); ++guard) {
        int64_t v = im->pick_victim(-1);
        if (v < 0) {
            im->note("热池满且没有可换出的块（全部在途 / 被 pin / 只剩它自己）");
            return nullptr;
        }
        MigrateResult r = evict_to_cold(v);
        if (r != MigrateResult::OK) {
            im->note(std::string("腾空间时换出失败: ") + to_string(r));
            return nullptr;
        }
        im->sync_all();                 // 等真实槽位
        s = im->take_free_slot();
    }
    if (s < 0) {
        im->note("腾空间循环没有收敛");
        return nullptr;
    }

    Impl::Block b;
    b.slot = s;
    b.valid = true;
    b.d.block_id = block_id;
    b.d.size = size;
    b.d.gpu_ptr = im->slot_ptr(s);
    b.d.cpu_ptr = nullptr;
    b.d.in_gpu = true;
    b.d.last_access_time = im->clock_ms;
    b.d.access_count = 0;
    b.d.alloc_time = im->clock_ms;

    auto res = im->blocks.emplace(block_id, std::move(b));
    im->hot_valid_bytes += size;
    if (im->policy) im->policy->on_alloc(block_id, size);

    size_t used_slots = im->n_slots - im->free_slots.size();
    im->hot_peak_slots = std::max(im->hot_peak_slots, used_slots);
    return &res.first->second.d;
}

MigrateResult GpuSwapManager::free(int64_t block_id) {
    Impl* im = impl_;
    im->poll();
    auto it = im->blocks.find(block_id);
    if (it == im->blocks.end()) return MigrateResult::ERR_BLOCK_NOT_FOUND;
    Impl::Block& b = it->second;
    if (b.migrating) {
        // 在途时释放 = 让 cudaFreeHost 去释放一个正在被 DMA 写入的缓冲区。
        // 直接拒绝。
        return MigrateResult::ERR_IN_PROGRESS;
    }
    if (!b.reader_events.empty()) {
        // pinned 的块不允许释放：换出可以靠 stream 顺序做安全，**释放不行** ——
        // 释放之后那块显存会分给别人，读者还在读，就是 use-after-free。
        // 这里 ERR_PINNED 才是对的语义（上面 evict 那里不是）。
        return MigrateResult::ERR_PINNED;
    }
    if (b.slot >= 0) {
        im->free_slots.push_back(b.slot);
        if (b.valid && b.d.in_gpu && im->hot_valid_bytes >= b.d.size) {
            im->hot_valid_bytes -= b.d.size;
        }
    }
    if (b.cold >= 0) im->push_cold_slot(b.cold);
    for (cudaEvent_t e : b.reader_events) {
        if (e) cudaEventDestroy(e);
    }
    if (im->policy) im->policy->on_free(block_id);
    im->blocks.erase(it);
    return MigrateResult::OK;
}


// ---- 迁移 ----

MigrateResult GpuSwapManager::evict_to_cold(int64_t block_id) {
    Impl* im = impl_;
    im->poll();
    auto it = im->blocks.find(block_id);
    if (it == im->blocks.end()) return MigrateResult::ERR_BLOCK_NOT_FOUND;
    Impl::Block& b = it->second;

    if (b.migrating) return MigrateResult::ERR_IN_PROGRESS;
    if (!b.d.in_gpu || b.slot < 0) return MigrateResult::ERR_INVALID_STATE;
    // 注意：pinned 的块**不在这里否决**。pin 是**顺序约束**不是否决权 ——
    // 显式换出照做，只是下面的 cudaStreamWaitEvent 会把 D2H 排到 reader
    // 后面。否决掉的话那个 wait 就永远执行不到，"真锁"也就无从谈起。
    // 自动挑 victim 的那条路（Impl::evictable）会绕开 pinned 的块，
    // 那才是 pin 在自动路径上的作用。

    int cold = -1;
    MigrateResult r = im->pop_cold_slot(&cold);   // 发起时就登记冷池预算
    if (r != MigrateResult::OK) return r;         // 冷池满 → 块留在 GPU，不做二级淘汰

    char* dst = im->cold_ptr(cold);
    if (!dst) {
        im->push_cold_slot(cold);                 // 回滚预留
        return MigrateResult::ERR_CPU_OOM;
    }

    cudaEvent_t ev = nullptr;
    if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) {
        im->push_cold_slot(cold);
        im->note("cudaEventCreate 失败");
        return MigrateResult::ERR_CUDA;
    }

    // pin → 让 swap stream 等读者跑完。**这是唯一的正确做法**：
    // 宿主侧的互斥锁在这里什么都保证不了。
    for (cudaEvent_t re : b.reader_events) {
        if (re) cudaStreamWaitEvent(im->swap_stream, re, 0);
    }

    cudaError_t st = cudaMemcpyAsync(dst, b.d.gpu_ptr, b.d.size,
                                     cudaMemcpyDeviceToHost, im->swap_stream);
    if (st != cudaSuccess) {
        cudaEventDestroy(ev);
        im->push_cold_slot(cold);
        im->note(std::string("cudaMemcpyAsync(D2H) 失败: ") + cudaGetErrorString(st));
        return MigrateResult::ERR_CUDA;
    }
    cudaEventRecord(ev, im->swap_stream);

    b.cold = cold;
    b.d.cpu_ptr = dst;
    b.migrating = true;
    b.evict_dir = true;
    b.done = ev;
    im->inflight.push_back(block_id);

    // 逐出序列指纹在**发起时刻**折进去（不是完成时刻）：这里才是"策略做了
    // 什么决策"的记录点。完成时刻的顺序由 DMA 决定，不反映决策。
    im->evictions++;
    im->fingerprint = crc32_step(im->fingerprint,
                                 static_cast<uint64_t>(block_id));
    return MigrateResult::OK;
}

MigrateResult GpuSwapManager::prefetch_to_hot(int64_t block_id) {
    Impl* im = impl_;
    im->poll();
    auto it = im->blocks.find(block_id);
    if (it == im->blocks.end()) return MigrateResult::ERR_BLOCK_NOT_FOUND;
    Impl::Block& b = it->second;

    if (b.migrating) return MigrateResult::ERR_IN_PROGRESS;
    if (b.d.in_gpu) return MigrateResult::ERR_INVALID_STATE;   // 已经在热池
    if (b.cold < 0 || !b.valid) return MigrateResult::ERR_INVALID_STATE;

    int s = -1;
    int64_t donor = -1;      // 链式过户时槽位的原主；失败要还回去而不是丢掉
    if (!im->free_slots.empty()) {
        s = im->free_slots.back();
        im->free_slots.pop_back();
    } else {
        // 链式过户：同一条 stream 上先 D2H(victim) 再 H2D(target)。
        int64_t v = im->pick_victim(block_id);
        if (v < 0 && !im->inflight.empty()) {
            // 全是"在途"造成的假性无候选：驻留块都在飞，谁都还没落地，
            // 于是 evictable() 把每一个都过滤掉了。一串连续 prefetch 会
            // 每次都撞上这个（实测：6 个槽位、12 个块，只换回得来 6 个）。
            // 这里**只等一次**再重试 —— 等待是有界的，而且此刻 swap stream
            // 本来就饱和了，除了等也没别的能推进状态。
            im->sync_all();
            v = im->pick_victim(block_id);
        }
        if (v < 0) {
            im->note("换回时热池满且没有可换出的块");
            return MigrateResult::ERR_GPU_OOM;
        }
        MigrateResult r = evict_to_cold(v);
        if (r != MigrateResult::OK) return r;
        auto vit = im->blocks.find(v);
        s = vit->second.slot;
        // 过户：victim 不再持有槽位，target 现在持有。同一块物理显存，
        // 占用恒为 1 个槽 —— 因为 D2H 读完之前 H2D 不会开始写。
        vit->second.slot = -1;
        donor = v;
    }

    // 槽位归还。**链式那把必须还给 donor，不能丢进 free_slots**：
    // donor 的 D2H 已经在途、正在读这个槽位，丢进 free_slots 等于让下一个
    // allocate 的宿主写和它并发。还给 donor 的话，它会走正常的 reap 路径，
    // 那时 D2H 才真的读完。
    auto release_slot = [&]() {
        if (s < 0) return;
        if (donor >= 0) {
            im->blocks.find(donor)->second.slot = s;
        } else {
            im->free_slots.push_back(s);
        }
        s = -1;
    };

    cudaEvent_t ev = nullptr;
    if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) {
        release_slot();
        im->note("cudaEventCreate 失败");
        return MigrateResult::ERR_CUDA;
    }

    char* src = im->cold_ptr(b.cold);
    if (!src) {
        cudaEventDestroy(ev);
        release_slot();
        return MigrateResult::ERR_CPU_OOM;
    }
    b.slot = s;
    b.d.gpu_ptr = im->slot_ptr(s);
    s = -1;   // 过户完成，归 target 了

    cudaError_t st = cudaMemcpyAsync(b.d.gpu_ptr, src, b.d.size,
                                     cudaMemcpyHostToDevice, im->swap_stream);
    if (st != cudaSuccess) {
        cudaEventDestroy(ev);
        if (donor >= 0) {
            im->blocks.find(donor)->second.slot = b.slot;
        } else {
            im->free_slots.push_back(b.slot);
        }
        b.slot = -1;
        b.d.gpu_ptr = nullptr;
        im->note(std::string("cudaMemcpyAsync(H2D) 失败: ") + cudaGetErrorString(st));
        return MigrateResult::ERR_CUDA;
    }
    cudaEventRecord(ev, im->swap_stream);

    b.migrating = true;
    b.evict_dir = false;
    b.done = ev;
    im->inflight.push_back(block_id);

    size_t used_slots = im->n_slots - im->free_slots.size();
    im->hot_peak_slots = std::max(im->hot_peak_slots, used_slots);
    return MigrateResult::OK;
}

MigrateResult GpuSwapManager::reserve_cold(size_t n_slots) {
    Impl* im = impl_;
    size_t want = std::min(n_slots, im->cold_slots_total);
    size_t have = im->cold_chunks.size();
    for (size_t i = 0; i < want; i += Impl::kColdChunkSlots) {
        if (i / Impl::kColdChunkSlots >= have) {
            if (!im->ensure_cold_chunk(i / Impl::kColdChunkSlots)) {
                return MigrateResult::ERR_CPU_OOM;
            }
        }
    }
    return MigrateResult::OK;
}

MigrateResult GpuSwapManager::sync() {
    Impl* im = impl_;
    im->sync_all();
    // sync 的返回码只表示"这一步同步本身成不成功"。**异步迁移的错误是
    // 粘在 last_error() 上的**，不在这里清掉 —— 一次失败的迁移不该因为
    // 后面一次成功的 sync 就变得看不见了。要看有没有出过错，读 last_error()。
    return im->inflight.empty() ? MigrateResult::OK : MigrateResult::ERR_CUDA;
}

MigrateResult GpuSwapManager::wait_block(int64_t block_id) {
    Impl* im = impl_;
    auto it = im->blocks.find(block_id);
    if (it == im->blocks.end()) return MigrateResult::ERR_BLOCK_NOT_FOUND;
    if (!it->second.migrating) {
        im->poll();
        return MigrateResult::OK;
    }
    cudaError_t st = cudaEventSynchronize(it->second.done);
    im->poll();
    if (st != cudaSuccess) {
        im->note(std::string("wait_block 失败: ") + cudaGetErrorString(st));
        return MigrateResult::ERR_CUDA;
    }
    return MigrateResult::OK;
}

MigrateResult GpuSwapManager::settle() {
    Impl* im = impl_;
    im->sync_all();
    // 重放一遍 on_alloc，让策略的基价与"当前驻留集"对齐。用在需要一致视图的
    // 场合（例如换了一个策略、或者一段注入的故障结束之后）。
    if (im->policy) {
        for (const auto& kv : im->blocks) {
            if (kv.second.d.in_gpu) im->policy->on_alloc(kv.first, kv.second.d.size);
        }
    }
    return MigrateResult::OK;
}


// ---- pin ----

MigrateResult GpuSwapManager::pin(int64_t block_id, cudaEvent_t reader_done) {
    Impl* im = impl_;
    im->poll();
    auto it = im->blocks.find(block_id);
    if (it == im->blocks.end()) return MigrateResult::ERR_BLOCK_NOT_FOUND;
    // 栈式：每次 pin 追加一个读者事件，unpin 弹一个。**不覆盖** ——
    // 覆盖成"后到的那个"并不安全（后 record 的事件不一定后完成），
    // 全留着、发 D2H 时逐个 wait 才是无条件的正确做法。
    it->second.reader_events.push_back(reader_done);   // 可以是 nullptr（纯宿主保护）
    return MigrateResult::OK;
}

MigrateResult GpuSwapManager::unpin(int64_t block_id) {
    Impl* im = impl_;
    auto it = im->blocks.find(block_id);
    if (it == im->blocks.end()) return MigrateResult::ERR_BLOCK_NOT_FOUND;
    auto& evs = it->second.reader_events;
    if (evs.empty()) return MigrateResult::ERR_INVALID_STATE;
    cudaEvent_t e = evs.back();
    evs.pop_back();
    // 事件的所有权归管理器：pin 之后调用者不用管，unpin/free/析构时销毁。
    // cudaEventDestroy 允许在完成前调用（cuda_runtime_api.h:2976-2979），
    // 所以这里不必等它完成。
    if (e) cudaEventDestroy(e);
    return MigrateResult::OK;
}

MigrateResult GpuSwapManager::touch(int64_t block_id) {
    Impl* im = impl_;
    auto it = im->blocks.find(block_id);
    if (it == im->blocks.end()) return MigrateResult::ERR_BLOCK_NOT_FOUND;
    it->second.d.last_access_time = im->clock_ms;
    it->second.d.access_count++;
    if (im->policy) im->policy->on_access(block_id);
    return MigrateResult::OK;
}


// ---- 配置与统计 ----

void GpuSwapManager::set_policy(IEvictionPolicy* policy) {
    Impl* im = impl_;
    if (im->own_policy && im->policy) delete im->policy;
    im->own_policy = false;
    im->policy = policy ? policy : &im->default_policy;
}

void GpuSwapManager::set_clock(int64_t now_ms) { impl_->clock_ms = now_ms; }
int64_t GpuSwapManager::clock() const { return impl_->clock_ms; }

size_t GpuSwapManager::hot_used_bytes() const {
    Impl* im = impl_;
    // 槽位粒度：**含在途**。在途换出的块显存还没还，在途换回的块槽位已过户，
    // 链式过户下同一块物理显存只算一次 —— 这个数就是真实的占用，不是保守上界。
    return (im->n_slots - im->free_slots.size()) * im->slot_bytes;
}
size_t GpuSwapManager::hot_valid_bytes() const { return impl_->hot_valid_bytes; }
size_t GpuSwapManager::cold_used_bytes() const {
    return impl_->cold_slots_used * impl_->slot_bytes;
}
size_t GpuSwapManager::hot_capacity() const { return impl_->hot_capacity; }
size_t GpuSwapManager::cold_capacity() const { return impl_->cold_capacity; }
size_t GpuSwapManager::slot_bytes() const { return impl_->slot_bytes; }
void* GpuSwapManager::raw_swap_stream() const {
    return reinterpret_cast<void*>(impl_->swap_stream);
}
size_t GpuSwapManager::hot_peak_bytes() const {
    return impl_->hot_peak_slots * impl_->slot_bytes;
}
int GpuSwapManager::num_blocks() const {
    return static_cast<int>(impl_->blocks.size());
}
int GpuSwapManager::num_blocks_in_gpu() const {
    int n = 0;
    for (const auto& kv : impl_->blocks) {
        if (kv.second.d.in_gpu) n++;
    }
    return n;
}
int GpuSwapManager::pending_migrations() const {
    return static_cast<int>(impl_->inflight.size());
}
bool GpuSwapManager::is_migrating(int64_t block_id) const {
    auto it = impl_->blocks.find(block_id);
    return it != impl_->blocks.end() && it->second.migrating;
}
bool GpuSwapManager::is_pinned(int64_t block_id) const {
    auto it = impl_->blocks.find(block_id);
    return it != impl_->blocks.end() && !it->second.reader_events.empty();
}
bool GpuSwapManager::inspect(int64_t block_id, BlockView* out) const {
    auto it = impl_->blocks.find(block_id);
    if (it == impl_->blocks.end()) return false;
    const Impl::Block& b = it->second;
    out->desc = b.d;
    out->valid = b.valid;
    out->has_cold = b.cold >= 0;
    out->migrating = b.migrating;
    out->pinned = !b.reader_events.empty();
    return true;
}
uint32_t GpuSwapManager::eviction_fingerprint() const { return impl_->fingerprint; }
int64_t GpuSwapManager::eviction_count() const { return impl_->evictions; }
const char* GpuSwapManager::last_error() const { return impl_->last_error.c_str(); }
