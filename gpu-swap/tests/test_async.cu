/*
 * test_async.cu — 证明迁移和推理**真的**重叠，以及 pin 是**真的** CUDA 锁
 *
 * 三件事，每件都带负对照 —— 没有负对照的测量分不清"测到了重叠"和
 * "这个方法根本测不出串行"。
 *
 * ── 1. 重叠率 ────────────────────────────────────────────────
 *
 * 判据**不能**写成 `max(t_infer, t_swap) < t_infer + t_swap`：对任意正数
 * a、b 都成立（等价于 min(a,b) > 0），是恒真的空断言，永远不可能失败。
 * 换成跨度口径：
 *
 *     t_union = 两 stream 最早 start 到最晚 end 的跨度
 *     overlap = (t_infer + t_swap - t_union) / min(t_infer, t_swap)
 *
 * 1 = 完全重叠，0 = 完全串行。要求 > 0.5，同时**负对照要求 < 0.2**。
 *
 * 配套的硬约束（少一条测出来的就不是重叠）：
 *   - infer 侧**不能用 NULL stream**：cudaMemcpyAsync 的重叠保证只对
 *     non-zero stream 成立，而且 legacy 默认流会和其它流互相串行。
 *   - 发完立刻断言 cudaStreamQuery(infer) == cudaErrorNotReady，
 *     否则测的是"kernel 早跑完了"。
 *   - 用 K 个背靠背 kernel 保证窗口内确实忙。
 *   - kernel 要**计算受限**（纯 FMA 自旋，不碰显存），否则测到的是
 *     L2/DRAM 带宽争抢，不是"重叠"。
 *   - 计时事件必须用默认 flag；迁移内部的事件用 DisableTiming（更轻）。
 *     **两类事件不能混用一个工厂** —— DisableTiming 的事件传给
 *     cudaEventElapsedTime 直接报 cudaErrorInvalidResourceHandle。
 *
 * ── 2. "立即返回"的非计时验证 ────────────────────────────────
 *
 * 不靠计时的版本更可信：发起整批换出，断言
 *   (a) 每个块 pending_migrations() 还在、in_gpu 还没翻转、预算还没扣
 *   (b) **发起总耗时**远小于"发起+同步"的总耗时
 * 访问器不 poll，所以 (a) 是确定性的，跟拷贝快不快无关。
 *
 * ── 3. pin 是 CUDA 侧的顺序约束（真锁 vs 假锁）────────────────
 *
 * 只加宿主锁是假锁：推理 kernel 在 infer stream 上排队 → 调用者 unpin →
 * 管理器在 swap_stream 上发 D2H，两者之间没有任何 CUDA 依赖，拷贝引擎
 * 读到的正是 kernel 正在写的显存。
 *
 * 验法是**看落进冷池的到底是什么数据**：先让一个 kernel 花几百毫秒把
 * 整块写成 marker，再换出、再换回、再读回来。
 *   - 上了 pin：读回来是 marker（D2H 确实等了 reader_done）
 *   - 不上 pin（负对照）：读回来是旧花样（D2H 跑到 kernel 前面去了）
 * 这个判据不需要计时，也不受机器快慢影响。
 *
 * ⚠️ 口径警告（项目一已经量过）：一次拷贝和一个 kernel 重叠**推不出**
 * "换出是免费的"。D2H 一旦超过约 1.1 GB/s（≈1200 块/s）就把 decode
 * 步长拖出噪声底（kvcache-tracer/measurements/pcie_interference_redline.txt）。
 * 所以下面顺带报出达到的 GB/s，别把重叠率读成"不要钱"。
 */

#include "gpu_swap.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

static int g_pass = 0, g_fail = 0;

#define CHECK(cond)                                                      \
    do {                                                                 \
        if (cond) { ++g_pass; }                                          \
        else {                                                           \
            ++g_fail;                                                    \
            std::printf("  \033[31m✗\033[0m %s:%d  %s\n", __FILE__,      \
                        __LINE__, #cond);                                \
        }                                                                \
    } while (0)

#define CHECK_EQ(a, b)                                                   \
    do {                                                                 \
        long long _a = (long long)(a);                                   \
        long long _b = (long long)(b);                                   \
        if (_a == _b) { ++g_pass; }                                      \
        else {                                                           \
            ++g_fail;                                                    \
            std::printf("  \033[31m✗\033[0m %s:%d  %s == %s (左 = %lld, " \
                        "右 = %lld)\n", __FILE__, __LINE__, #a, #b, _a,  \
                        _b);                                             \
        }                                                                \
    } while (0)

static const size_t SLOT = 2u << 20;

// ============================================================
// kernel
// ============================================================

/// 纯计算自旋：只碰寄存器，不读不写显存。**必须计算受限** ——
/// 一旦让 kernel 去扫显存，量到的就是 L2/DRAM 争抢，跟"PCIe 拷贝能不能
/// 和推理重叠"是两回事。
__global__ void spin_kernel(long long cycles) {
    long long start = clock64();
    double x = 1.0000001;
    while (clock64() - start < cycles) {
        // 一串互相依赖的 FMA，防止被编译器优化掉
        x = fma(x, 1.0000000001, 1e-12);
        x = fma(x, 0.9999999999, 1e-12);
    }
    if (x == 0.0) printf("");          // 保证 x 有副作用
}

/// 访存受限自旋：**故意**去扫一大片显存（256 MB，远大于 4090 的 72 MB L2）。
///
/// 存在的理由是上面那个 kernel 的对偶。上面那个只碰寄存器，于是它和 D2H
/// 拷贝**没有可争的资源** —— 拷贝走 copy engine、读的那点显存带宽可以忽略，
/// 实测出来的"干扰倍数"就是 1.0x。但那个 1.0x **不能外推到真实推理**：
/// attention 是访存受限的，它真的在扫 KV cache。这个 kernel 就是那个情形，
/// 用来把"到底在争什么资源"指出来。
__global__ void memspin_kernel(const float* __restrict__ p, long long cycles) {
    const size_t mask = (size_t{1} << 26) - 1;     // 2^26 个 float = 256 MB
    long long start = clock64();
    size_t stride = (size_t)gridDim.x * blockDim.x * 4;
    size_t i = (((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4) & mask;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    while (clock64() - start < cycles) {
        // 4 个独立 load（合并访存 + 足够 MLP），跨整个 buffer 循环扫描
        a0 += p[(i + 0) & mask];
        a1 += p[(i + 1) & mask];
        a2 += p[(i + 2) & mask];
        a3 += p[(i + 3) & mask];
        i = (i + stride) & mask;
    }
    if (a0 + a1 + a2 + a3 == 1.2345f) printf("");
}

/// 把整块写成 marker。跑在 infer stream 上，用来验 pin。
__global__ void fill_marker_kernel(unsigned int* p, size_t n,
                                   unsigned int marker) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride) p[i] = marker;
}

// ============================================================
// 计时
// ============================================================

static float ms_between(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0;
    if (cudaEventElapsedTime(&ms, a, b) != cudaSuccess) return -1;
    return ms;
}

/// 自校准地把"目标毫秒数"换算成 spin_kernel 的周期数。
///
/// 不用 cudaDeviceProp::clockRate —— CUDA 13 里这个字段已经删了（编译期
/// 直接报错）。就算有，它给的也是标称频率，boost 之后对不上。实测量一次
/// 再线性外推，顺带证明了 spin_kernel 确实是计算受限的。
static long long cycles_for_ms(double target_ms) {
    const long long probe = 200000000LL;
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a);
    spin_kernel<<<256, 256>>>(probe);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = ms_between(a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    if (ms <= 0) return probe;
    double per_ms = static_cast<double>(probe) / ms;
    std::printf("  校准: %.0f 周期/ms（探针 %lld 周期用了 %.2fms）\n",
                per_ms, probe, ms);
    return static_cast<long long>(per_ms * target_ms);
}

struct Span {
    double t_infer = 0, t_swap = 0, t_union = 0, overlap = 0;
    bool valid = false;
};

/// 跑一次"推理 + 换出"，量重叠率。
/// @param serialize true = 负对照：先让 swap stream 等 infer 跑完
static Span measure_overlap(GpuSwapManager& mgr, int n_blocks, long long cycles,
                            bool serialize) {
    Span s;
    cudaStream_t infer;

    // 冷池预热**必须**在 e_inf0 之前。cudaMallocHost 是隐式同步点：
    // 让它落在测量窗口里的话，发起的第一次换出会等设备空闲，
    // 负对照也会跟着"通过"。这不是推理，是实测撞出来的 ——
    // 不预热时一个 250ms 自旋 kernel 能把紧随其后的一次 2MB 换出整个挡住。
    mgr.reserve_cold(static_cast<size_t>(n_blocks));

    cudaStreamCreateWithFlags(&infer, cudaStreamNonBlocking);   // 绝不能是 NULL 流
    cudaStream_t swap = reinterpret_cast<cudaStream_t>(mgr.raw_swap_stream());

    cudaEvent_t e_inf0, e_inf1, e_swp0, e_swp1;
    // **默认 flag**（计时开）。DisableTiming 的事件喂给 cudaEventElapsedTime
    // 会直接报 cudaErrorInvalidResourceHandle。
    cudaEventCreate(&e_inf0); cudaEventCreate(&e_inf1);
    cudaEventCreate(&e_swp0); cudaEventCreate(&e_swp1);

    const int K = 6;
    cudaEventRecord(e_inf0, infer);
    for (int k = 0; k < K; ++k) {
        spin_kernel<<<512, 256, 0, infer>>>(cycles);
    }
    cudaEventRecord(e_inf1, infer);

    // 发完立刻查：kernel 必须还在跑。否则下面量到的"重叠"其实是
    // "infer 早就结束了，swap 在空转"。
    bool infer_busy = (cudaStreamQuery(infer) == cudaErrorNotReady);

    if (serialize) {
        // 负对照：把两段工作**串起来**，重叠率必须掉到 0 附近。
        cudaStreamWaitEvent(swap, e_inf1, 0);
    }

    cudaEventRecord(e_swp0, swap);
    for (int i = 0; i < n_blocks; ++i) {
        mgr.evict_to_cold(i);
    }
    cudaEventRecord(e_swp1, swap);

    cudaStreamSynchronize(infer);
    cudaStreamSynchronize(swap);
    cudaEventSynchronize(e_swp1);
    mgr.sync();

    // 以 e_inf0 为公共原点算跨度
    float rel_swp0 = ms_between(e_inf0, e_swp0);
    float rel_inf1 = ms_between(e_inf0, e_inf1);
    float rel_swp1 = ms_between(e_inf0, e_swp1);
    float t_infer = ms_between(e_inf0, e_inf1);
    float t_swap = ms_between(e_swp0, e_swp1);

    double start = std::min(0.0f, rel_swp0);
    double end = std::max(rel_inf1, rel_swp1);
    s.t_infer = t_infer;
    s.t_swap = t_swap;
    s.t_union = end - start;
    double denom = std::min(s.t_infer, s.t_swap);
    s.overlap = denom > 0 ? (s.t_infer + s.t_swap - s.t_union) / denom : 0.0;
    s.valid = infer_busy && denom > 0;

    cudaEventDestroy(e_inf0); cudaEventDestroy(e_inf1);
    cudaEventDestroy(e_swp0); cudaEventDestroy(e_swp1);
    cudaStreamDestroy(infer);
    return s;
}

// ============================================================
// 1. 重叠率 + 负对照
// ============================================================
static void test_overlap(long long cycles) {
    std::printf("\n── 重叠率（含负对照）\n");

    const int N = 48;                       // 48 × 2 MB = 96 MB 的 D2H
    GpuSwapManager pos(N * SLOT + SLOT, 4 * N * SLOT);
    for (int i = 0; i < N; ++i) pos.allocate(i, SLOT);
    Span a = measure_overlap(pos, N, cycles, /*serialize=*/false);

    GpuSwapManager neg(N * SLOT + SLOT, 4 * N * SLOT);
    for (int i = 0; i < N; ++i) neg.allocate(i, SLOT);
    Span b = measure_overlap(neg, N, cycles, /*serialize=*/true);

    std::printf("  重叠(并行)  infer=%.2fms swap=%.2fms union=%.2fms → overlap=%.3f\n",
                a.t_infer, a.t_swap, a.t_union, a.overlap);
    std::printf("  负对照(串行) infer=%.2fms swap=%.2fms union=%.2fms → overlap=%.3f\n",
                b.t_infer, b.t_swap, b.t_union, b.overlap);

    double gbps = (N * (double)SLOT) / (a.t_swap / 1000.0) / 1e9;
    std::printf("  达到带宽: %.2f GB/s  （项目一的干扰红线 ≈ 1.1 GB/s —— "
                "重叠 ≠ 免费）\n", gbps);
    std::printf("  块速率: %.0f 块/s  （红线 ≈ 1200 块/s）\n",
                N / (a.t_swap / 1000.0));

    CHECK(a.valid);
    CHECK(a.overlap > 0.5);        // 真的重叠了
    CHECK(b.overlap < 0.2);        // 负对照：串起来之后就不重叠了
    CHECK(a.overlap > b.overlap);  // 正负对照必须分得开
    CHECK(a.t_infer > 5.0);        // infer 侧确实忙了足够久，不是空转
}

// ============================================================
// 2. "立即返回"（不靠计时）
// ============================================================
static void test_immediate_return() {
    std::printf("\n── 换出立即返回（访问器不 poll，所以这是确定性判据）\n");
    const int N = 32;
    GpuSwapManager m(N * SLOT + SLOT, 4 * N * SLOT);
    CHECK(m.reserve_cold(N) == MigrateResult::OK);   // 见 measure_overlap 里的说明
    for (int i = 0; i < N; ++i) CHECK(m.allocate(i, SLOT) != nullptr);

    // ---- 单块：唯一能做确定性断言的口径 ----
    //
    // 批量不行。`evict_to_cold` 入口会收割**已经跑完**的在途迁移（有界、
    // 非阻塞），所以批到第 k 个时前面若干个已经落地了，pending 必然 < N。
    // 这不是缺陷，是"每调用一次推进一次状态机"的正常代价；但拿它当
    // "全部仍在途"的证据就是错的（实测 pending=31/32，差的正是第一个）。
    // 单块没这个问题：发起到断言之间没有任何一次管理器调用。
    const size_t used_before = m.hot_used_bytes();
    CHECK_EQ(used_before, static_cast<size_t>(N) * SLOT);

    CHECK(m.evict_to_cold(0) == MigrateResult::OK);
    CHECK_EQ(m.pending_migrations(), 1);             // 在途
    CHECK_EQ(m.hot_used_bytes(), used_before);       // 显存还没还
    CHECK_EQ(m.num_blocks_in_gpu(), N);              // in_gpu 还没翻转
    {
        BlockView v;
        CHECK(m.inspect(0, &v));
        CHECK(v.desc.in_gpu);
        CHECK(v.migrating);
    }
    // 在途时重复发起要被挡住，否则同一块会被两条 D2H 拷两遍
    CHECK(m.evict_to_cold(0) == MigrateResult::ERR_IN_PROGRESS);

    CHECK(m.wait_block(0) == MigrateResult::OK);
    CHECK_EQ(m.pending_migrations(), 0);
    CHECK_EQ(m.num_blocks_in_gpu(), N - 1);
    CHECK_EQ(m.hot_used_bytes(), used_before - SLOT);
    CHECK_EQ(m.cold_used_bytes(), SLOT);

    // ---- 批量：发起远快于落地 ----
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 1; i < N; ++i) {
        MigrateResult r = m.evict_to_cold(i);
        if (r != MigrateResult::OK) {
            std::printf("    第 %d 个换出失败: %s\n", i, to_string(r));
        }
    }
    const auto t_issued = std::chrono::steady_clock::now();

    // 唯一稳的不变量：已落地的可以扣账，没落地的绝不能提前扣 ——
    // 所以预算在 (0, 全部] 之间，而不是"一定等于全部"。
    CHECK(m.hot_used_bytes() > 0);
    CHECK(m.hot_used_bytes() <= (static_cast<size_t>(N) - 1) * SLOT);

    m.sync();
    const auto t_done = std::chrono::steady_clock::now();
    CHECK_EQ(m.pending_migrations(), 0);
    CHECK_EQ(m.num_blocks_in_gpu(), 0);
    CHECK_EQ(m.hot_used_bytes(), 0);
    CHECK_EQ(m.cold_used_bytes(), static_cast<size_t>(N) * SLOT);

    auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    const double issue_ms = ms(t0, t_issued);
    const double total_ms = ms(t0, t_done);
    std::printf("  发起 %.1f MB 用了 %.3fms（约合 %.0f GB/s 的提交速率）；"
                "等它落地总共 %.2fms（%.1f×）\n",
                (N - 1) * (double)SLOT / 1e6, issue_ms,
                (N - 1) * (double)SLOT / issue_ms / 1e6,
                total_ms, total_ms / std::max(issue_ms, 1e-3));
    // 同步实现的话发起耗时就该≈总耗时。留 3 倍余量。
    // 用 steady_clock 而不是 clock()：sync() 会在宿主侧空等，
    // clock() 量的是 CPU 时间，阻塞式同步下这个比值会假性变大。
    CHECK(total_ms > issue_ms * 3);
}

// ============================================================
// 3. pin 是真 CUDA 锁（不是宿主互斥）
// ============================================================
static bool pin_marker_survives(bool use_pin, long long cycles,
                                unsigned int* out_first) {
    GpuSwapManager m(8 * SLOT, 32 * SLOT);
    // 预热（同上）：冷池分配若落在窗口里，D2H 会被 cudaMallocHost 拖到
    // 设备空闲之后，负对照就变成假绿 —— 这正是之前实测到的现象。
    if (m.reserve_cold(1) != MigrateResult::OK) return false;
    if (!m.allocate(1, SLOT)) return false;

    BlockView v;
    m.inspect(1, &v);
    unsigned int* dev = static_cast<unsigned int*>(v.desc.gpu_ptr);
    size_t n_words = SLOT / sizeof(unsigned int);

    // 先铺一层"旧花样"，这样如果 D2H 抢在 kernel 前面跑，冷池里就是它。
    std::vector<unsigned int> old(n_words, 0x0BADF00Du);
    cudaMemcpy(dev, old.data(), SLOT, cudaMemcpyHostToDevice);

    cudaStream_t infer;
    cudaStreamCreateWithFlags(&infer, cudaStreamNonBlocking);

    // 同一个 stream 上：先自旋几百毫秒，再把整块改写成 marker。
    spin_kernel<<<64, 256, 0, infer>>>(cycles);
    fill_marker_kernel<<<(int)(n_words / 256), 256, 0, infer>>>(dev, n_words, 0xDEADBEEFu);

    cudaEvent_t reader_done;
    cudaEventCreateWithFlags(&reader_done, cudaEventDisableTiming);
    cudaEventRecord(reader_done, infer);      // 必须在 pin **之前** record

    if (use_pin) {
        if (m.pin(1, reader_done) != MigrateResult::OK) return false;
    } else {
        cudaEventDestroy(reader_done);        // 负对照：不要这个约束
    }

    if (m.evict_to_cold(1) != MigrateResult::OK) return false;
    m.sync();
    if (m.prefetch_to_hot(1) != MigrateResult::OK) return false;
    m.sync();

    std::vector<unsigned int> got(n_words, 0);
    m.inspect(1, &v);
    cudaMemcpy(got.data(), v.desc.gpu_ptr, SLOT, cudaMemcpyDeviceToHost);
    *out_first = got[0];
    bool all_marker = true;
    for (size_t i = 0; i < n_words; ++i) {
        if (got[i] != 0xDEADBEEFu) { all_marker = false; break; }
    }
    if (use_pin) m.unpin(1);                  // 事件的所有权在管理器手里
    cudaStreamDestroy(infer);
    return all_marker;
}

static void test_pin_is_cuda_side(long long cycles) {
    std::printf("\n── pin 是 CUDA 侧的顺序约束（真锁 vs 假锁）\n");
    unsigned int f_pos = 0, f_neg = 0;
    bool pos = pin_marker_survives(true, cycles, &f_pos);
    bool neg = pin_marker_survives(false, cycles, &f_neg);

    std::printf("  上了 pin ：落进冷池再读回来 = %s (首字 0x%08X)\n",
                pos ? "marker ✓" : "旧花样 ✗", f_pos);
    std::printf("  不上 pin ：落进冷池再读回来 = %s (首字 0x%08X)\n",
                neg ? "marker" : "旧花样 ✓(负对照如期)", f_neg);

    CHECK(pos);       // 有 pin → D2H 必须等 kernel 写完
    CHECK(!neg);      // 没 pin → D2H 抢跑，读到旧数据。这正说明 pin 是必需的。
}

// ============================================================
// 4. 编码器压力：换出与 kernel 并发跑一批，数据仍逐块正确
// ============================================================
static void test_concurrent_integrity(long long cycles) {
    std::printf("\n── 并发下的数据完整性（kernel 跑着的同时换出+换回）\n");
    const int N = 24;
    GpuSwapManager m(N * SLOT + SLOT, 4 * N * SLOT);

    // 每块写一个可辨认的花样
    for (int i = 0; i < N; ++i) {
        if (!m.allocate(i, SLOT)) { CHECK(false); return; }
        BlockView v;
        m.inspect(i, &v);
        std::vector<unsigned int> buf(SLOT / 4, 0xC0DE0000u + (unsigned)i);
        cudaMemcpy(v.desc.gpu_ptr, buf.data(), SLOT, cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();

    cudaStream_t infer;
    cudaStreamCreateWithFlags(&infer, cudaStreamNonBlocking);
    spin_kernel<<<256, 256, 0, infer>>>(cycles);

    for (int i = 0; i < N; ++i) m.evict_to_cold(i);
    m.sync();
    for (int i = 0; i < N; ++i) m.prefetch_to_hot(i);
    m.sync();
    cudaStreamSynchronize(infer);

    int ok = 0;
    for (int i = 0; i < N; ++i) {
        BlockView v;
        m.inspect(i, &v);
        std::vector<unsigned int> got(SLOT / 4, 0);
        cudaMemcpy(got.data(), v.desc.gpu_ptr, SLOT, cudaMemcpyDeviceToHost);
        bool good = true;
        for (size_t k = 0; k < got.size(); ++k) {
            if (got[k] != 0xC0DE0000u + (unsigned)i) { good = false; break; }
        }
        if (good) ++ok;
    }
    CHECK_EQ(ok, N);
    cudaStreamDestroy(infer);
}


// ============================================================
// 5. 重叠 ≠ 免费：算力跑着的时候，D2H 还能跑到多少
// ============================================================
//
// 上面测出了"重叠率 = 1.000"，很容易顺手读成"换出不要钱"。**读错了。**
// 重叠说的是两段工作**在时间上**不互相等待，没说带宽不受影响。
//
// 项目一量过这条红线：D2H 超过约 1.1 GB/s（≈1200 块/s）就把 decode 步长
// 拖出噪声底（kvcache-tracer/measurements/pcie_interference_redline.txt，
// baseline 步中位数 15.404 ms，噪声底 0.020 ms）。这里从**另一边**量同一件事：
// 算力占着 GPU 的时候，同一批拷贝还能跑多快。
//
// 两次测量唯一的差别是有没有 kernel 在跑，所以差值就是"干扰"。
static void test_interference(long long cycles, int n_blocks) {
    std::printf("\n── 算力干扰：重叠 ≠ 免费（同一批拷贝，只差有没有 kernel 在跑）\n");

    const size_t bytes = static_cast<size_t>(n_blocks) * SLOT;
    const int reps = 3;
    // 0 = 空载，1 = 计算受限 kernel，2 = 访存受限 kernel
    const int NKINDS = 3;
    double gbps[NKINDS][reps] = {{0}};
    int live[NKINDS] = {0, 0, 0};      // 拷贝结束时 kernel 仍在跑的重复次数
    int measured[NKINDS] = {reps, reps, reps};

    // 访存受限那档要扫的 256 MB。**在计时窗口外**分配 —— cudaMalloc 是隐式
    // 同步点，放进窗口里会把要量的东西毁掉（见 README 第 1 条）。
    float* mem_arena = nullptr;
    const size_t mem_floats = size_t{1} << 26;
    if (cudaMalloc(&mem_arena, mem_floats * sizeof(float)) == cudaSuccess) {
        cudaMemset(mem_arena, 0, mem_floats * sizeof(float));   // 让页真的落地
    } else {
        mem_arena = nullptr;
        std::printf("  （跳访存受限那档：256 MB 的扫描缓冲分配失败）\n");
    }

    for (int with_compute = 0; with_compute < NKINDS; ++with_compute) {
        if (with_compute == 2 && mem_arena == nullptr) break;
        for (int r = 0; r < reps; ++r) {
            GpuSwapManager m(n_blocks * SLOT + SLOT, 4 * (size_t)n_blocks * SLOT);
            m.reserve_cold(n_blocks);    // 见 measure_overlap 里的说明
            for (int i = 0; i < n_blocks; ++i) m.allocate(i, SLOT);

            cudaStream_t infer = nullptr;
            if (with_compute) {
                cudaStreamCreateWithFlags(&infer, cudaStreamNonBlocking);
                // 6 个背靠背的 kernel，保证拷贝全程 GPU 都忙
                for (int k = 0; k < 6; ++k) {
                    if (with_compute == 1) {
                        spin_kernel<<<512, 256, 0, infer>>>(cycles);
                    } else {
                        memspin_kernel<<<2048, 256, 0, infer>>>(mem_arena, cycles);
                    }
                }
            }

            cudaStream_t swap = reinterpret_cast<cudaStream_t>(m.raw_swap_stream());
            cudaEvent_t t0, t1;
            cudaEventCreate(&t0); cudaEventCreate(&t1);
            cudaEventRecord(t0, swap);
            for (int i = 0; i < n_blocks; ++i) m.evict_to_cold(i);
            cudaEventRecord(t1, swap);
            cudaEventSynchronize(t1);

            // ⚠️ 这里是这个测试最容易骗人的地方，必须显式校验。
            //
            // 拷贝结束时，推理 kernel **必须还在跑**。否则这次量的是
            // "空载 vs 空载"，倍数当然 ≈1.0x，而报告会把它当成
            // "算力不干扰拷贝"的结论读出来 —— 一个**看起来合理**的错误结论。
            //
            // 实测撞到过：连跑 5 次，2 次给 1.2x、3 次给 1.0x。光看数字
            // 无法判断哪个是真的。所以判据不能是"倍数是多少"，而是
            // "这次测量有没有量到东西"。
            bool overlapping =
                (infer != nullptr) &&
                (cudaStreamQuery(infer) == cudaErrorNotReady);

            m.sync();
            float ms = 0;
            cudaEventElapsedTime(&ms, t0, t1);

            // 无 kernel 那一轮没有"是否重叠"可校验，恒有效；
            // 有 kernel 那一轮必须**确实重叠着**才算数。
            bool valid = (ms > 0) && (infer == nullptr || overlapping);
            if (valid) {
                gbps[with_compute][r] = bytes / (ms / 1000.0) / 1e9;
                if (infer != nullptr) ++live[with_compute];
            } else {
                --measured[with_compute];      // 这次没量到东西，不算
            }

            if (infer) {
                cudaStreamSynchronize(infer);
                cudaStreamDestroy(infer);
            }
            cudaEventDestroy(t0); cudaEventDestroy(t1);

            // 这一轮必须是干净的：全部落地
            CHECK_EQ(m.num_blocks_in_gpu(), 0);
        }
    }

    // 中位数：单次采样在这个量上噪声很明显（同一二进制实测 20.2–24.2 GB/s
    // 的双峰分布），报一个单点等于报运气。
    auto median = [](double* v, int n) {
        if (n <= 0) return 0.0;
        std::sort(v, v + n);
        return v[n / 2];
    };
    double idle = median(gbps[0], measured[0]);
    double busy = median(gbps[1], measured[1]);
    double mem = median(gbps[2], measured[2]);

    std::printf("  空载（无 kernel）            : %6.2f GB/s  (%.0f 块/s)"
                "   [%d/%d 次有效]\n",
                idle, n_blocks / (bytes / idle / 1e9), measured[0], reps);
    std::printf("  算力受限（6 个只碰寄存器的 kernel）: %6.2f GB/s  (%.0f 块/s)"
                "   [%d/%d 次有效，%d 次确认重叠]\n",
                busy, n_blocks / (bytes / busy / 1e9), measured[1], reps,
                live[1]);
    if (measured[2] > 0) {
        std::printf("  访存受限（6 个扫 256MB 的 kernel） : %6.2f GB/s  (%.0f 块/s)"
                    "   [%d/%d 次有效，%d 次确认重叠]\n",
                    mem, n_blocks / (bytes / mem / 1e9), measured[2], reps,
                    live[2]);
    }
    std::printf("  干扰倍数: 算力受限 %.2fx / 访存受限 %.2fx"
                "   项目一红线 ≈ 1.1 GB/s / 1200 块/s\n",
                idle / std::max(busy, 1e-9), idle / std::max(mem, 1e-9));

    // 判据：只**报告**干扰，不设"必须重叠"的断言 —— 重叠率已由 test_overlap
    // 单独验过。这里要挡的是"没测到任何东西"：
    CHECK(idle > 1.0);           // 空载跑出个像样的带宽
    CHECK(busy > 0.05);          // 有干扰也不是 0（不是把拷贝卡死了）
    CHECK(measured[1] > 0);      // "算力受限"那档确实在 kernel 跑着时量的
    if (mem_arena) {
        CHECK(measured[2] > 0);  // "访存受限"那档同理
        cudaFree(mem_arena);
    }
}

/// 固定**工作量**（不是固定时间）的访存 kernel。跑固定的圈数，于是它的耗时
/// 直接反映"被打扰了多少" —— 这正是项目一红线的口径：他们量的是 decode
/// 步长在 D2H 流量下的变化，不是拷贝自己有多快。
__global__ void memwork_kernel(const float* __restrict__ p, int iters,
                               float* __restrict__ sink) {
    const size_t mask = (size_t{1} << 26) - 1;
    size_t stride = (size_t)gridDim.x * blockDim.x * 4;
    size_t i = (((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4) & mask;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (int k = 0; k < iters; ++k) {
        a0 += p[(i + 0) & mask];
        a1 += p[(i + 1) & mask];
        a2 += p[(i + 2) & mask];
        a3 += p[(i + 3) & mask];
        i = (i + stride) & mask;
    }
    if (a0 + a1 + a2 + a3 == 1.2345f) *sink = a0;
}

// 上面那两组量的是"算力占了 GPU，拷贝还能跑多快"。方向是**拷贝侧**。
//
// 但项目一的红线是**推理侧**的：D2H 超过 1.1 GB/s 就把 decode 步长拖出
// 0.020 ms 的噪声底（基线步长中位数 15.404 ms —— 也就是 0.13% 的敏感度）。
// 这两个量不是一回事：拷贝保持了带宽，不等于推理没被拖慢。
//
// 所以这里反过来量一次：**同一批 kernel，有没有拷贝在同时跑，耗时差多少。**
// 这是本仓库里唯一一个能直接对上项目一那条红线的测量。
static void test_copy_slows_compute(int n_blocks) {
    std::printf("\n── 反向：拷贝有没有拖慢推理（项目一红线的同一口径）\n");

    const size_t mem_floats = size_t{1} << 26;      // 256 MB，远大于 L2
    float* arena = nullptr;
    float* sink = nullptr;
    if (cudaMalloc(&arena, mem_floats * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&sink, sizeof(float)) != cudaSuccess) {
        std::printf("  （跳过：扫描缓冲分配失败）\n");
        if (arena) cudaFree(arena);
        if (sink) cudaFree(sink);
        return;
    }
    cudaMemset(arena, 0, mem_floats * sizeof(float));

    // 调出来的：让 kernel 的耗时**盖住整个拷贝窗口**（实测 400 圈 ≈ 1.2 ms，
    // 而 96 MB 拷贝要 4.8 ms —— 那样只重叠了 1/4 的窗口，"被拖慢多少"会被
    // 摊薄，读数偏小）。1600 圈 ≈ 4.8 ms，和拷贝等长。
    const int iters = 1600;
    const int reps = 5;
    double best[2] = {1e18, 1e18};      // 0 = 无拷贝，1 = 有拷贝
    double copy_ms = 0;

    {
        for (int r = 0; r < reps; ++r) {
            // ⚠️ 两种条件**交错**测，不是先跑完 A 再跑完 B。
            // 分开跑的话，两批之间的 GPU 状态漂移（clocks/温度/别的进程）
            // 会整个算进"差值"里 —— 实测撞到过负的拖慢率（-0.15%），
            // 而拖慢在物理上不可能为负，那个负数就是漂移的读数。
            for (int with_copy = 0; with_copy < 2; ++with_copy) {
                GpuSwapManager m(n_blocks * SLOT + SLOT,
                                 4 * (size_t)n_blocks * SLOT);
                m.reserve_cold(n_blocks);
                for (int i = 0; i < n_blocks; ++i) m.allocate(i, SLOT);

                cudaStream_t infer = nullptr;
                cudaStreamCreateWithFlags(&infer, cudaStreamNonBlocking);
                cudaEvent_t a, b, c, d;
                cudaEventCreate(&a); cudaEventCreate(&b);
                cudaEventCreate(&c); cudaEventCreate(&d);

                cudaEventRecord(a, infer);
                memwork_kernel<<<2048, 256, 0, infer>>>(arena, iters, sink);
                cudaEventRecord(b, infer);

                cudaStream_t swap =
                    reinterpret_cast<cudaStream_t>(m.raw_swap_stream());
                if (with_copy) {
                    cudaEventRecord(c, swap);
                    // 96 MB 的 D2H ≈ 21 GB/s，是项目一红线（1.1 GB/s）的 19 倍
                    for (int i = 0; i < n_blocks; ++i) m.evict_to_cold(i);
                    cudaEventRecord(d, swap);
                    cudaEventSynchronize(d);
                    float t = 0;
                    cudaEventElapsedTime(&t, c, d);
                    if (t > copy_ms) copy_ms = t;    // 取最大，别让一次抖动缩小它
                }

                cudaEventSynchronize(b);
                float t = 0;
                cudaEventElapsedTime(&t, a, b);
                if (t > 0 && t < best[with_copy]) best[with_copy] = t;

                m.sync();
                cudaStreamSynchronize(infer);
                cudaStreamDestroy(infer);
                cudaEventDestroy(a); cudaEventDestroy(b);
                cudaEventDestroy(c); cudaEventDestroy(d);
                // 只有跑过拷贝的那些轮才该全部落地；没拷贝的本就还在热池。
                if (with_copy) CHECK_EQ(m.num_blocks_in_gpu(), 0);
            }
        }
    }

    double overhead = (best[1] - best[0]) / best[0] * 100.0;
    std::printf("  推理 kernel 单独跑          : %6.2f ms\n", best[0]);
    std::printf("  与 %zu MB 的 D2H 同时跑     : %6.2f ms"
                "   （拷贝本身 %.2f ms，%.1f GB/s）\n",
                n_blocks * SLOT >> 20, best[1], copy_ms,
                (n_blocks * SLOT / (copy_ms / 1000.0)) / 1e9);
    std::printf("  推理被拖慢: %+.2f%%   （项目一红线的敏感度是 0.13%%）\n",
                overhead);

    // 判据：这只是**报告**，不设阈值断言 —— 具体多少取决于这台机器的
    // 码流和负载。要挡的是"没量到东西"：两个数都得是正的、像样的。
    CHECK(best[0] > 0.1);
    CHECK(best[1] > 0.1);

    cudaFree(arena);
    cudaFree(sink);
}

int main() {
    std::printf("=== gpu-swap 异步测试 ===\n");
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        std::printf("没有可用的 CUDA 设备\n");
        return 77;
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    std::printf("设备: %s (sm_%d%d, %d SM, %.1f GB)\n", prop.name, prop.major,
                prop.minor, prop.multiProcessorCount,
                prop.totalGlobalMem / 1073741824.0);

    // 每个 kernel ≈ 40ms（6 个背靠背 ≈ 240ms，够 D2H 跑完 96 MB）。
    // 换算用自校准，不查标称频率（CUDA 13 里 clockRate 字段已经没了）。
    long long cycles = cycles_for_ms(40.0);

    test_immediate_return();
    test_overlap(cycles);
    test_pin_is_cuda_side(cycles);
    test_concurrent_integrity(cycles);
    test_interference(cycles, 48);        // 48 × 2 MB = 96 MB
    test_copy_slows_compute(48);          // 反过来量：拷贝拖慢推理多少

    std::printf("\n=== 通过 %d，失败 %d ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
