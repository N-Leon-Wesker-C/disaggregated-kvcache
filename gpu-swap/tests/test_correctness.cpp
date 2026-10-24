/*
 * test_correctness.cpp — 数据一致性、边界、错误路径
 *
 * 这个文件里**每一条"看起来多余"的断言都是被一条具体的假阳性路径逼出来的**。
 * 三条最要紧的：
 *
 * 1. **"数据压根没搬也全绿"。** 如果实现从不真释放 GPU 副本，"写花样 → 换出
 *    → 换回 → 校验"会全过。所以每个换出用例都断言**可观测副作用**：
 *    in_gpu 翻转、hot_used 归零、cold_used == 一个槽位、sync 前
 *    pending_migrations() > 0。再加一条更强的：换出后分配新块，断言它
 *    **落在刚释放的那个地址上**。
 *
 * 2. **"verify 不会失败"。** 必须有**故意用错花样**去 verify 的用例，
 *    断言它确实报不一致。否则 verify 里任何 bug 都会让"数据一致"的验收
 *    空转通过。
 *
 * 3. **"换回后校验通过"可能只是因为数据从没离开过那块显存。**
 *    arena 方案下地址一定会复用，所以换出完成后要**往那个槽位写毒化花样**，
 *    再换回，再校验。
 *
 * 另外一条设计上的好处值得点出来：**访问器不 poll**。所以
 * `evict(); pending_migrations()` 这个组合是**确定性**的 —— 无论 4 KB 的
 * 拷贝在不在飞，计数都是 1，因为没有任何东西会去收割它。异步契约因此
 * 可以在没有 kernel、没有计时的前提下被严格验证。
 */

#include "gpu_swap.h"
#include "eviction_policy.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_pass = 0, g_fail = 0;
static const char* g_section = "";

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (cond) { ++g_pass; }                                            \
        else {                                                             \
            ++g_fail;                                                      \
            std::printf("  \033[31m✗\033[0m [%s] %s:%d  %s\n", g_section, \
                        __FILE__, __LINE__, #cond);                        \
        }                                                                  \
    } while (0)

// 统一转成 long long 再比：不然 size_t 和字面量 0 之间会刷一屏
// -Wsign-compare，把真正的错误淹掉。
#define CHECK_EQ(a, b)                                                     \
    do {                                                                   \
        long long _a = (long long)(a);                                     \
        long long _b = (long long)(b);                                     \
        if (_a == _b) { ++g_pass; }                                        \
        else {                                                             \
            ++g_fail;                                                      \
            std::printf("  \033[31m✗\033[0m [%s] %s:%d  %s == %s  "        \
                        "(左 = %lld, 右 = %lld)\n", g_section, __FILE__,   \
                        __LINE__, #a, #b, _a, _b);                         \
        }                                                                  \
    } while (0)

static void section(const char* s) {
    g_section = s;
    std::printf("\n── %s\n", s);
}

static constexpr size_t BLK = 4096;      // 块大小（远小于槽位，走"部分占用"路径）

/// 8 个热槽 / 16 个冷槽。槽位粒度由 ctor 决定（2 MB），预算给整数倍即可。
static constexpr size_t HOT = 8 * (2u << 20);
static constexpr size_t COLD = 16 * (2u << 20);

static void fill(GpuSwapManager& m, int64_t id, uint64_t seed) {
    BlockView v;
    if (!m.inspect(id, &v) || !v.desc.gpu_ptr) {
        std::printf("  (fill: block %lld 不在显存)\n", (long long)id);
        return;
    }
    // 用宿主的确定性花样：**不经管理器**，直接写显存。
    std::vector<unsigned char> buf(v.desc.size);
    uint64_t x = seed;
    for (size_t i = 0; i < buf.size(); ++i) {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        buf[i] = static_cast<unsigned char>(x >> 33);
    }
    cudaMemcpy(v.desc.gpu_ptr, buf.data(), buf.size(), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

static int64_t verify(GpuSwapManager& m, int64_t id, uint64_t seed) {
    BlockView v;
    if (!m.inspect(id, &v) || !v.desc.gpu_ptr) return -1;
    std::vector<unsigned char> got(v.desc.size), want(v.desc.size);
    uint64_t x = seed;
    for (size_t i = 0; i < want.size(); ++i) {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        want[i] = static_cast<unsigned char>(x >> 33);
    }
    cudaMemcpy(got.data(), v.desc.gpu_ptr, got.size(), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    int64_t bad = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] != want[i]) ++bad;
    }
    return bad;
}


// ============================================================
// 1. 换出的可观测副作用 + 数据一致性
// ============================================================
static void test_evict_prefetch_roundtrip() {
    section("换出/换回：副作用 + 数据一致");
    GpuSwapManager m(HOT, COLD);
    const size_t SLOT = m.slot_bytes();
    CHECK(SLOT > 0);

    BlockDesc* d = m.allocate(1, BLK);
    CHECK(d != nullptr);
    CHECK_EQ(m.hot_used_bytes(), SLOT);
    CHECK_EQ(m.hot_valid_bytes(), BLK);
    uint64_t addr0 = reinterpret_cast<uint64_t>(d->gpu_ptr);
    CHECK(addr0 != 0);
    fill(m, 1, 0xABCDEF01u);

    // ---- 换出：立即返回 ----
    CHECK(m.evict_to_cold(1) == MigrateResult::OK);

    // 这三条是**确定性**的：访问器不 poll，所以没人会来收割。
    CHECK_EQ(m.pending_migrations(), 1);
    CHECK(m.is_migrating(1));
    CHECK_EQ(m.hot_used_bytes(), SLOT);          // 显存还没还，预算继续算在热池
    {
        BlockView v;
        CHECK(m.inspect(1, &v));
        CHECK(v.desc.in_gpu);                    // 在途时 in_gpu **不翻转** —— 异步语义
        CHECK(v.valid);
    }

    // ---- 落地 ----
    CHECK(m.sync() == MigrateResult::OK);
    CHECK_EQ(m.pending_migrations(), 0);
    {
        BlockView v;
        CHECK(m.inspect(1, &v));
        CHECK(!v.desc.in_gpu);
        CHECK(v.valid);                          // 冷池副本可用
        CHECK(v.has_cold);
        CHECK(v.desc.gpu_ptr == nullptr);
    }
    CHECK_EQ(m.hot_used_bytes(), 0);             // 槽位真的还了
    CHECK_EQ(m.hot_valid_bytes(), 0);
    CHECK_EQ(m.cold_used_bytes(), SLOT);

    // ---- 换回 ----
    CHECK(m.prefetch_to_hot(1) == MigrateResult::OK);
    CHECK_EQ(m.pending_migrations(), 1);
    CHECK(m.sync() == MigrateResult::OK);
    {
        BlockView v;
        CHECK(m.inspect(1, &v));
        CHECK(v.desc.in_gpu);
        CHECK(!v.has_cold);                      // 冷槽已归还
    }
    CHECK_EQ(m.cold_used_bytes(), 0);
    CHECK_EQ(m.hot_valid_bytes(), BLK);

    CHECK_EQ(verify(m, 1, 0xABCDEF01u), 0);      // 数据穿过去了
}

// ============================================================
// 2. verify 必须会失败（负对照）
// ============================================================
static void test_verify_negative_control() {
    section("verify 的负对照：错的种子必须报不一致");
    GpuSwapManager m(HOT, COLD);
    CHECK(m.allocate(1, BLK) != nullptr);
    fill(m, 1, 0x1111u);
    CHECK_EQ(verify(m, 1, 0x1111u), 0);
    // 如果这一条也是 0，那 verify 就是个永远说"好"的函数，
    // 上面所有"数据一致"的验收都白做。
    int64_t bad = verify(m, 1, 0x2222u);
    CHECK(bad > 0);
    // 不能断言"全错"：两份花样都是伪随机字节，按概率本来就有约 1/256 的
    // 位置撞上（实测 4088/4096）。真正要钉的是"几乎全错"，那才说明两个
    // 种子给出的是两份不同的花样，而不是只差几个字节。
    CHECK(bad > static_cast<int64_t>(BLK) / 2);
}

// ============================================================
// 3. 槽位复用 + 毒化后再换回
// ============================================================
static void test_slot_reuse_and_poison() {
    section("槽位复用与毒化");
    GpuSwapManager m(HOT, COLD);
    const size_t SLOT = m.slot_bytes();

    BlockDesc* d1 = m.allocate(1, BLK);
    CHECK(d1 != nullptr);
    uint64_t addr1 = reinterpret_cast<uint64_t>(d1->gpu_ptr);
    fill(m, 1, 0xAAAAu);

    CHECK(m.evict_to_cold(1) == MigrateResult::OK);
    CHECK(m.sync() == MigrateResult::OK);

    // 断言地址真的被复用 —— 这是"毒化"这个测试成立的前提。
    BlockDesc* d2 = m.allocate(2, BLK);
    CHECK(d2 != nullptr);
    CHECK_EQ(reinterpret_cast<uint64_t>(d2->gpu_ptr), addr1);

    // 往同一个槽位写一份**不同**的花样。如果换回是空操作（比如懒得真拷），
    // 校验就会读出这一份。
    fill(m, 2, 0xBBBBu);
    CHECK(m.free(2) == MigrateResult::OK);
    CHECK_EQ(m.cold_used_bytes(), SLOT);         // block1 还在冷池

    CHECK(m.prefetch_to_hot(1) == MigrateResult::OK);
    CHECK(m.sync() == MigrateResult::OK);
    {
        BlockView v;
        CHECK(m.inspect(1, &v));
        CHECK(v.desc.in_gpu);
        // 换回来的就是被毒化的那一个槽位 —— 不然这个测试证明不了什么
        CHECK_EQ(reinterpret_cast<uint64_t>(v.desc.gpu_ptr), addr1);
    }
    CHECK_EQ(verify(m, 1, 0xAAAAu), 0);          // 原花样回来了
    CHECK(verify(m, 1, 0xBBBBu) > 0);            // 毒化花样被覆盖掉了
}

// ============================================================
// 4. 边界与错误路径
// ============================================================
static void test_errors() {
    section("边界与错误路径");
    GpuSwapManager m(HOT, COLD);

    CHECK(m.allocate(0, 0) == nullptr);                          // size == 0
    CHECK(m.allocate(0, m.slot_bytes() + 1) == nullptr);         // 超过槽位
    CHECK(m.allocate(1, BLK) != nullptr);
    CHECK(m.allocate(1, BLK) == nullptr);                        // 重复 id
    CHECK(std::strlen(m.last_error()) > 0);

    CHECK(m.free(999) == MigrateResult::ERR_BLOCK_NOT_FOUND);
    CHECK(m.evict_to_cold(999) == MigrateResult::ERR_BLOCK_NOT_FOUND);
    CHECK(m.prefetch_to_hot(999) == MigrateResult::ERR_BLOCK_NOT_FOUND);
    CHECK(m.touch(999) == MigrateResult::ERR_BLOCK_NOT_FOUND);

    // 已经在热池的块不能"换回"
    CHECK(m.prefetch_to_hot(1) == MigrateResult::ERR_INVALID_STATE);

    // 重复换出：第二次绝不能是 OK。可能是 IN_PROGRESS（还在飞）也可能是
    // INVALID_STATE（第一次已经落地了）—— 4 KB 的拷贝太快，两种都合法。
    CHECK(m.evict_to_cold(1) == MigrateResult::OK);
    MigrateResult r2 = m.evict_to_cold(1);
    CHECK(r2 == MigrateResult::ERR_IN_PROGRESS ||
          r2 == MigrateResult::ERR_INVALID_STATE);

    m.sync();
    CHECK(m.evict_to_cold(1) == MigrateResult::ERR_INVALID_STATE);   // 已不在显存
    CHECK(m.unpin(1) == MigrateResult::ERR_INVALID_STATE);           // 没 pin 过
}

// ============================================================
// 5. 冷池满
// ============================================================
static void test_cold_pool_full() {
    section("冷池满：ERR_CPU_OOM 且块留在 GPU");
    const size_t SLOT = 2u << 20;
    GpuSwapManager m(4 * SLOT, 1 * SLOT);       // 4 热槽，只有 1 个冷槽
    CHECK_EQ(m.cold_capacity(), SLOT);

    CHECK(m.allocate(1, BLK) != nullptr);
    CHECK(m.allocate(2, BLK) != nullptr);

    CHECK(m.evict_to_cold(1) == MigrateResult::OK);
    // 冷池只有一个槽，第二个换出必须被拒绝
    CHECK(m.evict_to_cold(2) == MigrateResult::ERR_CPU_OOM);

    {
        BlockView v;
        CHECK(m.inspect(2, &v));
        CHECK(v.desc.in_gpu);                   // 块留在 GPU，没被搬走也没被丢
        CHECK(v.valid);
        CHECK(!v.has_cold);
    }
    m.sync();
    CHECK_EQ(m.cold_used_bytes(), SLOT);
    CHECK(std::string(m.last_error()).find("冷池") != std::string::npos);
}

// ============================================================
// 6. pin 保护
// ============================================================
static void test_pin_protects() {
    section("pin 住的块不会被选中换出");
    const size_t SLOT = 2u << 20;
    GpuSwapManager m(3 * SLOT, 8 * SLOT);       // 3 个热槽

    CHECK(m.allocate(1, BLK) != nullptr);
    CHECK(m.allocate(2, BLK) != nullptr);
    CHECK(m.allocate(3, BLK) != nullptr);
    CHECK_EQ(m.num_blocks_in_gpu(), 3);
    CHECK_EQ(m.hot_used_bytes(), 3 * SLOT);

    // nullptr = 纯宿主侧保护。CUDA 侧的顺序约束（reader_done 事件）在
    // test_async.cu 里用一个真 kernel 测。
    CHECK(m.pin(3, nullptr) == MigrateResult::OK);
    CHECK(m.is_pinned(3));

    // pin 是**顺序约束，不是否决权** —— 这个区分很容易搞反，所以两个方向
    // 都要有断言。显式换出照做（D2H 会排在 reader 后面，这里没有真 reader，
    // 所以立刻落地）；自动挑 victim 则要绕开它。
    CHECK(m.evict_to_cold(3) == MigrateResult::OK);
    CHECK(m.sync() == MigrateResult::OK);
    {
        BlockView v;
        CHECK(m.inspect(3, &v));
        CHECK(!v.desc.in_gpu);                  // 真的换出去了
        CHECK(v.pinned);                        // 但 pin 还在
    }
    // 换出去就"不是可换出集合的一员"了，可释放性仍然是 pin 说了算：
    // 换出能靠 stream 顺序做安全，**释放不能**（释放后显存分给别人而
    // 读者还在读 = use-after-free）。
    CHECK(m.free(3) == MigrateResult::ERR_PINNED);

    // 换回来，接着验"自动挑 victim 绕开 pinned"。
    CHECK(m.prefetch_to_hot(3) == MigrateResult::OK);
    CHECK(m.sync() == MigrateResult::OK);
    CHECK(m.unpin(3) == MigrateResult::OK);     // 先解掉，下面重新钉
    CHECK(m.pin(3, nullptr) == MigrateResult::OK);

    // 第 4 个块要挤进来 → 必须换出点什么，但 3 被钉住了
    const int64_t evictions_before = m.eviction_count();
    CHECK(m.allocate(4, BLK) != nullptr);
    {
        BlockView v;
        CHECK(m.inspect(3, &v));
        CHECK(v.desc.in_gpu);                   // 3 还在
    }
    CHECK(m.eviction_count() > evictions_before);   // 换了别人

    CHECK(m.unpin(3) == MigrateResult::OK);
    CHECK(!m.is_pinned(3));
    CHECK(m.free(3) == MigrateResult::OK);          // 解除后就能释放了
}

// ============================================================
// 7. 在途冻结：free 必须拒绝（或已经落地）
// ============================================================
static void test_free_while_migrating() {
    section("在途时 free");
    GpuSwapManager m(HOT, COLD);
    CHECK(m.allocate(1, BLK) != nullptr);
    CHECK(m.evict_to_cold(1) == MigrateResult::OK);

    // 4 KB 的 D2H 只要几微秒，所以"拒绝"和"已经落地所以放行"都合法。
    // 真正要钉死的不是返回值，而是**不变量**：不管哪条路，释放之后账目自洽。
    MigrateResult r = m.free(1);
    CHECK(r == MigrateResult::ERR_IN_PROGRESS || r == MigrateResult::OK);
    m.sync();
    CHECK_EQ(m.pending_migrations(), 0);
    if (r == MigrateResult::OK) {
        CHECK(!m.is_migrating(1));
        CHECK_EQ(m.num_blocks(), 0);
        CHECK_EQ(m.hot_used_bytes(), 0);
        CHECK_EQ(m.cold_used_bytes(), 0);
    } else {
        CHECK_EQ(m.num_blocks(), 1);
    }
}

// ============================================================
// 8. 确定性：同一序列跑两遍，逐出指纹必须相同
// ============================================================
//
// 这是本项目里**最便宜、最有说服力**的"策略真的做了决策"的证据。
// 聚合计数做不到 —— 项目一实测过：同一个策略的三次重复，计数几乎逐位相同，
// 而逐出指纹三次互不相同（逐出序列逐次不可复现）。所以"指纹相同"是有信息
// 的（说明确定性成立），"指纹不同"才是没信息的那一侧。
static uint32_t run_sequence(IEvictionPolicy* pol) {
    const size_t SLOT = 2u << 20;
    GpuSwapManager m(4 * SLOT, 16 * SLOT);
    m.set_policy(pol);
    // 固定序列 + 固定时钟，任何不确定性都只可能来自实现本身。
    for (int i = 0; i < 12; ++i) {
        m.set_clock(1000 + i * 10);
        m.allocate(i, BLK);
        for (int k = 0; k <= i % 4; ++k) m.touch(i);
    }
    m.settle();
    return m.eviction_fingerprint();
}

static void test_determinism() {
    section("逐出序列的确定性（指纹可复现）");
    {
        AdaptivePolicy p1, p2;
        CHECK_EQ(run_sequence(&p1), run_sequence(&p2));
    }
    {
        LRUPolicy p1, p2;
        CHECK_EQ(run_sequence(&p1), run_sequence(&p2));
    }
    {
        LFUPolicy p1, p2;
        CHECK_EQ(run_sequence(&p1), run_sequence(&p2));
    }
    // 不同策略**应该**给出不同的序列（否则说明策略根本没起作用）。
    // 这一条较弱，但方向上是对的：四个策略在同一个序列上不该全部撞车。
    AdaptivePolicy a;
    LRUPolicy l;
    LFUPolicy f;
    uint32_t fa = run_sequence(&a), fl = run_sequence(&l), ff = run_sequence(&f);
    CHECK(fa != fl || fl != ff);
}

// ============================================================
// 9. AdaptivePolicy 的保护率上报
// ============================================================
static void test_adaptive_report() {
    section("AdaptivePolicy：保护率上报与夹断可见");
    const size_t SLOT = 2u << 20;
    GpuSwapManager m(8 * SLOT, 64 * SLOT);

    AdaptivePolicy::Config cfg;
    cfg.young_age_ms = 500;
    AdaptivePolicy pol(cfg);
    m.set_policy(&pol);

    // 造一批"读次数极高"的块：绝对阈值 50 会保护几乎全部，
    // 于是夹断必须触发（protect_max_rate = 0.30）。
    for (int i = 0; i < 8; ++i) {
        m.set_clock(10000 + i);          // 时钟远远走过 young_age
        m.allocate(i, BLK);
        for (int k = 0; k < 100; ++k) m.touch(i);
    }
    // 塞第 9 个块，逼出一次换出
    m.set_clock(20000);
    m.allocate(100, BLK);

    AdaptivePolicy::Report r = pol.report();
    std::printf("    决策=%lld 夹断=%lld 放开热块=%lld 放开年龄=%lld "
                "保护率 min/mean/max=%.3f/%.3f/%.3f 首/末四分位=%.3f/%.3f\n",
                (long long)r.decisions, (long long)r.clamped,
                (long long)r.relaxed_hot, (long long)r.relaxed_young,
                r.protect_min, r.protect_mean, r.protect_max,
                r.protect_first_quartile, r.protect_last_quartile);
    CHECK(r.decisions > 0);
    CHECK(r.clamped > 0);                 // 占位阈值 50 在这里必然超线
    CHECK(r.protect_max <= 0.31);         // 夹断生效：保护率没被放开
    CHECK(r.victims > 0);
}

// ============================================================
// 9b. 松手计数器的负对照
// ============================================================
//
// 为什么这条必须有：两个计数器是**同一种事件的两个阶段**，只看"跑出来的
// 数"分不清谁是谁 —— 计数器接反了的时候，两条都可能是非零，而报告照样
// 打得出来、保护率照样算得对。所以必须构造一个**只有年龄规则**可能生效
// 的场景：热块阈值取 INT64_MAX（任何块都不可能被判为热），断言
// "放开热块"必须恰好为 0。计数器接反的版本上这条会失败。
//
// 顺带一个操作上的教训：`放开热块=0 放开年龄=0` 这种输出**不能**用来
// 验收这条修复 —— 两条分支都没走到，改没改都一样。
static void test_relax_counters() {
    const size_t SLOT = 2u << 20;
    {
        section("松手计数器负对照：热块规则没生效，就不许记成'放开热块'");
        GpuSwapManager m(4 * SLOT, 64 * SLOT);

        AdaptivePolicy::Config cfg;
        cfg.hot_access_threshold = INT64_MAX;  // 热块规则**永远**不生效
        cfg.young_age_ms = 1000000;            // 所有块永远算"年轻"
        AdaptivePolicy pol(cfg);
        m.set_policy(&pol);

        m.set_clock(100);
        for (int i = 0; i < 4; ++i) {          // 4 个槽刚好占满
            CHECK(m.allocate(i, BLK) != nullptr);
            m.touch(i);                        // 读 1 次，离阈值无穷远
        }
        CHECK(m.allocate(100, BLK) != nullptr);  // 塞第 5 个，逼出一次换出

        AdaptivePolicy::Report r = pol.report();
        std::printf("    决策=%lld 夹断=%lld 放开热块=%lld 放开年龄=%lld\n",
                    (long long)r.decisions, (long long)r.clamped,
                    (long long)r.relaxed_hot, (long long)r.relaxed_young);
        CHECK_EQ(r.decisions, 1);
        CHECK(r.victims > 0);
        CHECK_EQ(r.relaxed_hot, 0);    // ← 计数器接反的话这里是 1
        CHECK_EQ(r.relaxed_young, 1);  // 年龄规则挡住了全部候选
    }
    {
        // 正向级联：保护率上限放到 1.0（关掉夹断）、阈值为 -1（所有块都热），
        // 于是两条规则必须**按顺序**都放开。这一条不能区分修复前后
        // （接反的版本同样两条都 +1），它锁的是级联形状：
        // "放开热块"只能发生在"放开年龄"之后，不可能单独出现。
        section("松手计数器正向级联：两道都放开时两条都记一次");
        GpuSwapManager m(4 * SLOT, 64 * SLOT);

        AdaptivePolicy::Config cfg;
        cfg.protect_max_rate = 1.0;            // 关掉夹断，否则 70 分位会把候选还回来
        cfg.hot_access_threshold = -1;         // 所有块都是"热"的
        AdaptivePolicy pol(cfg);
        m.set_policy(&pol);

        m.set_clock(100);
        for (int i = 0; i < 4; ++i) {
            CHECK(m.allocate(i, BLK) != nullptr);
            m.touch(i);
        }
        CHECK(m.allocate(100, BLK) != nullptr);

        AdaptivePolicy::Report r = pol.report();
        std::printf("    决策=%lld 夹断=%lld 放开热块=%lld 放开年龄=%lld\n",
                    (long long)r.decisions, (long long)r.clamped,
                    (long long)r.relaxed_hot, (long long)r.relaxed_young);
        CHECK_EQ(r.decisions, 1);
        CHECK_EQ(r.relaxed_young, 1);
        CHECK_EQ(r.relaxed_hot, 1);
    }
}

// ============================================================
// 10. 年龄规则真的挡住了新块
// ============================================================
static void test_young_rule() {
    section("年龄规则：刚分配的块优先不被换出");
    const size_t SLOT = 2u << 20;
    GpuSwapManager m(2 * SLOT, 8 * SLOT);

    AdaptivePolicy::Config cfg;
    cfg.young_age_ms = 1000;              // 1 秒
    AdaptivePolicy pol(cfg);
    m.set_policy(&pol);

    m.set_clock(0);
    CHECK(m.allocate(1, BLK) != nullptr);   // 老块
    m.set_clock(5000);                      // 时间走过 5 秒
    CHECK(m.allocate(2, BLK) != nullptr);   // 新块，age = 0

    m.touch(1);                             // 老块被读过一次，新块没有
    m.set_clock(5100);
    CHECK(m.allocate(3, BLK) != nullptr);   // 逼出一次换出

    // 老块有访问、新块年龄 < 1000ms，两条规则都指向"换出老块"。
    BlockView v1, v2;
    CHECK(m.inspect(1, &v1));
    CHECK(m.inspect(2, &v2));
    CHECK(v2.desc.in_gpu);                  // 新块被保护住了
    std::printf("    block1 in_gpu=%d  block2 in_gpu=%d\n",
                v1.desc.in_gpu ? 1 : 0, v2.desc.in_gpu ? 1 : 0);
}

// ============================================================
// 11. 换出/换回一大批，逐个校验
// ============================================================
static void test_bulk_roundtrip() {
    section("批量：12 个块全部换出再全部换回，逐个校验");
    const size_t SLOT = 2u << 20;
    GpuSwapManager m(6 * SLOT, 32 * SLOT);
    const int N = 12;

    for (int i = 0; i < N; ++i) {
        if (!m.allocate(i, BLK)) {
            std::printf("    (第 %d 个块分配失败: %s)\n", i, m.last_error());
            break;
        }
        fill(m, i, 0x5000 + i);
    }
    for (int i = 0; i < N; ++i) {
        if (!m.is_migrating(i)) m.evict_to_cold(i);
    }
    // 注意这里**不能**断言 pending == 0：evict 是立即返回的，最后几次
    // 还在飞。而访问器不 poll，所以没人会顺手收割它们 —— 这正是
    // "立即返回"这个契约的可观测形式。要等就显式 sync。
    CHECK(m.pending_migrations() > 0);
    m.sync();
    CHECK_EQ(m.pending_migrations(), 0);
    CHECK_EQ(m.num_blocks_in_gpu(), 0);

    // 逐个"换回 → 落地 → 校验 → 再换回下一个"。
    //
    // 不能一次性发起 12 个 prefetch 再统一校验：热池只有 6 个槽位，
    // 第 7 个 prefetch 必然把前 6 个里的某一个换出去，于是"12 个都在
    // 显存里"这个断言**在物理上就不可能成立**。校验必须在块真的驻留
    // 在显存里的那一刻做。
    int good = 0;
    for (int i = 0; i < N; ++i) {
        MigrateResult r = m.prefetch_to_hot(i);
        if (r != MigrateResult::OK) {
            std::printf("    第 %d 个块换回失败: %s (%s)\n", i,
                        to_string(r), m.last_error());
            continue;
        }
        m.sync();
        if (verify(m, i, 0x5000 + i) == 0) ++good;
    }
    CHECK_EQ(good, N);

    // 换完一轮之后，显存里的驻留数必须**等于槽位数**（不多不少）。
    CHECK_EQ(m.num_blocks_in_gpu(), 6);
    CHECK_EQ(m.num_blocks(), N);             // 一个都没丢
    CHECK_EQ(m.hot_used_bytes(), 6 * SLOT);
    // 冷池里躺着剩下的 6 个
    CHECK_EQ(m.cold_used_bytes(), 6 * SLOT);
}


int main() {
    std::printf("=== gpu-swap 正确性测试 ===\n");
    int dev = 0;
    cudaError_t st = cudaGetDevice(&dev);
    if (st != cudaSuccess) {
        std::printf("没有可用的 CUDA 设备: %s\n", cudaGetErrorString(st));
        return 77;
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    std::printf("设备: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    test_evict_prefetch_roundtrip();
    test_verify_negative_control();
    test_slot_reuse_and_poison();
    test_errors();
    test_cold_pool_full();
    test_pin_protects();
    test_free_while_migrating();
    test_determinism();
    test_adaptive_report();
    test_relax_counters();
    test_young_rule();
    test_bulk_roundtrip();

    std::printf("\n=== 通过 %d，失败 %d ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
