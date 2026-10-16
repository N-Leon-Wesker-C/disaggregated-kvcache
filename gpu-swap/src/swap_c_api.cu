/*
 * swap_c_api.cu — C 接口的实现
 *
 * 这个文件存在的唯一理由：让 Python（ctypes）能驱动真的 CUDA 库。
 *
 * 所有边界的写法都是同一个模式：
 *
 *     GSM_ENTRY(h)                  // 判空句柄，给出 GSM_ERR_BAD_HANDLE
 *     try { ... } catch (...) {     // 异常绝不外泄
 *         return fail(h, "…");
 *     }
 *
 * 一旦漏掉一个 try/catch，Python 侧看到的不是 traceback 而是进程 abort ——
 * 而且是在 ctypes 那一层崩的，看不出是 C++ 抛的。
 */

#include "gpu_swap_c.h"
#include "gpu_swap.h"
#include "eviction_policy.h"

#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

struct gsm_handle {
    GpuSwapManager* mgr = nullptr;
    // 策略**按句柄持有**，不用函数内 static。GDSF 有状态（通胀因子 L 和
    // 每个块的基价），做成 static 的话两个句柄会互相污染 —— 这种 bug 在
    // 单句柄的测试里 100% 看不出来。
    LRUPolicy lru;
    LFUPolicy lfu;
    GDSFPolicy gdsf;
    AdaptivePolicy* adaptive = nullptr;   // 非空 ⇔ 当前策略是 adaptive
    std::string err;
};

namespace {

void set_err(gsm_handle* h, const std::string& msg) {
    if (h) h->err = msg;
}

int code_of(MigrateResult r) { return static_cast<int>(r); }

/// 把"管理器内部的粘性错误"也带出来。异步失败只在 reap 时才知道，
/// 所以很多操作返回 OK 但 last_error 里已经有东西了 —— 别让它烂在里面。
int finish(gsm_handle* h, MigrateResult r) {
    if (r == MigrateResult::OK) {
        const char* e = h->mgr->last_error();
        if (e && *e) set_err(h, std::string("(上一次异步迁移) ") + e);
    } else {
        set_err(h, std::string(to_string(r)) + ": " + h->mgr->last_error());
    }
    return code_of(r);
}

uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/// 花样生成。fill 和 verify **共用这一个函数** —— 分成两份的话，
/// 两边一旦不同步，verify 就会永远失败或者永远成功，而后者更危险。
void make_pattern(unsigned char* buf, size_t n, uint64_t seed) {
    uint64_t state = seed ^ 0xA5A5A5A5A5A5A5A5ull;
    size_t i = 0;
    while (i < n) {
        uint64_t v = splitmix64(state);
        for (int b = 0; b < 8 && i < n; ++b) {
            buf[i++] = static_cast<unsigned char>((v >> (b * 8)) & 0xff);
        }
    }
}

int ensure_live(gsm_handle* h) {
    if (!h) return GSM_ERR_BAD_HANDLE;
    if (!h->mgr) {
        set_err(h, "句柄没有管理器（gsm_create 里 CUDA 初始化失败过）");
        return GSM_ERR_BAD_HANDLE;
    }
    return GSM_OK;
}

}  // namespace

#define GSM_ENTRY(h)                       \
    do {                                   \
        int _rc = ensure_live(h);          \
        if (_rc != GSM_OK) return _rc;     \
    } while (0)

#define GSM_TRY                                    \
    try {

#define GSM_CATCH(h)                                              \
    } catch (const std::exception& e) {                           \
        set_err(h, std::string("C++ 异常: ") + e.what());         \
        return GSM_ERR_EXCEPTION;                                 \
    } catch (...) {                                               \
        set_err(h, "C++ 未知异常");                                \
        return GSM_ERR_EXCEPTION;                                 \
    }


extern "C" {

gsm_handle* gsm_create(size_t hot_pool_bytes, size_t cold_pool_bytes) {
    gsm_handle* h = new (std::nothrow) gsm_handle();
    if (!h) return nullptr;
    try {
        h->mgr = new GpuSwapManager(hot_pool_bytes, cold_pool_bytes);
        if (!h->mgr) {
            set_err(h, "GpuSwapManager 构造失败");
        } else if (*h->mgr->last_error()) {
            // cudaMalloc 失败不会抛异常，只会让 arena 为空。必须在这里就说清楚，
            // 否则后面 allocate 只会返回一个含糊的 OOM。
            set_err(h, h->mgr->last_error());
        }
    } catch (const std::exception& e) {
        set_err(h, std::string("构造异常: ") + e.what());
    } catch (...) {
        set_err(h, "构造时未知异常");
    }
    return h;
}

void gsm_destroy(gsm_handle* h) {
    if (!h) return;
    try {
        delete h->adaptive;      // 先删策略：管理器只持非拥有指针
        h->adaptive = nullptr;
        if (h->mgr) h->mgr->set_policy(nullptr);
        delete h->mgr;
    } catch (...) {
        // 析构里出事已经没什么可做的了，吞掉。
    }
    delete h;
}

int gsm_allocate(gsm_handle* h, int64_t block_id, size_t size) {
    GSM_ENTRY(h);
    GSM_TRY
        BlockDesc* d = h->mgr->allocate(block_id, size);
        if (!d) {
            set_err(h, std::string("allocate 失败: ") + h->mgr->last_error());
            return GSM_ERR_INVALID_STATE;
        }
        return GSM_OK;
    GSM_CATCH(h)
}

int gsm_free(gsm_handle* h, int64_t block_id) {
    GSM_ENTRY(h);
    GSM_TRY
        return finish(h, h->mgr->free(block_id));
    GSM_CATCH(h)
}

int gsm_touch(gsm_handle* h, int64_t block_id) {
    GSM_ENTRY(h);
    GSM_TRY
        return finish(h, h->mgr->touch(block_id));
    GSM_CATCH(h)
}

int gsm_evict(gsm_handle* h, int64_t block_id) {
    GSM_ENTRY(h);
    GSM_TRY
        return finish(h, h->mgr->evict_to_cold(block_id));
    GSM_CATCH(h)
}

int gsm_prefetch(gsm_handle* h, int64_t block_id) {
    GSM_ENTRY(h);
    GSM_TRY
        return finish(h, h->mgr->prefetch_to_hot(block_id));
    GSM_CATCH(h)
}

int gsm_sync(gsm_handle* h) {
    GSM_ENTRY(h);
    GSM_TRY
        return finish(h, h->mgr->sync());
    GSM_CATCH(h)
}

int gsm_wait_block(gsm_handle* h, int64_t block_id) {
    GSM_ENTRY(h);
    GSM_TRY
        return finish(h, h->mgr->wait_block(block_id));
    GSM_CATCH(h)
}

int gsm_settle(gsm_handle* h) {
    GSM_ENTRY(h);
    GSM_TRY
        return finish(h, h->mgr->settle());
    GSM_CATCH(h)
}

int gsm_reserve_cold(gsm_handle* h, size_t n_slots) {
    GSM_ENTRY(h);
    GSM_TRY
        return finish(h, h->mgr->reserve_cold(n_slots));
    GSM_CATCH(h)
}

int gsm_pin(gsm_handle* h, int64_t block_id, void* reader_done_event) {
    GSM_ENTRY(h);
    GSM_TRY
        return finish(h, h->mgr->pin(block_id,
                                     reinterpret_cast<cudaEvent_t>(reader_done_event)));
    GSM_CATCH(h)
}

int gsm_unpin(gsm_handle* h, int64_t block_id) {
    GSM_ENTRY(h);
    GSM_TRY
        return finish(h, h->mgr->unpin(block_id));
    GSM_CATCH(h)
}

void gsm_set_clock(gsm_handle* h, int64_t now_ms) {
    if (!h || !h->mgr) return;
    try { h->mgr->set_clock(now_ms); } catch (...) {}
}

int64_t gsm_clock(gsm_handle* h) {
    if (!h || !h->mgr) return -1;
    try { return h->mgr->clock(); } catch (...) { return -1; }
}

int gsm_set_policy(gsm_handle* h, const char* name,
                   int64_t young_age_ms, int64_t hot_access_threshold,
                   double protect_max_rate) {
    GSM_ENTRY(h);
    if (!name) {
        set_err(h, "策略名是空指针");
        return GSM_ERR_BAD_ARG;
    }
    GSM_TRY
        std::string n(name);
        AdaptivePolicy* old = h->adaptive;
        h->adaptive = nullptr;
        if (n == "adaptive") {
            AdaptivePolicy::Config cfg;
            if (young_age_ms >= 0) cfg.young_age_ms = young_age_ms;
            if (hot_access_threshold >= 0) cfg.hot_access_threshold = hot_access_threshold;
            if (protect_max_rate >= 0.0) cfg.protect_max_rate = protect_max_rate;
            h->adaptive = new AdaptivePolicy(cfg);
            h->mgr->set_policy(h->adaptive);
        } else if (n == "gdsf") {
            h->mgr->set_policy(&h->gdsf);
        } else if (n == "lru") {
            h->mgr->set_policy(&h->lru);
        } else if (n == "lfu") {
            h->mgr->set_policy(&h->lfu);
        } else {
            set_err(h, std::string("未知策略: ") + n);
            return GSM_ERR_BAD_ARG;
        }
        delete old;                          // 延后删：set_policy 换掉之后才安全
        h->err.clear();
        return GSM_OK;
    GSM_CATCH(h)
}

int gsm_policy_report(gsm_handle* h, gsm_policy_report_t* out) {
    GSM_ENTRY(h);
    if (!out) {
        set_err(h, "out 是空指针");
        return GSM_ERR_BAD_ARG;
    }
    GSM_TRY
        std::memset(out, 0, sizeof(*out));
        if (!h->adaptive) {
            out->is_adaptive = 0;
            set_err(h, "当前策略不是 adaptive，没有保护率统计");
            return GSM_ERR_BAD_ARG;
        }
        AdaptivePolicy::Report r = h->adaptive->report();
        out->is_adaptive = 1;
        out->decisions = r.decisions;
        out->victims = r.victims;
        out->clamped = r.clamped;
        out->relaxed_hot = r.relaxed_hot;
        out->relaxed_young = r.relaxed_young;
        out->young_block_hits = r.young_block_hits;
        out->hot_block_hits = r.hot_block_hits;
        out->resident_seen = r.resident_seen;
        out->protect_min = r.protect_min;
        out->protect_mean = r.protect_mean;
        out->protect_max = r.protect_max;
        out->protect_first_quartile = r.protect_first_quartile;
        out->protect_last_quartile = r.protect_last_quartile;
        out->cutoff_last = r.cutoff_last;
        out->cutoff_clamped_last = r.cutoff_clamped_last;
        out->inflation = r.inflation;
        return GSM_OK;
    GSM_CATCH(h)
}

size_t gsm_hot_used(gsm_handle* h) { return h && h->mgr ? h->mgr->hot_used_bytes() : 0; }
size_t gsm_hot_valid(gsm_handle* h) { return h && h->mgr ? h->mgr->hot_valid_bytes() : 0; }
size_t gsm_cold_used(gsm_handle* h) { return h && h->mgr ? h->mgr->cold_used_bytes() : 0; }
size_t gsm_hot_capacity(gsm_handle* h) { return h && h->mgr ? h->mgr->hot_capacity() : 0; }
size_t gsm_cold_capacity(gsm_handle* h) { return h && h->mgr ? h->mgr->cold_capacity() : 0; }
size_t gsm_hot_peak(gsm_handle* h) { return h && h->mgr ? h->mgr->hot_peak_bytes() : 0; }
int    gsm_num_blocks(gsm_handle* h) { return h && h->mgr ? h->mgr->num_blocks() : 0; }
int    gsm_num_blocks_in_gpu(gsm_handle* h) {
    return h && h->mgr ? h->mgr->num_blocks_in_gpu() : 0;
}
int    gsm_pending_migrations(gsm_handle* h) {
    return h && h->mgr ? h->mgr->pending_migrations() : 0;
}
int    gsm_is_migrating(gsm_handle* h, int64_t id) {
    return h && h->mgr && h->mgr->is_migrating(id) ? 1 : 0;
}
int    gsm_is_pinned(gsm_handle* h, int64_t id) {
    return h && h->mgr && h->mgr->is_pinned(id) ? 1 : 0;
}
uint32_t gsm_eviction_fingerprint(gsm_handle* h) {
    return h && h->mgr ? h->mgr->eviction_fingerprint() : 0;
}
int64_t gsm_eviction_count(gsm_handle* h) {
    return h && h->mgr ? h->mgr->eviction_count() : 0;
}

size_t gsm_slot_bytes(gsm_handle* h) {
    return h && h->mgr ? h->mgr->slot_bytes() : 0;
}

const char* gsm_last_error(gsm_handle* h) {
    if (!h) return "空句柄";
    if (h->err.empty() && h->mgr) {
        const char* e = h->mgr->last_error();
        return (e && *e) ? e : "";
    }
    return h->err.c_str();
}

const char* gsm_result_str(int code) {
    switch (code) {
        case GSM_OK: return "OK";
        case GSM_ERR_GPU_OOM: return "ERR_GPU_OOM";
        case GSM_ERR_CPU_OOM: return "ERR_CPU_OOM";
        case GSM_ERR_BLOCK_NOT_FOUND: return "ERR_BLOCK_NOT_FOUND";
        case GSM_ERR_IN_PROGRESS: return "ERR_IN_PROGRESS";
        case GSM_ERR_INVALID_STATE: return "ERR_INVALID_STATE";
        case GSM_ERR_PINNED: return "ERR_PINNED";
        case GSM_ERR_CUDA: return "ERR_CUDA";
        case GSM_ERR_BAD_HANDLE: return "ERR_BAD_HANDLE";
        case GSM_ERR_BAD_ARG: return "ERR_BAD_ARG";
        case GSM_ERR_EXCEPTION: return "ERR_EXCEPTION";
    }
    return "?";
}

int gsm_block_info(gsm_handle* h, int64_t block_id, gsm_block_info_t* out) {
    GSM_ENTRY(h);
    if (!out) {
        set_err(h, "out 是空指针");
        return GSM_ERR_BAD_ARG;
    }
    GSM_TRY
        BlockView v;
        if (!h->mgr->inspect(block_id, &v)) {
            set_err(h, "block 不存在");
            return GSM_ERR_BLOCK_NOT_FOUND;
        }
        std::memset(out, 0, sizeof(*out));
        out->in_gpu = v.desc.in_gpu ? 1 : 0;
        out->valid = v.valid ? 1 : 0;
        out->migrating = v.migrating ? 1 : 0;
        out->pinned = v.pinned ? 1 : 0;
        out->has_cold = v.has_cold ? 1 : 0;
        out->size = v.desc.size;
        out->gpu_addr = reinterpret_cast<uint64_t>(v.desc.gpu_ptr);
        out->last_access = v.desc.last_access_time;
        out->access_count = v.desc.access_count;
        out->alloc_time = v.desc.alloc_time;
        return GSM_OK;
    GSM_CATCH(h)
}

uint64_t gsm_gpu_addr(gsm_handle* h, int64_t block_id) {
    if (!h || !h->mgr) return 0;
    BlockView v;
    if (!h->mgr->inspect(block_id, &v)) return 0;
    return reinterpret_cast<uint64_t>(v.desc.gpu_ptr);
}

int gsm_fill_pattern(gsm_handle* h, int64_t block_id, uint64_t seed) {
    GSM_ENTRY(h);
    GSM_TRY
        BlockView v;
        if (!h->mgr->inspect(block_id, &v)) {
            set_err(h, "block 不存在");
            return GSM_ERR_BLOCK_NOT_FOUND;
        }
        if (v.migrating) {
            set_err(h, "block 正在迁移，不能写花样");
            return GSM_ERR_IN_PROGRESS;
        }
        if (!v.desc.gpu_ptr || !v.desc.in_gpu) {
            set_err(h, "block 不在显存，不能写花样");
            return GSM_ERR_INVALID_STATE;
        }
        std::vector<unsigned char> buf(v.desc.size);
        make_pattern(buf.data(), buf.size(), seed);
        // 走**管理器自己的 swap_stream**，不落默认流：默认流会和所有其它流
        // 隐式串行，而且更糟的是它和 swap_stream 上的在途迁移之间没有我们
        // 想要的顺序 —— 排进同一条队才有"先到先做"的保证。
        cudaStream_t s = reinterpret_cast<cudaStream_t>(h->mgr->raw_swap_stream());
        cudaError_t st = cudaMemcpyAsync(v.desc.gpu_ptr, buf.data(), buf.size(),
                                         cudaMemcpyHostToDevice, s);
        if (st != cudaSuccess) {
            set_err(h, std::string("fill 拷贝失败: ") + cudaGetErrorString(st));
            return GSM_ERR_CUDA;
        }
        st = cudaStreamSynchronize(s);
        if (st != cudaSuccess) {
            set_err(h, std::string("fill 同步失败: ") + cudaGetErrorString(st));
            return GSM_ERR_CUDA;
        }
        return GSM_OK;
    GSM_CATCH(h)
}

int gsm_verify_pattern(gsm_handle* h, int64_t block_id, uint64_t seed,
                       int64_t* out_mismatch_bytes) {
    GSM_ENTRY(h);
    if (!out_mismatch_bytes) {
        set_err(h, "out_mismatch_bytes 是空指针");
        return GSM_ERR_BAD_ARG;
    }
    GSM_TRY
        *out_mismatch_bytes = -1;
        BlockView v;
        if (!h->mgr->inspect(block_id, &v)) {
            set_err(h, "block 不存在");
            return GSM_ERR_BLOCK_NOT_FOUND;
        }
        if (v.migrating) {
            set_err(h, "block 正在迁移，不能校验");
            return GSM_ERR_IN_PROGRESS;
        }
        if (!v.desc.gpu_ptr || !v.desc.in_gpu) {
            set_err(h, "block 不在显存，不能校验");
            return GSM_ERR_INVALID_STATE;
        }
        std::vector<unsigned char> got(v.desc.size), want(v.desc.size);
        make_pattern(want.data(), want.size(), seed);
        cudaStream_t s = reinterpret_cast<cudaStream_t>(h->mgr->raw_swap_stream());
        cudaError_t st = cudaMemcpyAsync(got.data(), v.desc.gpu_ptr, got.size(),
                                         cudaMemcpyDeviceToHost, s);
        if (st != cudaSuccess) {
            set_err(h, std::string("verify 拷贝失败: ") + cudaGetErrorString(st));
            return GSM_ERR_CUDA;
        }
        st = cudaStreamSynchronize(s);
        if (st != cudaSuccess) {
            set_err(h, std::string("verify 同步失败: ") + cudaGetErrorString(st));
            return GSM_ERR_CUDA;
        }
        int64_t bad = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            if (got[i] != want[i]) ++bad;
        }
        *out_mismatch_bytes = bad;
        return GSM_OK;
    GSM_CATCH(h)
}

}  // extern "C"
