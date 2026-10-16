/*
 * gpu_swap_c.h — GpuSwapManager 的 C 接口（给 ctypes 用）
 *
 * 两条硬规矩：
 *
 * 1. **C++ 异常绝不允许穿过这个边界。** ctypes 对面是 Python 解释器，
 *    异常穿过去不是 "traceback"，是进程直接 abort，而且看不出是哪一层的问题。
 *    每个入口都 try/catch，错误落进句柄里的 last_error，由
 *    gsm_last_error() 取。
 *
 * 2. **句柄是不透明指针**，调用者不解释它的内容。所有返回 `const char*`
 *    的函数，其指针在下一次对同一句柄的调用之前有效 —— 要留就自己拷走。
 *
 * 线程亲和：CUDA 上下文是 per-thread 的。一个句柄只能在创建它的那个线程里用。
 */

#ifndef GPU_SWAP_C_H
#define GPU_SWAP_C_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gsm_handle gsm_handle;

/* 结果码与 C++ 的 MigrateResult 一一对应，别改数值。 */
enum {
    GSM_OK = 0,
    GSM_ERR_GPU_OOM = 1,
    GSM_ERR_CPU_OOM = 2,
    GSM_ERR_BLOCK_NOT_FOUND = 3,
    GSM_ERR_IN_PROGRESS = 4,
    GSM_ERR_INVALID_STATE = 5,
    GSM_ERR_PINNED = 6,
    GSM_ERR_CUDA = 7,
    /* 以下是 C 层自己加的，C++ 侧没有对应 */
    GSM_ERR_BAD_HANDLE = 8,
    GSM_ERR_BAD_ARG = 9,
    GSM_ERR_EXCEPTION = 10,
};

/* 生涯 */
gsm_handle* gsm_create(size_t hot_pool_bytes, size_t cold_pool_bytes);
void        gsm_destroy(gsm_handle* h);

/* 分配 / 释放 / 访问 */
int  gsm_allocate(gsm_handle* h, int64_t block_id, size_t size);
int  gsm_free(gsm_handle* h, int64_t block_id);
int  gsm_touch(gsm_handle* h, int64_t block_id);

/* 迁移。evict / prefetch 都是**立即返回**的，要等用 gsm_sync / gsm_wait_block。 */
int  gsm_evict(gsm_handle* h, int64_t block_id);
int  gsm_prefetch(gsm_handle* h, int64_t block_id);
int  gsm_sync(gsm_handle* h);
int  gsm_wait_block(gsm_handle* h, int64_t block_id);
int  gsm_settle(gsm_handle* h);

/* 预热冷池：先把 n 个槽位的 pinned 内存分配好。
 * **必须**在任何 kernel 之前（或最后一段 kernel 之后）调用 ——
 * cudaMallocHost 是隐式同步点，第一次换出时惰性分配会把设备拖停，
 * 那样测出来的"异步"是假的（数据对，但不重叠）。 */
int  gsm_reserve_cold(gsm_handle* h, size_t n_slots);

/* pin：在 reader_done 事件完成前不许换出该块。
 * **事件的所有权转给管理器** —— pin 之后调用者不要自己 destroy，
 * 由 unpin / free / 析构负责。传 0 表示没有在途的 GPU 读，纯宿主保护。
 * reader_done 必须**已经 record 过**（cudaStreamWaitEvent 在调用时快照状态）。 */
int  gsm_pin(gsm_handle* h, int64_t block_id, void* reader_done_event);
int  gsm_unpin(gsm_handle* h, int64_t block_id);

/* 模拟时钟（毫秒）。只喂策略时间戳，管理器自己不用它做超时或统计。 */
void    gsm_set_clock(gsm_handle* h, int64_t now_ms);
int64_t gsm_clock(gsm_handle* h);

/* 策略。name: "lru" | "lfu" | "gdsf" | "adaptive"
 * 后三个参数 < 0 表示"用头文件里的默认值"。 */
int  gsm_set_policy(gsm_handle* h, const char* name,
                    int64_t young_age_ms, int64_t hot_access_threshold,
                    double protect_max_rate);

typedef struct {
    int64_t decisions, victims, clamped, relaxed_hot, relaxed_young;
    int64_t young_block_hits, hot_block_hits, resident_seen;
    double  protect_min, protect_mean, protect_max;
    double  protect_first_quartile, protect_last_quartile;
    int64_t cutoff_last, cutoff_clamped_last;
    double  inflation;          /* 排序内核当前的通胀幅度（有界） */
    int64_t inflation_rebases;  /* 通胀被迫重基准的次数，>0 = 退化过 */
    int32_t is_adaptive;   /* 0 = 当前策略不是 Adaptive，其余字段无意义 */
} gsm_policy_report_t;

int  gsm_policy_report(gsm_handle* h, gsm_policy_report_t* out);

/* 统计。全部**不含隐式状态推进** —— 读统计量不会顺带收割在途迁移。 */
size_t   gsm_hot_used(gsm_handle* h);
size_t   gsm_hot_valid(gsm_handle* h);
size_t   gsm_cold_used(gsm_handle* h);
size_t   gsm_hot_capacity(gsm_handle* h);
size_t   gsm_cold_capacity(gsm_handle* h);
size_t   gsm_hot_peak(gsm_handle* h);
size_t   gsm_slot_bytes(gsm_handle* h);
int      gsm_num_blocks(gsm_handle* h);
int      gsm_num_blocks_in_gpu(gsm_handle* h);
int      gsm_pending_migrations(gsm_handle* h);
int      gsm_is_migrating(gsm_handle* h, int64_t block_id);
int      gsm_is_pinned(gsm_handle* h, int64_t block_id);
uint32_t gsm_eviction_fingerprint(gsm_handle* h);
int64_t  gsm_eviction_count(gsm_handle* h);

/* 错误与诊断 */
const char* gsm_last_error(gsm_handle* h);
const char* gsm_result_str(int code);

typedef struct {
    int32_t  in_gpu;
    int32_t  valid;
    int32_t  migrating;
    int32_t  pinned;
    int32_t  has_cold;      /* 在冷池里持有槽位 */
    int32_t  _pad;
    size_t   size;
    uint64_t gpu_addr;      /* 0 = 当前不在显存 */
    int64_t  last_access;
    int64_t  access_count;
    int64_t  alloc_time;
} gsm_block_info_t;

int gsm_block_info(gsm_handle* h, int64_t block_id, gsm_block_info_t* out);
/* 槽位地址（arena 内的绝对地址）。测试用它断言"换回的块落在了刚释放的槽位"。 */
uint64_t gsm_gpu_addr(gsm_handle* h, int64_t block_id);

/* 数据完整性辅助。花样在**宿主**用确定性 PRNG 生成后 memcpy 进出显存，
 * 走 swap_stream 并立即同步 —— 不落在默认流上跟迁移打架。 */
int gsm_fill_pattern(gsm_handle* h, int64_t block_id, uint64_t seed);
int gsm_verify_pattern(gsm_handle* h, int64_t block_id, uint64_t seed,
                       int64_t* out_mismatch_bytes);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* GPU_SWAP_C_H */
