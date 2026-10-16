/* policy_thresholds.h — **自动生成，不要手改**
 *
 * 生成器: gpu-swap/scripts/derive_thresholds.py
 * 数据源: kvcache-tracer/traces/ 下 12 个 jsonl
 * 校验:   python scripts/derive_thresholds.py --check
 *
 * 口径来自 kvcache-tracer/scripts/analyze.py（脚本直接 import 它，
 * 不重新实现）：读次数取 free 事件里的 reads 字段，生命周期按 FIFO
 * 配对 alloc/free 得到**每一段**获取区间，不是每个 block 一条。
 *
 * 测量条件：这批运行里**没有发生任何逐出**（GPU 池装得下工作集），
 * 所以它们刻画的是 block 的自然行为，**不是逐出行为**。
 * 拿这些常数去预测「换出之后会怎样」是超纲的。
 */

#pragma once

#include <cstdint>

// ---- 跨 workload 的保护率（读次数 > T 的块占比，逐文件取中位数）----
//   T=  50: code-gen=94.8%  legacy-c1=27.0%  legacy-c4=26.3%  legacy-c8=28.2%  long-summarize=98.9%  short-qa=82.7%
//   T= 100: code-gen=87.4%  legacy-c1=23.3%  legacy-c4=22.4%  legacy-c8=24.1%  long-summarize=97.2%  short-qa=31.1%
//   T= 121: code-gen=84.1%  legacy-c1=22.9%  legacy-c4=22.0%  legacy-c8=23.7%  long-summarize=96.4%  short-qa=8.3%
//   T= 456: code-gen=9.8%  legacy-c1=7.4%  legacy-c4=7.4%  legacy-c8=7.4%  long-summarize=49.9%  short-qa=0.2%
//
// T=50 的跨 workload 极差 = 72.6 个百分点。
// 这个极差就是"绝对阈值不可迁移"的量化形式：同一个常数在 legacy-c1 上
// 放行 73% 的块，在 long-summarize 上只放行 1%。候选集大小差一个量级，
// 策略的实际行为就不是同一个策略了。项目一在 tau 上栽过一次同一类
// 的坑（tau 取 63ms 时 adaptive 逐位等于 LRU）。
// 因此 **主规则必须是相对分位**，下面这个常数只做次级保险 + 夹断。

static constexpr int64_t GSM_HOT_ACCESS_THRESHOLD = 50;
static constexpr double  GSM_MAX_PROTECTION_RATE = 0.3000;

// ---- 年龄规则（毫秒）----
//
// 解释不是"生命周期 < 1s 的块短命所以别换" —— **生命周期在线不可判**，
// 要等 free 才知道。在线可判的是**年龄**，正确解释是"刚分配 = 马上
// 还会被读"：块刚分配说明它的请求还在跑，计数低只是因为还没轮到它。
//
// 取值 = 跨 workload 的生命周期 P25 最小值 = 710 ms。
// 各 workload 的 <1s 区间占比（最高者）：66.3%
// 逐 workload 生命周期分位：
//   code-gen         P10=1921ms P25=4204ms P50=5263ms P90=7315ms <1s=5.2%
//   legacy-c1        P10=552ms P25=710ms P50=740ms P90=2098ms <1s=66.3%
//   legacy-c4        P10=563ms P25=721ms P50=755ms P90=2813ms <1s=64.2%
//   legacy-c8        P10=725ms P25=758ms P50=1217ms P90=4125ms <1s=42.1%
//   long-summarize   P10=4429ms P25=5192ms P50=6138ms P90=7808ms <1s=1.6%
//   short-qa         P10=472ms P25=1056ms P50=1403ms P90=1908ms <1s=22.4%
static constexpr int64_t GSM_YOUNG_AGE_MS = 710;

// ---- 换出代价比（来自项目一 PCIe 实测，不是 trace）----
//
// 重算一个块 1620 us vs 换回一个块 52.4 us = **31x**。
// 出处: kvcache-tracer/GUIDE.md 的 prefill 重算实测与 PCIe 实测表。
// 这个比值是 AdaptivePolicy 里 GDSF 打分的 cost 项。
static constexpr double GSM_RECOMPUTE_US_PER_BLOCK = 1620.0;
static constexpr double GSM_SWAP_IN_US_PER_BLOCK = 52.4;

// ---- 数据源指纹（--check 用）----
// 生成时的 trace 文件 md5，任一个变了就说明阈值该重算。
//   c1.12400.jsonl  c8001ada762065dc1c48592a9f374a29
//   c4.14108.jsonl  b05f9433786d755b1d63325d77ec4ea5
//   c8.15281.jsonl  9f74f37a53446c3a9373605ade1bfb53
//   code-gen.c1.49860.jsonl  15e62650de33e36417be5759d2e50123
//   code-gen.c4.55701.jsonl  21cfbfb188da554477dbf9d31bd3e344
//   code-gen.c8.58071.jsonl  c2a73c2af2bff9d963c9dc8181842f9a
//   long-summarize.c1.59816.jsonl  d8ae6dccd8ed591dab46e8a40c7e85dc
//   long-summarize.c4.65972.jsonl  6d1d86b1d2ba533f38c630b6aed59798
//   long-summarize.c8.68418.jsonl  3f769c1bf9540593b89d3afb7847a6bf
//   short-qa.c1.39661.jsonl  3cb4338cba5aeef0476e674c7b223976
//   short-qa.c4.45694.jsonl  3c5cd173f23ba6a9e3342667162871b4
//   short-qa.c8.48097.jsonl  c0d19d488d84382063901628b2b547c9
