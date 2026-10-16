# gpu-swap — 没有 cmake，就用 make。
#
# 三个产物：
#   build/libgpuswap.so      真的 CUDA 库（nvcc 编）
#   build/test_correctness   数据一致性 / 边界 / 错误路径（宿主 C++，g++ 编）
#   build/test_async         迁移与推理是否真的重叠（要写 kernel，nvcc 编）
#
# 为什么 .so 必须 -Xcompiler -fPIC：nvcc 默认不生成位置无关代码，
# Python 的 ctypes.CDLL 加载一个非 PIC 的 .so 会在重定位阶段失败。
# 为什么必须 -Wl,-rpath：只设 LD_LIBRARY_PATH 的话，换个 shell 就找不到
# libcudart，而报错信息是"cannot open shared object file"外加一串
# 看不懂的路径 —— 直接烧进 rpath，谁跑都一样。

CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
CXX       ?= g++

CUDA_INC  := -I$(CUDA_HOME)/include
CUDA_LIB  := -L$(CUDA_HOME)/lib64
# rpath 要两种写法：nvcc 的驱动层不认 -Wl,（直接报 "Unknown option"），
# 只接受 -Xlinker；而 g++ 两边都认。所以分开定义，各用各的。
NVCC_RPATH := -Xlinker -rpath -Xlinker $(CUDA_HOME)/lib64
CXX_RPATH  := -Wl,-rpath,$(CUDA_HOME)/lib64

# -gencode 而不是裸的 -arch：显式指定 SASS(compute_89) + PTX 两档。
# 只有 PTX 的话首次运行要 JIT，测出来的时间不是稳态的。
ARCH ?= -gencode arch=compute_89,code=sm_89

# 改了 include/*.h 而 make 说"无事可做" —— 实测栽过：一次 25 分钟的真实对比
# 跑完，才发现它量的是 9 分钟前的旧内核（.so 比头文件还老），白跑。
#
# ⚠️ nvcc **不能**用 -Xcompiler -MMD：生成的 .d 会把 nvcc 的中间产物
# （/tmp/tmpxft_*_swap_core.cudafe1.cpp）写成前置依赖，而那个文件编译完就没了，
# 于是下一次 make 直接报 "No rule to make target '/tmp/tmpxft_...'" —— 增量构建
# 坏掉。所以 nvcc 这条走 $(HEADERS) 钝器（见下面），-MMD 只给 g++ 用。
NVCCFLAGS := -std=c++17 -O2 -lineinfo $(ARCH) -Iinclude $(CUDA_INC) \
             -Xcompiler -fPIC -Xcompiler -Wall -Xcompiler -Wextra
CXXFLAGS  := -std=c++17 -O2 -Wall -Wextra -Iinclude $(CUDA_INC) -MMD -MP

BUILD := build

LIB      := $(BUILD)/libgpuswap.so
LIB_OBJS := $(BUILD)/swap_core.o $(BUILD)/swap_c_api.o

.PHONY: all clean test test-correctness test-async bench thresholds check-thresholds

all: $(LIB) $(BUILD)/test_correctness $(BUILD)/test_async $(BUILD)/bench_policy

$(BUILD):
	@mkdir -p $(BUILD)

# $(HEADERS) 是**钝器**（任一头文件变了就全量重编），但必须留着：
# -MMD 生成的 .d 是"构建之后"才有的依赖，对一个已存在的 build/ 目录
# **起不了引导作用** —— 第一次加上 -MMD 时，make 依然认为"无事可做"。
# 本仓只有几个头文件、两个源文件，全量重编的代价是零，换来的是
# "绝不静默用旧内核"。
HEADERS := $(wildcard include/*.h)

$(BUILD)/%.o: src/%.cu $(HEADERS) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(LIB): $(LIB_OBJS)
	$(NVCC) -shared $(ARCH) $(CUDA_LIB) $(NVCC_RPATH) $^ -o $@ -lcudart

# 测试用 g++ 编、链 .so。**不能用 nvcc 编 .cpp** —— 那会把整个测试
# 当成 CUDA 翻译单元，宿主侧的 class 定义会走 device 编译路径，
# 报一堆和代码无关的错。
$(BUILD)/test_correctness: tests/test_correctness.cpp $(LIB) | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@ -L$(BUILD) -lgpuswap $(CUDA_LIB) -lcudart \
	       $(CXX_RPATH) -Wl,-rpath,'$$ORIGIN'

# 异步测试要写 kernel，所以是 .cu，用 nvcc 编。
$(BUILD)/test_async: tests/test_async.cu $(LIB) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $(CUDA_LIB) $< -o $@ -L$(BUILD) -lgpuswap -lcudart \
	       $(NVCC_RPATH) -Xlinker -rpath -Xlinker '$$ORIGIN'

test: all
	./$(BUILD)/test_correctness
	./$(BUILD)/test_async

test-correctness: $(BUILD)/test_correctness
	./$(BUILD)/test_correctness
test-async: $(BUILD)/test_async
	./$(BUILD)/test_async

# 策略本身不碰 CUDA，所以这个用 g++ 编 —— 不必为它排 GPU 的队。
# （但用的是真实的数据结构，不是重新实现一遍：双实现漂移是本项目的老毛病。）
$(BUILD)/bench_policy: tests/bench_policy.cpp $(LIB) | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@ -L$(BUILD) -lgpuswap $(CUDA_LIB) -lcudart \
	       $(CXX_RPATH) -Wl,-rpath,'$$ORIGIN'

bench: $(BUILD)/bench_policy
	./$(BUILD)/bench_policy

# policy_thresholds.h 是**生成但提交**的：构建不该依赖另一个项目的 trace
# 目录。所以这里只提供重新生成和校验，不作为 $(LIB) 的依赖。
thresholds:
	python3 -I scripts/derive_thresholds.py
check-thresholds:
	python3 -I scripts/derive_thresholds.py --check

clean:
	rm -rf $(BUILD)

# 编译时生成的依赖（见 NVCCFLAGS 的注释）。放在末尾，首次构建时为空。
-include $(wildcard $(BUILD)/*.d)
