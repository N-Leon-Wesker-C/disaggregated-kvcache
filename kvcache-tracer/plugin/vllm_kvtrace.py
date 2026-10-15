"""
vllm_kvtrace — 通过 vLLM 官方 plugin 机制安装 KV Cache 追踪 hook

============================================================
为什么需要 plugin 而不是「启动前 monkey-patch」
============================================================

vLLM v1 的 EngineCore 跑在一个**独立进程**里，而 KVCacheManager
（真正分配/释放 block 的对象）就在那个进程里。问题在于那个进程是
怎么来的：

  frontend 进程在创建 EngineCore 时会调用 get_mp_context()
  （vllm/v1/engine/utils.py），它的返回值取决于
  vllm/utils/system_utils.py:_maybe_force_spawn()：

      if cuda_is_initialized():
          os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

  实测（vLLM 0.25.1，单卡 4090）：frontend 刚 import vllm 时 CUDA
  还没初始化，此刻问是 "fork"；但真正创建 EngineCore 时 frontend
  已经初始化过 CUDA，答案是 **"spawn"**。

spawn 出来的是全新解释器，父进程里做的 monkey-patch 一律不存在。
所以在启动脚本里 patch 再 fork 的假设是错的 —— hook 一次都不会触发。

变通办法（换 fork、改 sitecustomize、subprocess 里 import）都很脆。
vLLM 自带的 plugin 机制就是为这种情况准备的，见
vllm/plugins/__init__.py：

    # Default plugins group will be loaded in all processes
    # (process0, engine core process and worker processes)
    DEFAULT_PLUGINS_GROUP = "vllm.general_plugins"

而且加载点在 EngineCoreProc.__init__（v1/engine/core.py:108），
早于 scheduler / KVCacheManager 的构造，正好能赶上。

============================================================
安装
============================================================

    cd kvcache-tracer/plugin
    pip install -e .

装完之后不用改启动命令，正常跑就行：

    vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000

hook 会在每个 vLLM 进程里自动装好，事件落到
    kvcache_trace.<pid>.jsonl

不想追踪时： unset / 别装这个 plugin 即可。
"""

import os
import sys


def _locate_tracer():
    """找到 tracer.py 所在目录。

    优先用环境变量 KVTRACE_SCRIPTS；否则按包安装位置推算
    （pip install -e . 时本文件仍在 kvcache-tracer/plugin/ 下，
    所以 ../scripts 是稳定的）。
    """
    scripts = os.environ.get("KVTRACE_SCRIPTS")
    if scripts and os.path.isfile(os.path.join(scripts, "tracer.py")):
        return scripts

    here = os.path.dirname(os.path.abspath(__file__))
    scripts = os.path.join(os.path.dirname(here), "scripts")
    if os.path.isfile(os.path.join(scripts, "tracer.py")):
        return scripts

    raise RuntimeError(
        "找不到 tracer.py。请设置 KVTRACE_SCRIPTS=<kvcache-tracer/scripts 的绝对路径>")


def register():
    """vLLM ``vllm.general_plugins`` 入口点调用的函数。

    vLLM 会在 frontend / EngineCore / worker 进程里各调用一次 ——
    这正是我们要的：真正跑 KVCacheManager 的那个进程也会被覆盖到。
    """
    scripts = _locate_tracer()
    if scripts not in sys.path:
        sys.path.insert(0, scripts)

    import tracer

    tracer.install_hooks()
    tracer.save_on_exit()

    # 子进程是 spawn 出来的，父进程的 print 看不到；带上 pid 方便对号入座
    print(f"[KVTrace][pid={os.getpid()}] plugin 已加载，hook 安装完成")
