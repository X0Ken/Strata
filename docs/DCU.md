# Hygon Z100L：原生 DTK 后端

本 fork 使用 `STRATA_ENABLE_DCU=ON` 支持 Z100L/gfx906 wave64。测试日期：2026-10-21。
原生移植及跨流调度优化已完成。最新双卡生成 17.0–17.2 token/s，2048 token 预填充 210.2 token/s。
最新数据见“优化后完整复测”；下方首轮表保留作历史对照。
上游桌面安装包、自动下载的引擎和一键安装器尚未适配本后端，请按下面手动构建。

## 实测环境与运行库

| 项目 | 配置 |
| --- | --- |
| CPU | KVM 虚拟机，16 vCPU，Hygon C86 7380，AVX2，无 AVX512 |
| 内存 | 62.8 GiB，无 swap |
| DCU | 2 × Hygon Z100L，各 32752 MiB，gfx906，wave64 |
| 系统 | Ubuntu 22.04.1，Linux 5.15.0-43-generic |
| 驱动 / SDK | hydcu 6.2.31 / Hygon DTK 25.04，Clang 15 |
| 构建工具 | CMake 3.31.6，Release，STRATA_PORTABLE=ON |
| 当前实测 Strata 代码 | `1dad29d5024eb441a70f2862af55efc145f51590`；上游基线 `9de04cd4` |
| ggml 固定版本 | `3cf03257f219afbe7334045ff7c6a06ac68c627d` |

已从四次实际推理进程的 `/proc/PID/maps` 确认加载：

```text
/usr/local/hygon/dtk-25.04/hip/lib/libgalaxyhip.so.5.2.25085.1211-205b0686
/usr/local/hygon/dtk-25.04/lib/librocblas.so.4.1
```

BLAS 直接调用 DTK 提供的 rocBLAS，没有经过 hipBLAS；运行进程未加载 CUDA 库或 `/opt/rocm` 库。
构建缓存中 `STRATA_ENABLE_CUDA`、`STRATA_ENABLE_HIP`、`STRATA_HIP_GFX906` 均为 OFF。
复用的 wave64 内核仍使用源代码级 CUDA/HIP 名称映射，DTK 自身也使用 amdgcn/gfx906/HIP ABI 名称。
这不代表启用了 AMD 运行库或设备型号伪装。部分通用日志仍沿用 CUDA0/1 表示逻辑设备编号。
不要设置 `HSA_OVERRIDE_GFX_VERSION`。

## 构建

先安装节点对应的海光驱动和 DTK。以下命令在仓库根目录执行，要求 CMake 3.24 或更新版本；本次使用 3.31.6。
`STRATA_GGML_DIR` 指向上述固定提交的 llama.cpp 源码目录，避免构建期间再次联网拉取。

```bash
source /usr/local/hygon/dtk-25.04/env.sh
export LIBRARY_PATH="$ROCM_PATH/llvm/lib/clang/15.0.0/lib/linux:${LIBRARY_PATH:-}"
cmake -S . -B build-dcu \
  -DSTRATA_ENABLE_DCU=ON -DSTRATA_DTK_ROOT="$ROCM_PATH" \
  -DCMAKE_HIP_ARCHITECTURES=gfx906 -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_GGML_DIR=/root/code/strata-dcu/ggml-pinned \
  -DSTRATA_PORTABLE=ON -DSTRATA_BUILD_TESTS=ON -DSTRATA_MMQ_TQ_OVERLAY=OFF \
  -DCMAKE_C_COMPILER="$ROCM_PATH/llvm/bin/clang" \
  -DCMAKE_CXX_COMPILER="$ROCM_PATH/llvm/bin/clang++"
cmake --build build-dcu -j8 --target strata strata-device
```

本节点的 NFS `/share` 会使 CMake 探测卡住，实际构建在私有 mount namespace 内执行：
`unshare --mount --propagation private bash`，只在该 namespace 中 `umount -l /share` 后构建。
这仅是该节点的构建环境处理，不要在主机全局卸载共享目录。

## 模型与运行配置

模型为 `unsloth/Qwen3.8-Flash-Next-GGUF` 的 `UD-Q3_K_XL`，三分片合计约 83.81 GiB。
固定 HF revision：`766911a6b7369840a91dbcd95f9f997acaab6cd6`，已有三分片 SHA256 校验。
持久化目录：`/share/maasfile/model/public/unsloth/Qwen3.8-Flash-Next-GGUF/UD-Q3_K_XL/`。
实测读取节点本地缓存 `/root/code/llama-dcu-validation/model-cache/`，避免 NFS 干扰跑分。

Python 运行依赖见仓库 `requirements.txt`。打包还需要固定 ggml 源码的 `gguf-py`：

```bash
export GGML=/root/code/strata-dcu/ggml-pinned
export MODEL=/root/code/llama-dcu-validation/model-cache/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf
export PACK=/root/code/strata-dcu/pack-ud-q3
PYTHONPATH="$GGML/gguf-py${PYTHONPATH:+:$PYTHONPATH}" STRATA_GGUF_PY="$GGML/gguf-py" \
  python3 tools/iq_pack.py --gguf "$MODEL" --out "$PACK" --compat-bf16
```

生成约 1.38 GiB 的密集权重、索引和 tokenizer，不再复制专家权重。
460 个张量转换为兼容格式，其中 196 个有 BF16 舍入，最大绝对误差 0.0144；专家与 PLE 字节保持原样。
因此不能将整个打包模型称为与原 GGUF 位级一致，也尚未完成全面质量评测。

基线固定：4096 上下文，prefill chunk 128，8 个 CPU worker，24 GiB resident RAM 预算，
`--expert-cache auto --expert-profile data/expert-profile.bin --vram-reserve-mib 4096`。
双卡增加 `--layer-split 24 --split-device 1`，单卡 `HIP_VISIBLE_DEVICES=0`，双卡 `0,1`。
`STRATA_ROUTE_TAIL_SKIP=0` 完整计算路由选中的专家；`--pcie-frac 0` 将缓存未命中的专家交给 CPU。
关闭适配、prompt cache、suffix draft。打包格式要求 `--spec 2`，但不加载 MTP，实测没有草稿，每轮一个 token。
模型 EOS 为 248044、248046。

复现完整验证（输出目录请使用新的空目录）：

```bash
python3 tools/dcu/validate.py --engine build-dcu/strata \
  --pack "$PACK" --gguf "$MODEL" --gpus 2,1 --out /tmp/strata-dcu-validation
```

原始记录的 `configuration` 行包含完整引擎参数、可见卡、运行库、启动时间和实际缓存大小。
双卡 HTTP 测试配置见数据目录 `http-config.json`；替换其中 exe、cwd、模型和日志绝对路径后运行：

```bash
HIP_VISIBLE_DEVICES=0,1 STRATA_ROUTE_TAIL_SKIP=0 \
  python3 -m serve.server --engine strata --config /path/to/http-config.json \
  --host 127.0.0.1 --port 18101 --offline
```

## 首轮基线性能

本节为修正前代码 `bbff7895` 的历史基线。以下均为三次运行的中位数，每次生成 128 token，输入 128/512/1024/2048 token。
使用相同 token IDs，`cache_n=0`，关闭推测解码；计时不包含模型启动。
专家缓存随请求保持常驻，并非每次启动冷模型。
预填充与生成速度来自各引擎内部计时。TTFT 为客户端首 token：Strata 使用 IPC，llama.cpp 使用 HTTP SSE，
两者协议开销不同。全部原始样本与输入 SHA256 公开保留，不剔除慢样本。

| 引擎 | 输入 token | 预填充 token/s | 生成 token/s | TTFT 秒 |
| --- | ---: | ---: | ---: | ---: |
| Strata 单卡 | 128 | 65.4 | 8.1 | 2.079 |
| Strata 单卡 | 512 | 64.2 | 8.0 | 8.105 |
| Strata 单卡 | 1024 | 62.7 | 8.0 | 16.455 |
| Strata 单卡 | 2048 | 61.7 | 8.2 | 33.328 |
| Strata 双卡 | 128 | 52.5 | 3.5 | 2.753 |
| Strata 双卡 | 512 | 81.5 | 3.5 | 6.517 |
| Strata 双卡 | 1024 | 88.1 | 3.5 | 11.903 |
| Strata 双卡 | 2048 | 169.7 | 10.1 | 12.157 |
| llama.cpp 双卡 | 128 | 55.1 | 17.9 | 2.483 |
| llama.cpp 双卡 | 512 | 77.4 | 18.1 | 6.852 |
| llama.cpp 双卡 | 1024 | 82.6 | 17.6 | 12.658 |
| llama.cpp 双卡 | 2048 | 83.8 | 17.8 | 24.693 |

双卡 Strata 在这组输入中的生成速度不单调，短输入还慢于单卡；首轮不能据此宣称多卡加速。
2048 token 的双卡预填充中位数更快，生成仍落后于 llama.cpp。待进一步定位缓存、CPU 专家与阶段等待开销。
llama.cpp-dcu 对照代码 `63071d45`，同一节点同日运行，参数：
`-ngl all -ncmoe 4 -ot per_layer_token_embd=CPU -sm layer -ts 1,1 -c 4096 -b 512 -ub 128 -fa on -t 8 -np 1 --jinja`。
Strata pack 有上述 BF16 转换，因此这也是两个实际执行方案的对照，不能当成只有调度实现不同的严格消融实验。

资源监测每秒一次。双卡每张 VRAM 峰值约 29.8 GiB，单卡约 29.8 GiB。
双卡进程 RSS 峰值约 15.7 GiB，单卡约 55.6 GiB；RSS 含可回收的 mmap 文件页，不能直接等同独占物理内存。
全机 MemAvailable 最低双卡约 34.9 GiB、单卡约 30.3 GiB，详见原始资源记录。

## 瓶颈定位与调度优化

旧 DCU 分支没有定义 `STRATA_USE_HIP`，共享专家的 `sh_stream_on()` 因而走了 CUDA 默认分支：
每层把共享专家放到另一条流，添加事件并等待合流。原生 DTK 上这些跨流依赖的代价抵消了重叠收益。
GPU 诊断模式为了计时会自动关掉这条分支，所以不能直接拿诊断模式的较高吞吐当成性能结论。

使用同一旧二进制，以 `STRATA_SH_STREAM=1/0` 做对照，并扫描 `--pcie-frac` 和辅助线程放置。
下面均为双卡、无诊断、128 输出 token；每个输入长度分别取中位数。
`pcie-frac=0` 的未命中专家交给 CPU，`=1` 全部交给 DCU；两者都完整计算选中的专家。
每次启动实际缓存容量记录在 configuration 中，自动容量会略有波动。

| 配置 | PCIe 分担 | 每长度样本数 | 输入 128：生成 token/s | 输入 2048：生成 token/s |
| --- | ---: | ---: | ---: | ---: |
| 层切分，跨流开，线程默认 | 0 | 3 | 10.2 | 10.2 |
| 层切分，跨流开，线程默认 | 1 | 3 | 9.2 | 9.2 |
| 层切分，跨流开，辅助线程分离 | 0 | 3 | 10.0 | 10.1 |
| 层切分，跨流关，线程默认 | 0 | 3 | 17.5 | 16.5 |
| 层切分，跨流关，辅助线程分离 | 0 | 3 | 17.5 | 17.6 |
| 层切分，跨流关，辅助线程分离 | 1 | 3 | 16.0 | 16.2 |
| peer，跨流关，辅助线程分离 | 0 | 2 | 15.6 | 15.8 |
| helper，跨流关，辅助线程分离 | 0 | 2 | 14.8 | 14.9 |

默认线程放置、PCIe 分担为 0 时，关跨流后提升约 62%–72%。跨流开启时把分担拉到 1 反而更慢，
因此没有直接套用其他卡型的 PCIe 分担默认值。辅助线程从 CPU 0 移到空闲 CPU 的单独对照没有稳定收益，
不改变其默认值。peer 与 helper 也未超过关跨流后的层切分，并且各自只做了基础题和两次重复的定长跑分，
没有完成完整重启套件，因此继续推荐层切分。

引擎自身计数（PCIe 分担 0、两种输入长度合并六个样本的中位数）：

| 计时项 | 跨流开 | 跨流关 |
| --- | ---: | ---: |
| 整个验证窗口 ms/token | 98.145 | 58.195 |
| 首阶段等待 GPU 到达 ms/token | 33.025 | 12.145 |
| 首阶段 CPU 专家计算 ms/token | 10.275 | 10.020 |

CPU 工作量基本没有下降，主要收益来自调度等待减少。这些计数含阶段、重叠及不同范围，不能简单相加当成全局时间占比。
修正提交 `06e3213a` 仅将原生 DCU 的共享专家跨流默认关闭，保留 `STRATA_SH_STREAM=1` 手动对照；
不改变 CUDA 或上游 HIP 的默认行为，也不改变执行库。仍使用 DTK `libgalaxyhip` 与直接 rocBLAS。

七组探索共 36 次基本答案检查、88 次无诊断跑分及 2 次 GPU 阶段诊断。
88 次无诊断跑分的输出 token SHA256 全部一致；这是该组重复文本补全的证据，不是所有输入位级一致或完整质量证明。
原始数据和各阶段日志见 [optimization](../bench/results/2026-10-21-hygon-z100l/optimization/)。
首轮表中约 3.5 token/s 的低速与同批复测约 10.2 token/s 的差异没有单独归因，不用最慢那批计算优化倍数。

复现扫描时使用本机 `validate.py` 生成的目录，以便配置中的模型路径正确，两个输出目录都应尚不存在：

```bash
python3 tools/dcu/tune.py --engine build-dcu/strata --baseline /tmp/strata-dcu-validation \
  --out /tmp/strata-stream-on --shared-stream 1 --fractions 0 --repeats 3
python3 tools/dcu/tune.py --engine build-dcu/strata --baseline /tmp/strata-dcu-validation \
  --out /tmp/strata-stream-off --shared-stream 0 --fractions 0 --repeats 3
```

## 优化后完整复测

原生调度修改和验证工具已提交到 `main`：`1dad29d5024eb441a70f2862af55efc145f51590`。
重编译后显式取消 `STRATA_SH_STREAM`、`STRATA_VERIFY_PROFILE`、`STRATA_DECODE_TIMING` 环境覆盖，
验证源代码默认值，维持 PCIe 分担 0、辅助线程放置 off、4096 MiB 预留、完整路由及相同模型配置。
四次实际进程映射再次确认只使用 DTK 的 libgalaxyhip 和 rocBLAS。

两张卡各 17 项选定算子测试、42 次功能请求、重复/重启检查、24 次定长跑分全部通过。
HTTP 的两次请求也通过，长请求实际输入 2435 token。单卡与双卡各 12 个性能样本的输出 token 哈希均与调优扫描一致。

测量方式与首轮相同，三次中位数、128 输出 token、`cache_n=0`；同日 llama.cpp 对照沿用同一组 12 个原始样本。
Strata TTFT 使用 IPC，llama.cpp 使用 HTTP SSE，协议开销不同。这组性能输入是重复文本补全，不代表代码、长对话或多并发吞吐。

| 引擎 | 输入 token | 预填充 token/s | 生成 token/s | TTFT 秒 |
| --- | ---: | ---: | ---: | ---: |
| Strata 单卡 | 128 | 76.4 | 10.9 | 1.763 |
| Strata 单卡 | 512 | 75.3 | 10.8 | 6.890 |
| Strata 单卡 | 1024 | 72.6 | 10.9 | 14.197 |
| Strata 单卡 | 2048 | 70.1 | 11.1 | 29.328 |
| Strata 双卡 | 128 | 115.9 | 17.2 | 1.160 |
| Strata 双卡 | 512 | 182.1 | 17.1 | 2.874 |
| Strata 双卡 | 1024 | 200.7 | 17.1 | 5.166 |
| Strata 双卡 | 2048 | 210.2 | 17.0 | 9.809 |
| llama.cpp 双卡 | 128 | 55.1 | 17.9 | 2.483 |
| llama.cpp 双卡 | 512 | 77.4 | 18.1 | 6.852 |
| llama.cpp 双卡 | 1024 | 82.6 | 17.6 | 12.658 |
| llama.cpp 双卡 | 2048 | 83.8 | 17.8 | 24.693 |

双卡当前生成略慢于 llama.cpp（17.0–17.2 对 17.6–18.1 token/s），
2048 token 预填充约为对照的 2.5 倍（210.2 对 83.8 token/s），客户端 TTFT 中位数 9.809 对 24.693 秒。
继续推荐层切分、PCIe 分担 0，并保留显存余量；未用漏算专家或推测草稿换取这次增益。

新配置单卡 VRAM 峰值约 29.1 GiB，双卡每卡约 28.9 GiB。
双卡/单卡的全机 MemAvailable 最低约 47.2 / 30.7 GiB（引擎 READY 后按秒采样）。
进程 VmHWM 约 34.9 / 56.2 GiB，包含启动阶段峰值和可回收 mmap 文件页，与运行期 MemAvailable 不是同一口径。
所有启动配置、资源采样、答案、计时、输入及输出 SHA256 均在
[final](../bench/results/2026-10-21-hygon-z100l/optimization/final/)。

## 已验证与限制

- 每张卡 17 项选定 CTest，合计 34/34。包括新增 rocBLAS FP16/BF16、1024 线程启动、prefill device symbol 回归及 14 项已有算子测试。
- 单、双卡各 21 次基本答案检查，合计 42/42；六个短答案重复及跨进程重启保持一致。
- 单、双卡各 12 次定长性能测试，合计 24 次；同日 llama.cpp 双卡对照 12 次。
- OpenAI chat HTTP 两次请求通过，其中一个实际输入 2435 token，正确提取 `DCU_LONG_2048`。

移植中实际修复了 DTK 默认 256 线程限制与上游 1024 线程启动的冲突，
opt-in shared memory 查询报错时回退到实际 64 KiB，以及匿名 namespace device globals 的 symbol 查询失败。
低显存预留 700 MiB 的初始单卡方案曾在长输入后报 `no kernel image`；
改为 4096 MiB 预留、限制可见卡并关闭路由尾部跳过后，最终整套测试通过。
这些变更同时应用，尚未通过逐项消融证明该晚期错误的唯一原因。

上述是有限功能和重复性验证，不是完整模型质量评测或 24 小时稳定性证明。
未验证 MTP、多并发、其他模型或卡型，以及 CUDA、AMD HIP、SYCL 后端的构建回归。
数据：[2026-10-21-hygon-z100l](../bench/results/2026-10-21-hygon-z100l/)。
