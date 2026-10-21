# Hygon Z100L：原生 DTK 后端

本 fork 使用 `STRATA_ENABLE_DCU=ON` 支持 Z100L/gfx906 wave64。测试日期：2026-10-21。
原生移植、跨流调度和设备端专家规划已实测。最新无 MTP 双卡生成 20.8–21.1 token/s，2048 token 预填充 212.2 token/s。
最新并行规划 MTP=3（验证窗口 4）：重复文本中位数 48.3–48.5 token/s，代码/中英文说明 36.9–44.1 token/s。
推荐在设备规划、融合及多行 GEMV 开关上增加 `STRATA_DCU_PAR_PLAN=1`；详见最后的清场重测一节。
前两轮表格保留作历史对照。
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
| 第二轮无 MTP 实测代码 | `f3e63609`；同日旧版对照 `1aa1073f`；上游基线 `9de04cd4` |
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

在第一轮中，双卡生成略慢于 llama.cpp（17.0–17.2 对 17.6–18.1 token/s），
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
第一轮未验证 MTP。多并发、其他模型或卡型，以及 CUDA、AMD HIP、SYCL 后端的构建回归仍未验证。
数据：[2026-10-21-hygon-z100l](../bench/results/2026-10-21-hygon-z100l/)。

## 第二轮：逐层插针与设备端专家规划

优化代码：`011e6908`，CLI 约束修复及本轮完整复测：`f3e63609`。本轮仍使用原生 DTK、完整 top-10 路由、相同量化和无草稿解码。
下面的开关均为 opt-in；不设置时保留原来的执行路径。

- `STRATA_DCU_GPU_PLAN=1`：GPU 根据 resident table 生成常驻专家分组，提前开始计算命中专家，和 CPU 未命中专家计算重叠；省掉等待主机发布命中计划、复制该计划，以及 PCIe 分担为零时的等待和空专家启动。
- `STRATA_DCU_FUSE_COMBINE=1`：在设备端规划路径中，把 CPU 行读取、GPU 命中结果相加、按原顺序加权归约合成一个算子。只从映射主机内存读取未命中行，省去中间 `parts` 的写入与再次读取。
- `STRATA_DCU_MMVQ_ROWS=2` 或 `4`：Q8_0 单列 GEMV 每个 wave 处理多行，复用激活；每行的点积和归约顺序保持不变。其他格式与多列调用不变，默认是 `1`。

设备端规划当前要求 native pack、`--pcie-frac 0 --adapt-every 0`、完整专家路由，不支持 batch、pipeline windows、peer/helper 或 FS slots。无 MTP 测量使用 `f3e63609`；后续 MTP 兼容代码及实测另列。
程序会拒绝不兼容的启动配置和非零 PCIe 分担请求。融合归约不覆盖全专家常驻的已有快路径，GPU 阶段诊断开启时也会关闭该融合，避免把有诊断的数值当性能结果。

### 插针和时间线复现

`STRATA_DCU_PROFILE_JSONL=/absolute/path/layers.jsonl` 记录每张卡、每一层的 CPU 等待、专家服务时间及最大值，并分别输出 prefill/decode。
仅设置该变量不会添加 GPU 时间戳内核。需要 GPU 分段时再设置 `STRATA_VERIFY_PROFILE=1`；这会扰动调度并关闭部分融合，只用于定位。
`gpu_segment_ns[1..24]` 为连续阶段差分，19/20/21/22/23/24 分别对应 waitA、VRAM hits、waitB、PCIe groups、waitCPU、copy/combine；25 为层间隔、26 为最后一层后的 head，27..32 在逐层原始记录中未拆分。
CPU 等待不是 GPU 空闲，服务时间包含分发及线程池工作；这些计时范围有重叠，不能相加成全局耗时。

```bash
source /usr/local/hygon/dtk-25.04/env.sh
# baseline 是 validate.py 生成的目录；out 必须尚不存在。
python3 tools/dcu/profile.py --engine build-dcu/strata \
  --baseline /tmp/strata-dcu-validation --out /tmp/strata-native-trace
# hipprof 输出 DB 文件名随进程变化。先预热，在首个输出 token 后才启动采样。
python3 tools/dcu/analyze_profile.py --db /tmp/strata-native-trace/hip-prof-PID.db \
  --layers /tmp/host-counters/layers.jsonl --out /tmp/profile-summary.json \
  --trace /tmp/strata-trace.json.gz
```

Chrome trace 可解压为 JSON 后导入时间线查看器。解析器使用 `STR_TABLE` 解析内核名，并从 HIP 的 `args` 解析 API 名；两张表的数字 Name 不是相同的命名空间。
输出同时列出累计时长和区间并集，避免把重叠内核或与 GPU 重叠的主机 API 耗时重复计算。

旧路径原生采样区间 992.65 ms：32187 次内核启动，内核累计 885.02 ms，轮询等待累计 388.59 ms；
Q8 wave GEMV 累计约 136–137 ms（另有其他格式 MMVQ），显式 COPY 记录累计 0.88 ms，名字含 copy 的内核另有 13.26 ms。
映射内存读取也可能发生在其他算子内部，不能把 0.88 ms 当作全部 PCIe 成本。
同一采样中 `hipGraphLaunch` 主机调用累计 311.73 ms，和设备执行重叠，不能再加到 GPU 时间上。
单条序列按层切分时，两个设备的计算存在前后依赖；该采样没有观察到跨卡内核重叠。

无 GPU 时间戳的 CPU 插针中，两个长生成请求累计专家服务约 11.5/15.3 ms 每 token；
有阶段诊断时 waitA 约 27–29 ms、GPU 命中专家约 5 ms 每窗口。诊断改变了运行路径，不能用这些数值直接预测提速上限。


### 第二轮完整复测结果

双卡按层切分，关闭 MTP、lookup、suffix、缓存复用和路由裁剪。新配置开启以上三个开关，
`MMVQ_ROWS=4`、8 workers、显存预留 2048 MiB；旧二进制 `1aa1073f` 为原配置，预留 4096 MiB。
专家缓存槽位因此从 21457 增加到 23347。下表是整套配置的对照，不能把全部增益归因于单个算子融合。
同日先完成新配置完整验证，再运行旧二进制，最后复跑新配置。每项三次中位数，128 输出 token，`cache_n=0`。

| 输入 token | 旧版生成 token/s | 新配置生成 token/s | 新配置再次复跑 token/s | 新配置预填充 token/s | 新配置 TTFT 秒 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 17.3 | 21.1 | 21.2 | 116.7 | 1.152 |
| 512 | 17.4 | 21.1 | 21.2 | 185.9 | 2.802 |
| 1024 | 17.5 | 20.8 | 21.3 | 202.9 | 5.097 |
| 2048 | 17.3 | 20.9 | 20.9 | 212.2 | 9.702 |

新配置完整验证比同日旧版提升约 19%–22%。三类自然语言/代码请求各两次，固定生成 256 token：

| 请求 | 旧版生成 token/s | 新配置生成 token/s | 新配置再次复跑 token/s |
| --- | ---: | ---: | ---: |
| Python 日志解析 | 18.35 | 21.60 | 21.60 |
| 英文哈希表说明 | 18.80 | 21.85 | 21.90 |
| 中文分布式任务队列说明 | 18.55 | 21.55 | 21.70 |

这些固定输出预算用于测速度，不是完整任务质量分数。重复文本性能请求的输出 SHA256 与旧版一致；
两类自然语言长答案与旧配置不同，但各配置内重复及新配置再次复跑一致。相同缓存容量、相同 GEMV 配置下，只切换设备规划与融合的追加对照，12 组输出 SHA256 全部一致。不能据此声称全部文本逐 token 等价。
原始答案、输入、哈希和单次计时都保留在数据目录中。

完整新配置通过 21 项功能请求、重复及跨进程重启检查、12 组定长性能请求和 6 组代码/语言请求。
HTTP 三次请求通过，长输入分别实际为 2435、3635 token。两卡各 17 项既有选定测试，加上各 3 项设备规划、映射归约和 MMVQ 回归，共 40 次 CTest 执行通过。
设备规划、映射归约各 128 组随机/边界用例通过，后者包含 1–8 token、全命中/混合/全未命中及主机命中行 NaN 污染检查。
Q8 多行 GEMV 的 1/2/4 行配置在两卡做过单列、多列逐位对比。

新配置两卡 VRAM 采样峰值约 30.9 GiB/卡，READY 报告空闲 1517 MiB；4096 上下文只是本次已测范围。
不要把更低预留直接推广到更长上下文或其他模型。

### 优化后的时间线与消融

两个原生采样都包含 696 次 MoE 层归约，下面是采样区间累计值，包含 profiler 开销，不是独立无诊断跑分：

| 项目 | 旧路径 | 优化配置 |
| --- | ---: | ---: |
| 采样跨度 ms | 992.65 | 784.54 |
| 内核启动次数 | 32187 | 25227 |
| 等待内核次数 | 2088 | 696 |
| 等待内核累计 ms | 388.59 | 226.38 |
| 内核累计 ms | 885.02 | 702.04 |
| 独立 copy_rows_from_mapped 次数 | 696 | 0 |
| MMVQ 累计 ms | 158.76 | 145.45 |

优化后的等待仍占内核累计时间约 32%。设备规划本身累计约 19.27 ms，并非免费；CPU 未命中计算的等待和 dense/head GEMV 仍值得继续优化。
两个采样均未观察到跨卡内核重叠，单序列层切分仍有设备间前后依赖。
Chrome 时间线分别为 `baseline-trace.json.gz`、`optimized-trace.json.gz`；SQLite 原始数据也已压缩保留。

同二进制 GPU 规划开关实验从约 17.6–17.9 到 18.3–18.6 token/s。
融合独立消融的整模型差异较小，单行配置的中位数 18.05 与 18.2 不足以证明显著收益。
清场后的单算子微基准在宽度 3072、不同命中比例下，旧三步约 20–27 微秒，融合约 8–14 微秒；不能将该倍率外推到整个模型。
多行 GEMV 与增加专家缓存带来后续收益，4 workers 和 12 workers 的清场复测均不及 8 workers。

三组受另一批 vDCU 容器持续 GEMM 影响的结果已排除，容器未被本任务修改或停止；等待其自然退出后重跑。
另外三次启动曾因本任务检查进程与跑分启动重叠而 OOM，发生在 READY 前，保留日志并排除，不计为有效跑分。
最终队列采用进程锁串行执行。排除名单和所有原始失败记录均保留。

复现新配置完整验证：

```bash
export HIP_VISIBLE_DEVICES=0,1 STRATA_ROUTE_TAIL_SKIP=0
export STRATA_DCU_GPU_PLAN=1 STRATA_DCU_FUSE_COMBINE=1 STRATA_DCU_MMVQ_ROWS=4
python3 tools/dcu/validate.py --engine build-dcu/strata \
  --pack "$PACK" --gguf "$MODEL" --gpus 2 --workers 8 --reserve-mib 2048 \
  --chat-bench --out /tmp/strata-deep-validation
```

数据：[deep-optimization](../bench/results/2026-10-21-hygon-z100l/deep-optimization/)，
最终无 MTP 记录分别为 `deep-final`、`deep-old-final`、`deep-best-confirm`，不要将探索、预热、诊断和受干扰样本混进最终中位数。

## MTP：每轮 3 个草稿，验证窗口 4

代码 `ebe417d8` 允许设备端规划路径使用最多 8 token 的 MTP 窗口；本次实际只验证窗口 4。
目标仍为原 UD-Q3_K_XL。草稿分支使用官方 `Qwen/Qwen3.8-Flash-Next` 固定 revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540` 的 31 个 MTP BF16 张量，逐个 SHA256 校验后转换为 Q2_0 专家、兼容 dense 和 CJK 草稿词表。
CJK 草稿词表有 106299 个 ID；目标模型继续用完整词表，接受率只代表这套已测草稿配置。

下载及转换文件持久化在：

```text
/share/maasfile/model/public/Qwen/Qwen3.8-Flash-Next/mtp-de4b8e4d43b917e7706784d8bb445c9af86a3540/
```

官方端点在该网络下超时，实际从 hf-mirror.com 按相同固定 revision 获取，保留全部源张量 SHA256 检查。
原始张量约 5.21 GB，GGUF 约 0.889 GB，`rt/` 运行文件约 786 MiB；节点本地缓存为
`/root/code/strata-dcu/mtp-rt-de4b8e4d`，四个运行文件的 SHA256 已再次核对。
源 revision、运行文件大小与哈希见数据目录中的 `mtp-runtime-provenance.json`。

固定参数为 `--mtp <rt> --spec 4 --mtp-max-t 4 --spec-min-p 0 --suffix-draft 0 --lookup-chain 0`。
每轮最多三个草稿，并非验证窗口 3。仍为 4096 上下文、8 workers、PCIe 分担 0、2048 MiB 预留、完整路由和冷 prompt。
MTP 增加显存占用，自动专家缓存槽位从 23347 降至 22982；不能把无 MTP 的缓存容量直接套到 MTP。

| 输入 token | 普通执行路径 MTP 生成 token/s | 设备端串行规划＋融合 MTP 生成 token/s |
| ---: | ---: | ---: |
| 128 | 45.3 | 43.3 |
| 512 | 未测 | 42.6 |
| 1024 | 未测 | 44.6 |
| 2048 | 44.9 | 44.5 |

三次中位数，每次输出 128 token。草稿接受率约 97%–100%，这类重复文本容易预测，不可推广到自然语言。
串行规划＋融合对 MTP 尚未显示稳定提速；不能把无 MTP 的收益机械叠加到 MTP 上。

| 请求 | 普通路径 MTP 生成 token/s | 串行规划＋融合 MTP 生成 token/s | 草稿接受率 |
| --- | ---: | ---: | ---: |
| Python 日志解析 | 40.45 | 40.55 | 74.3% |
| 英文哈希表说明 | 36.40 | 36.35 | 63.3% |
| 中文任务队列说明 | 33.95 | 34.00 | 57.1% |

各两次、每次固定输出 256 token。普通路径和串行设备规划路径共同覆盖的 12 组输出 SHA256 全部一致，
草稿提出/接受计数也一致。该比较保持相同缓存容量、MTP、GEMV 配置，只切换设备规划与映射融合。
MTP 完整验证通过 21 项功能检查、重复和跨进程重启检查、12 组定长性能和 6 组语言/代码请求；HTTP 三次请求通过，含 2435、3635 token 输入。
实际进程映射仍只加载原生 DTK 的 libgalaxyhip 和 rocBLAS。

MTP 两卡采样峰值约 30.9 / 31.4 GiB，READY 报告空闲 1505 MiB，但测量中的显存峰值余量更小。
这是 4096 上下文下的有限验证，不能据此保证更大上下文或长期多并发稳定性。
原始数据分别为 `deep-mtp3-compat`、`deep-mtp3-final`；代码和二进制摘要分别见 `mtp-source.json`、`mtp-binary.sha256`。

复现 MTP 完整验证（继承前面的原生 DTK 环境）：

```bash
export STRATA_DCU_GPU_PLAN=1 STRATA_DCU_FUSE_COMBINE=1 STRATA_DCU_MMVQ_ROWS=4
python3 tools/dcu/validate.py --engine build-dcu/strata \
  --pack "$PACK" --gguf "$MODEL" --gpus 2 --workers 8 --reserve-mib 2048 \
  --mtp /root/code/strata-dcu/mtp-rt-de4b8e4d --drafts 3 \
  --chat-bench --out /tmp/strata-mtp3-validation
```

## 追加：并行生成 resident 计划

代码 `51108284`，新增 opt-in `STRATA_DCU_PAR_PLAN=1`，与 `STRATA_DCU_GPU_PLAN=1` 配合使用。
每个路由项由一个 lane 处理，使用共享内存扫描，仍按专家第一次出现的顺序分组、按原始路由位置排列组内条目。
未命中和非法 ID 不进入命中计划；浮点算术保持不变。未设置该开关时沿用已测的单线程规划。

两卡分别对串行和并行路径运行 128 组检查，总共 512 组逐位计划检查通过。
同一二进制、每项 512 次的 event 微基准结果：

| 路由项数 | 串行规划 微秒/次 | 并行规划 微秒/次 |
| ---: | ---: | ---: |
| 10（无 MTP） | 19.22–19.23 | 7.24–7.25 |
| 40（验证窗口 4） | 142.88–142.91 | 15.24 |
| 80（边界用例） | 363.42–363.50 | 21.62–21.70 |

这是 event 测得的单次操作开销，包含连续提交影响，不是整模型倍率。
无 MTP 的追加跑分：128/2048 输入三次中位数为 20.9/21.3 token/s；Python/英文说明/中文说明两次中位数为 22.05/22.10/22.00 token/s。
相对串行规划，整模型差异较小，不能据此断言无 MTP 下有稳定收益。共同的 12 组输出 SHA256 均相同。
原始数据为 `deep-parallel-nomtp`、`deep-plan-micro-*.log` 和 `parallel-nomtp-comparison.json`。


并行规划的 MTP 首次试跑期间，另一批 vDCU 压测进程重新出现，READY 报告空闲显存为 0。
该目录 `deep-parallel-mtp3` 已明确标为受干扰并排除，约 15 token/s 的样本不用于评价规划实现。
未修改或暂停外部容器；等待其退出、显存释放后已完成追加重测，结果如下。


### 并行规划 MTP 清场重测

相同原生 DTK、目标量化、MTP=3、窗口 4、8 workers、2048 MiB 预留与完整路由。

| 输入 token | 串行设备规划 token/s | 并行设备规划 token/s |
| ---: | ---: | ---: |
| 128 | 43.3 | 48.5 |
| 2048 | 44.5 | 48.3 |

三次中位数、每次输出 128 token。代码和语言请求各两次、输出 256 token：

| 请求 | 串行规划 token/s | 并行规划 token/s |
| --- | ---: | ---: |
| Python 日志解析 | 40.55 | 44.10 |
| 英文哈希表说明 | 36.35 | 39.50 |
| 中文任务队列说明 | 34.00 | 36.90 |

清场重测的 12 组共同输出 SHA256、草稿提出数与接受数均与串行规划路径一致。
两项基本答案检查、六组定长跑分、六组语言/代码请求及三项 HTTP 检查通过，最长 HTTP 输入 3635 token。
这次是整数规划实现的追加验证，前面的 21 项完整 MTP 功能及重启检查使用串行规划代码；不将两批验证混为同一批。
数据为 `deep-parallel-mtp3-clean-1` 和 `parallel-mtp-comparison.json`，代码及二进制摘要见 `parallel-source.json`、`parallel-binary.sha256`。
复现时在 MTP 验证命令前增加 `export STRATA_DCU_PAR_PLAN=1`，其余参数保持相同。
单算子倍率不可外推到整模型，实际选择应依据上述场景跑分。
