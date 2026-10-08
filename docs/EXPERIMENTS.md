# 最终公开数据测试

本文记录在 `codes/tests/data/public/performance` 上进行的最终完整测试。测试分为两部分：

1. 正确性测试：覆盖目录中声明的全部 `M`、`K`，逐行对照标准答案；
2. 端到端性能测试：比较 CPU、CUDA 单线程基线和 CUDA 自动分档。

## 1. 测试环境

| 项目 | 配置 |
| --- | --- |
| 测试日期 | 2026-10-08 |
| CPU | Intel Core i5-13500HX，14 核 20 线程 |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU，6 GiB，Compute Capability 8.9 |
| NVIDIA Driver | 592.82 |
| CUDA Toolkit | 12.9，nvcc 12.9.86 |
| 构建配置 | Visual Studio 2022，x64 Release，`CMAKE_CUDA_ARCHITECTURES=89` |
| CUDA batch size | 262144；本次每个输入均可在一个 batch 内处理完 |

运行正式矩阵前先执行：

```powershell
ctest --test-dir codes/build-cuda129 -C Release --output-on-failure
```

结果为 `1/1` 通过。

## 2. 比较的方法

正确性测试包含以下七种方法：

| 方法 | 含义 |
| --- | --- |
| CPU | `doq.exe metrics`，CPU reference |
| CUDA thread | 一线程处理一份 Snapshot |
| CUDA group4 | 4 线程协作处理一份 Snapshot |
| CUDA group8 | 8 线程协作处理一份 Snapshot |
| CUDA group16 | 16 线程协作处理一份 Snapshot |
| CUDA warp | 32 线程协作处理一份 Snapshot |
| CUDA auto | 根据整批加权工作深度自动选择协作档位 |

性能表重点比较三个交付阶段：

- CPU reference；
- CUDA thread，代表最初的一线程基线；
- CUDA auto，代表加入多档协作和自动选档后的最终版本。

## 3. 完整正确性结果

### 3.1 事件输入

每个事件文件都以 `M=1000` 重放，并检查：

- `resolutions.csv`；
- `states.csv`；
- `snapshots.txt`；
- `K=1` 和 `K=5` 的 CPU 与六种 CUDA 指标输出。

`PASS` 表示文件行数、顺序和每一个整数都与公开标准答案一致。

| 数据集 | 事件数 | M | K | Replay 三类输出 | CPU | CUDA 六种模式 |
| --- | ---: | ---: | --- | --- | --- | --- |
| `events_queue` | 10000 | 1000 | 1、5 | PASS | PASS | 全部 PASS |
| `events_levels` | 10000 | 1000 | 1、5 | PASS | PASS | 全部 PASS |
| `events_fanout` | 10000 | 1000 | 1、5 | PASS | PASS | 全部 PASS |
| `events_churn` | 10000 | 1000 | 1、5 | PASS | PASS | 全部 PASS |
| `events_mixed` | 10000 | 1000 | 1、5 | PASS | PASS | 全部 PASS |

### 3.2 独立 Snapshot 输入

| 数据集 | Snapshot 数 | 最大单侧深度 | 完整测试 K | CPU | CUDA 六种模式 |
| --- | ---: | ---: | --- | --- | --- |
| `snapshots_launch` | 64 | 32 | 1、5、32 | PASS | 全部 PASS |
| `snapshots_shallow` | 1024 | 16 | 1、5、16 | PASS | 全部 PASS |
| `snapshots_medium` | 128 | 256 | 1、64、255、256 | PASS | 全部 PASS |
| `snapshots_deep` | 4 | 16384 | 1、1024、4096、16383、16384 | PASS | 全部 PASS |
| `snapshots_varied` | 128 | 16385 | 5、257、4096、16384 | PASS | 全部 PASS |

事件输入共 10 组 `(数据集, K)`，独立 Snapshot 共 19 组 `(数据集, K)`。29 组参数中的 CPU、thread、group4、group8、group16、warp 和 auto 输出全部通过。

## 4. 原始文本端到端性能

### 4.1 计时方法

这里测量一次完整命令的实际经过时间，包含：

```text
进程启动
→ 文本读取和校验
→ 输入结构构造
→ 指标计算
→ CSV 写出
```

每个配置先预热一次，再测量 7 次并取中位数。CPU、CUDA thread、CUDA auto 按轮换顺序交错执行，减少固定测试顺序带来的偏差。

表中的两个加速比定义为：

```text
CUDA 相对 CPU 加速比 = CPU 时间 / CUDA 时间
auto 相对 thread 加速比 = thread 时间 / auto 时间
```

因此，加速比大于 1 才表示前者更快。

### 4.2 完整结果

| 数据集 | S | K | CPU/ms | CUDA thread/ms | CUDA auto/ms | auto 相对 CPU | auto 相对 thread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `snapshots_launch` | 64 | 1 | 10.114 | 88.810 | 80.636 | 0.125x | 1.101x |
| `snapshots_launch` | 64 | 5 | 9.966 | 84.787 | 87.695 | 0.114x | 0.967x |
| `snapshots_launch` | 64 | 32 | 10.126 | 85.828 | 82.507 | 0.123x | 1.040x |
| `snapshots_shallow` | 1024 | 1 | 23.733 | 96.776 | 99.578 | 0.238x | 0.972x |
| `snapshots_shallow` | 1024 | 5 | 23.222 | 96.738 | 97.624 | 0.238x | 0.991x |
| `snapshots_shallow` | 1024 | 16 | 23.719 | 96.491 | 96.733 | 0.245x | 0.997x |
| `snapshots_medium` | 128 | 1 | 31.455 | 106.257 | 103.155 | 0.305x | 1.030x |
| `snapshots_medium` | 128 | 64 | 30.571 | 105.983 | 111.346 | 0.275x | 0.952x |
| `snapshots_medium` | 128 | 255 | 33.650 | 110.479 | 111.539 | 0.302x | 0.991x |
| `snapshots_medium` | 128 | 256 | 29.819 | 107.492 | 104.244 | 0.286x | 1.031x |
| `snapshots_deep` | 4 | 1 | 51.387 | 128.052 | 128.946 | 0.399x | 0.993x |
| `snapshots_deep` | 4 | 1024 | 51.210 | 127.953 | 126.210 | 0.406x | 1.014x |
| `snapshots_deep` | 4 | 4096 | 56.470 | 135.959 | 132.050 | 0.428x | 1.030x |
| `snapshots_deep` | 4 | 16383 | 53.582 | 132.859 | 126.238 | 0.424x | 1.052x |
| `snapshots_deep` | 4 | 16384 | 52.737 | 129.967 | 127.493 | 0.414x | 1.019x |
| `snapshots_varied` | 128 | 5 | 46.241 | 118.015 | 116.232 | 0.398x | 1.015x |
| `snapshots_varied` | 128 | 257 | 49.535 | 128.624 | 132.177 | 0.375x | 0.973x |
| `snapshots_varied` | 128 | 4096 | 47.192 | 120.365 | 120.266 | 0.392x | 1.001x |
| `snapshots_varied` | 128 | 16384 | 45.423 | 130.315 | 130.514 | 0.348x | 0.998x |

## 5. 结果解释

### 5.1 正确性结论

所有正式模式都与标准答案逐行一致。这说明多线程协作、shuffle 归约、统一 amounts 布局和自动选档只改变执行方式，没有改变业务结果。

### 5.2 为什么公开数据上的 CUDA 端到端更慢

公开 performance 目录中的五个独立快照文件分别只有 4～1024 份 Snapshot，五个文件合计 1348 份，而且每次命令都要重新启动进程和初始化 CUDA Runtime。在这个规模下：

- CPU 处理时间只有约 10～56 ms；
- CUDA 完整命令约为 81～132 ms；
- 文本解析、CUDA 初始化、传输和 CSV 写出占比远大于 kernel 本身。

因此本表中的 CUDA 相对 CPU 加速比只有 `0.114x～0.428x`。这说明对这批小文件，直接使用 CPU 更合适。

### 5.3 为什么 auto 与 thread 差距不明显

19 个测试点中，auto 有 10 个点快于 thread。按 19 个比值计算，auto 相对 thread 的几何平均加速比约为 `1.008x`，只有约 0.8%。这个差距小于部分测试点自身的运行波动，不能据此声称自动分档改善了公开小文件的端到端性能。

自动分档优化的是批量计算阶段。它的收益需要足够多的 Snapshot 或足够长的重复 kernel 才能超过启动、解析和写文件等固定成本。因此：

- 本文公开数据结果用于证明完整格式处理和业务正确性；
- 大规模固定快照实验用于评价 GPU 吞吐、H2D、kernel 和多档协作收益；
- 两种实验测量的范围不同，结论并不矛盾。

## 6. 最终结论

1. 五个事件性能集的 Replay、Snapshot 和 Metrics 全部符合标准答案；
2. 五个独立 Snapshot 性能集的全部 19 个 K 配置均通过；
3. CPU 和六种 CUDA 模式结果完全一致；
4. 原始公开小文件上，CPU 端到端速度更快；
5. 自动分档在这批小文件上的差异被固定开销和运行波动淹没，其价值应以大规模批处理实验判断。
