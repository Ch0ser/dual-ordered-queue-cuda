# 双侧有序队列

本项目实现三部分功能：

1. 按顺序回放事件，维护 A、B 两侧有序队列；
2. 在 CPU 上批量计算 `SnapshotMetrics`；
3. 在 CUDA 上计算相同指标，并与 CPU 结果和性能对比。

下面的命令默认在 `codes` 目录执行：

```powershell
cd E:\P.A.R.A\Project\ZJU_task\codes
```

## 1. 文件结构

```text
codes/
├─ CMakeLists.txt              编译配置
├─ README.md                   使用说明
├─ DESIGN.md                   数据结构和实现设计
├─ src/                        正式源码
│  ├─ main.cpp                 命令行入口
│  ├─ types.hpp                公共数据类型
│  ├─ io.hpp / io.cpp          文件读写和流程组织
│  ├─ order_book.hpp / .cpp    事件回放和订单簿
│  ├─ metrics.hpp / .cpp       CPU 指标计算
│  ├─ metrics_cuda.hpp / .cu   CUDA 指标计算
│  └─ metrics_cuda_stub.cpp    没有 CUDA 时的替代实现
├─ tests/
│  ├─ test_main.cpp            正确性测试
│  └─ data/public/             公开输入和标准答案
├─ bench/                      性能测试程序
├─ bench_data/                 性能数据集和实验结果
├─ build/                      CPU 构建目录
└─ build-cuda129/              CUDA 构建目录
```

## 2. `src` 中每个文件做什么

### `main.cpp`

程序入口，只负责：

1. 读取命令行参数；
2. 判断执行 `replay`、`metrics` 还是 `metrics-cuda`；
3. 调用 `io.cpp` 中相应的高层函数；
4. 捕获错误并返回退出码。

它不实现订单簿和指标计算。

### `types.hpp`

定义整个项目共用的数据：

```text
Event
Resolution
StateSummary
ApplyResult
LevelSnapshot
Snapshot
SnapshotMetrics
Side / Op / ApplyStatus
```

它相当于各模块共同使用的数据格式说明。

### `io.hpp` 和 `io.cpp`

负责输入输出和流程组织：

- 解析 `events.txt` 和 `snapshots.txt`；
- 检查文本格式和事件序号；
- 调用 `OrderBook`、CPU metrics 或 CUDA metrics；
- 写出 CSV 和 Snapshot 文件。

其中 `.hpp` 声明外部可以调用的函数，`.cpp` 实现具体流程。

### `order_book.hpp` 和 `order_book.cpp`

负责阶段一的业务逻辑：

- INSERT；
- REMOVE；
- A、B 两侧按 key 排序；
- 同一 level 内保持 FIFO；
- 生成 `Resolution`、`StateSummary` 和 `Snapshot`。

### `metrics.hpp` 和 `metrics.cpp`

负责 CPU 版本的 `SnapshotMetrics` 计算。它读取 Snapshot，不修改订单簿，也是 CUDA 结果的正确性基准。

### `metrics_cuda.hpp` 和 `metrics_cuda.cu`

负责 CUDA 版本的 `SnapshotMetrics` 计算，包括：

- 连续批数据布局；
- Host 到 Device 传输；
- CUDA kernel；
- Device 到 Host 传输；
- 根据快照深度自动选择 kernel 档位。

### `metrics_cuda_stub.cpp`

CPU-only 构建不会编译 `metrics_cuda.cu`，而是编译这个文件。这样没有 CUDA 的机器仍能运行 `replay` 和 CPU `metrics`；调用 CUDA 功能时会返回明确错误。

## 3. 文件之间怎么调用

```text
main.cpp
   │
   ▼
io.cpp
   ├── replay ─────────► order_book.cpp
   ├── metrics ────────► metrics.cpp
   └── metrics-cuda ───► metrics_cuda.cu

各模块共同使用 types.hpp 中的数据类型
```

三条实际调用链如下。

### 事件回放

```text
main
→ replay_file
→ 逐行解析 Event
→ OrderBook::apply
→ 写 resolutions.csv 和 states.csv
→ 按间隔 M 写 snapshots.txt
```

### CPU 指标计算

```text
main
→ compute_metrics_file
→ 读取 Snapshot
→ compute_snapshot_metrics
→ 写 metrics.csv
```

### CUDA 指标计算

```text
main
→ compute_metrics_cuda_file
→ 分批读取 Snapshot
→ H2D
→ CUDA kernel
→ D2H
→ 写 metrics_cuda.csv
```

完整数据流是：

```text
events.txt
    │
    │ replay
    ▼
resolutions.csv + states.csv + snapshots.txt
                                      │
                                      ├─ metrics      → CPU 结果
                                      └─ metrics-cuda → CUDA 结果
```

`OrderBook` 和 metrics 不直接调用。阶段一导出 Snapshot，阶段二再把 Snapshot 当作独立输入。

## 4. 编译

### 4.1 编译 CUDA 版本

本机使用 Visual Studio 2022、CUDA 12.9 和 RTX 4050：

```powershell
cmake -S . -B build-cuda129 `
  -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_CUDA_ARCHITECTURES=89

cmake --build build-cuda129 --config Release -j
ctest --test-dir build-cuda129 -C Release --output-on-failure
```

主要程序位于：

```text
build-cuda129/Release/doq.exe
build-cuda129/Release/doq_tests.exe
build-cuda129/Release/doq_benchmark.exe
build-cuda129/Release/doq_metrics_benchmark.exe
```

修改源码后通常只需重新编译和测试：

```powershell
cmake --build build-cuda129 --config Release -j
ctest --test-dir build-cuda129 -C Release --output-on-failure
```

### 4.2 只编译 CPU 版本

```powershell
cmake -S . -B build -G "MinGW Makefiles" `
  -DDOQ_ENABLE_CUDA=OFF

cmake --build build -j
ctest --test-dir build --output-on-failure
```

## 5. 常用用法

最终程序提供三个命令：

| 命令 | 输入 | 输出 |
|---|---|---|
| `replay` | `events.txt` | `resolutions.csv`、`states.csv`、`snapshots.txt` |
| `metrics` | `snapshots.txt` | CPU `metrics.csv` |
| `metrics-cuda` | `snapshots.txt` | CUDA `metrics.csv` |

### 5.1 `replay`：回放事件

命令格式：

```text
doq.exe replay <events.txt> <M> <output_directory>
```

示例：

```powershell
.\build-cuda129\Release\doq.exe replay .\tests\data\public\events\events_example.txt 1 .\out_m1
```

`M` 是快照间隔。`M=1` 表示每处理完一个事件就生成一次 Snapshot。

事件文件格式：

```text
N
I seq item_id side key amount
R seq item_id
```

示例：

```text
6
I 1 101 A 100 10
I 2 102 A 100 5
I 3 201 B 99 12
R 4 102
I 5 301 B 105 4
I 6 401 A 106 2
```

输出目录中会生成：

| 文件 | 内容 |
|---|---|
| `resolutions.csv` | 实际发生处理的 current item、resident item、key 和 amount |
| `states.csv` | 每个事件结束后的队列摘要和处理状态 |
| `snapshots.txt` | 按间隔 M 保存的 level 聚合状态 |

三个文件的格式如下。

`resolutions.csv` 无表头，每次实际处理输出一行：

```text
event_seq,current_item_id,resident_item_id,key,amount
```

`states.csv` 无表头，每个输入事件固定输出一行：

```text
event_seq,a_extreme_key,b_extreme_key,a_amount,b_amount,active_count,status
```

`snapshots.txt` 第一行是快照数量 `S`。每份快照先写一行头部，再写 A、B 两侧的 level：

```text
S

event_seq NA NB
接下来 NA 行：key amount    # A 侧，key 从大到小
接下来 NB 行：key amount    # B 侧，key 从小到大
```

`NA`、`NB` 是两侧的 level 数量。Snapshot 只保留每个 level 的 key 和 amount 总和，不保存 `item_id` 和同 level 内的 FIFO 明细。

`M` 只控制 Snapshot 频率，不改变事件执行过程，也不改变 `states.csv` 和 `resolutions.csv`。

### 5.2 `metrics`：CPU 计算指标

命令格式：

```text
doq.exe metrics <snapshots.txt> <K> <metrics.csv>
```

示例：

```powershell
.\build-cuda129\Release\doq.exe metrics .\out_m1\snapshots.txt 2 .\out_m1\metrics_cpu.csv
```

`K` 表示每侧最多统计前多少个 level，合法范围为 `1～16384`。它统计的是 level，不是 item。

输出每行包含：

```text
event_seq,
a_extreme_key,b_extreme_key,
a_amount_top_k,b_amount_top_k,
key_gap,
balance_num,balance_den,
weighted_key_num,weighted_key_den
```

### 5.3 `metrics-cuda`：CUDA 计算指标

命令格式：

```text
doq.exe metrics-cuda <snapshots.txt> <K> <metrics.csv> [batch_size] [kernel_mode]
```

示例：

```powershell
.\build-cuda129\Release\doq.exe metrics-cuda .\out_m1\snapshots.txt 2 .\out_m1\metrics_cuda.csv 65536 auto
```

可选参数：

| 参数 | 含义 |
|---|---|
| `batch_size` | 每批送入 CUDA 的 Snapshot 数量；省略时自动计算 |
| `kernel_mode` | `auto`、`thread`、`group4`、`group8`、`group16` 或 `warp` |

`batch_size` 和 kernel 模式只影响执行方式和性能，不应改变结果。

## 6. 一次跑通完整流程

```powershell
$exe = '.\build-cuda129\Release\doq.exe'

& $exe replay `
  .\tests\data\public\events\events_example.txt 3 `
  .\out\quickstart

& $exe metrics `
  .\out\quickstart\snapshots.txt 1 `
  .\out\quickstart\metrics_cpu.csv

& $exe metrics-cuda `
  .\out\quickstart\snapshots.txt 1 `
  .\out\quickstart\metrics_cuda.csv 65536 auto
```

比较 CPU 和 CUDA 输出：

```powershell
Compare-Object `
  (Get-Content .\out\quickstart\metrics_cpu.csv) `
  (Get-Content .\out\quickstart\metrics_cuda.csv)
```

没有输出表示逐行相同。

## 7. 与公开标准答案比较

下面使用 `events_example.txt`、`M=3`、`K=1`：

```powershell
$expected = '.\tests\data\public\expected\events_example'
$actual = '.\out\quickstart'

Compare-Object `
  (Get-Content "$expected\resolutions.csv") `
  (Get-Content "$actual\resolutions.csv")

Compare-Object `
  (Get-Content "$expected\states.csv") `
  (Get-Content "$actual\states.csv")

Compare-Object `
  (Get-Content "$expected\snapshots.m3.txt") `
  (Get-Content "$actual\snapshots.txt")

Compare-Object `
  (Get-Content "$expected\metrics.m3.k1.csv") `
  (Get-Content "$actual\metrics_cpu.csv")
```

四次比较都没有输出，表示结果正确。

## 8. 测试

CUDA 构建：

```powershell
ctest --test-dir build-cuda129 -C Release --output-on-failure
```

CPU 构建：

```powershell
ctest --test-dir build --output-on-failure
```

公开测试数据位于：

```text
tests/data/public/events/        事件输入
tests/data/public/snapshots/     独立快照输入
tests/data/public/expected/      标准答案
tests/data/public/performance/   较大的固定数据
```

## 9. 常用性能实验

CPU/CUDA 对比程序的格式：

```text
doq_metrics_benchmark
  <dataset_path>
  <snapshot_count>
  <nominal_depth>
  <K>
  <batch_size>
  <repetitions>
  <nested|preflattened>
  [auto|thread|group4|group8|group16|warp]
  [serial|overlap]
```

日常先运行50万 Snapshot：

```powershell
.\build-cuda129\Release\doq_metrics_benchmark.exe `
  .\bench_data\datasets\snapshots_500k_d50.bin `
  500000 50 50 65536 3 preflattened auto serial
```

主要关注：

```text
cpu_seconds_per_pass
cuda_seconds_per_pass
cuda_h2d_seconds_per_pass
cuda_kernel_seconds_per_pass
cuda_speedup_vs_cpu
cpu_cuda_match=yes
```

性能实验的完整过程、1m/2m 结果和分档临界点见 `../docs/EXPERIMENTS.md`。

## 10. 常用参数速查

| 参数 | 所属命令 | 含义 |
|---|---|---|
| `M` | `replay` | 每处理多少个事件生成一次 Snapshot |
| `K` | `metrics`、`metrics-cuda` | 每侧最多统计前多少个 level |
| `batch_size` | `metrics-cuda` | 一批处理多少份 Snapshot |
| `kernel_mode` | `metrics-cuda` | 每份 Snapshot 使用多少线程协作 |

退出码：

| 退出码 | 含义 |
|---:|---|
| `0` | 成功 |
| `1` | 参数内容、文件、数据或运行过程出错 |
| `2` | 命令名称或参数数量错误 |

## 11. 进一步阅读

- `DESIGN.md`：数据结构、复杂度和 CPU/CUDA 设计；
- `../docs/candidate_task.md`：原始任务要求；
- `../docs/EXPERIMENTS.md`：优化实验和最终性能；
- `tests/data/public/README.md`：公开数据规模和对应标准答案。
