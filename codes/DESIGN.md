# 双侧有序队列与 CUDA SnapshotMetrics 设计

## 完成范围

当前版本完成任务的三个必做阶段：

1. CPU 事件重放、Resolution、StateSummary、Snapshot 导出和 CPU SnapshotMetrics；
2. CUDA 批量 SnapshotMetrics，并用 CPU reference 逐字段验证；
3. 对数据布局、内存申请、pinned memory、kernel 并行粒度、H2D 次数、batch size 和 stream overlap 进行测量、优化或否决。

可选的 `MODIFY` 没有实现。完整事件重放具有前后状态依赖，仍保留在 CPU；GPU 目标选择相互独立的大量 Snapshot。

## CPU 队列设计

`OrderBook` 使用四类相互配合的容器：

- A 侧：`std::map<int64_t, Level, std::greater<>>`，`begin()` 为最大 key；
- B 侧：`std::map<int64_t, Level, std::less<>>`，`begin()` 为最小 key；
- level 内：`std::list<Item>`，支持队首处理、队尾插入以及通过迭代器删除队中元素；
- 活跃索引：`unordered_map<item_id, ActiveLocation>`，保存 side、key 和 list iterator；
- 永久 ID 集合：`unordered_set<item_id>`，记录每个成功 INSERT 使用过的编号。

每个 level 同时维护 `total_amount`，因此 StateSummary 可以在 `O(1)` 时间读取两侧极值 level 的总量。

INSERT 严格按任务书顺序判断非法参数和重复 ID。当前 item 循环处理另一侧极值 level 的队首，resident 归零时同步删除活跃索引，level 变空时删除 map 节点；当前 item 最后仍有剩余才进入本侧队尾。REMOVE 先通过活跃索引定位 list iterator，再更新 level 总量、list、活跃索引和空 level，永久 ID 集合不回收编号。

测试接口 `validate_invariants` 会完整检查两侧排序边界、FIFO 顺序、正 amount、活跃 ID 唯一性以及索引指向。

## 流式文件处理

事件和 Snapshot 都采用流式解析，不要求把最多一千万条事件或所有快照一次装入内存。每个事件按下面的次序处理：

```text
apply -> 写 Resolution -> 写 StateSummary -> 判断并写 Snapshot
```

非法事件也计入快照间隔。输入首行给出事件数，因此 Snapshot 数量可预先计算为 `ceil(N/M)`。CPU `metrics` 逐批读取 Snapshot、调用无状态 reference 函数并写出 CSV。

## 为什么选择 SnapshotMetrics 上 GPU

事件重放会修改共享队列，后一个事件依赖前一个事件的状态；强行并行会引入复杂的冲突检测与同步。不同 SnapshotMetrics 之间没有状态依赖，输入只读、每份输出固定，因此天然适合批量并行。

一个 GPU 工作单元的语义始终是一份 Snapshot：读取两侧前 `min(K, depth)` 个 level amount 和两侧 extreme key，输出一条 SnapshotMetrics。输出数组与 descriptor 下标一致，D2H 后天然保持输入顺序。

## Device 输入布局

正式 `metrics-cuda` 在解析和校验文本时直接构造两块连续的 pinned Host 输入：

1. descriptor 数组：`event_seq`、两侧 `offset/count` 和 extreme key；
2. 统一 amounts 数组：每份 Snapshot 的 A 前 K 档后紧接 B 前 K 档。

A/B 各自的 offset 都相对统一 amounts 数组。可变 depth 用 `offset + count` 表示，不补零，短 Snapshot 不浪费传输和显存。Device workspace 对应保存 descriptor、统一 amounts 和输出三块 buffer；容量不足时增长，后续 batch 直接复用。

第一版 CUDA 路径先构造嵌套 `vector<Snapshot>`，随后再压平，并在每批中反复申请和释放 Device 内存。最终路径依次完成了：

- 读取文本时直接形成连续布局；
- Host 输入改为 `cudaMallocHost` pinned buffer 并复用容量；
- Device workspace 复用，取消每批固定的 `cudaMalloc/cudaFree`；
- A/B amounts 合并，输入由三次 H2D 减为两次；
- 默认 batch 按约 256 MiB 最坏情况输入预算计算，并封顶 262,144。

一次把 descriptor 和 amounts 全部打包进 arena 没有稳定快于两次复制，因此没有引入额外的对齐和指针重定位逻辑。

## Kernel 并行方式

所有 kernel 使用 256 threads/block，并在网格超过上限时使用 grid-stride loop。

保留的 `thread` 基线是一线程处理一份 Snapshot，线程内部顺序累加 A/B。这一模式等价于自动分档接入正式命令前的 kernel，可通过下面的命令复现：

```powershell
doq metrics-cuda snapshots.txt 50 metrics_thread.csv 262144 thread
```

最终实现另外提供 4、8、16、32 个线程协作处理一份 Snapshot。每个 lane 跨步读取 amount，并使用 `__shfl_down_sync` 在子 warp 内归约；lane 0 计算派生字段并写出结果。它不需要为每份 Snapshot 分配 Shared Memory。

正式命令默认使用 `auto`。每批计算：

```text
work_i = a_count_i + b_count_i
weighted_work = sum(work_i * work_i) / sum(work_i)
```

然后选择：

| weighted_work | 模式 |
|---:|---|
| `<= 96` | group4 |
| `97～448` | group8 |
| `449～1023` | group16 |
| `>= 1024` | group32/warp |

工作量加权值等价于“随机抽取一次 level 访问时，其所属 Snapshot 的期望工作量”。它避免大量浅快照把少量但计算占比很高的深快照稀释。阈值来自 shallow、medium、deep、varied、A/B 不对称和深浅混合数据的实测；它们是当前 RTX 4050 Laptop GPU 上的经验选择，不是跨硬件常数。

`metrics-cuda` 也允许显式指定 `thread/group4/group8/group16/warp`，用于复现实验和排查阈值。

## 没有采用的方案

### GPU 直接解析文本

实验路径把原始 ASCII Snapshot 传入 GPU 并在 kernel 中解析十进制整数。在输入已经准备好的阶段，ASCII 体积更大且整数解析昂贵，CUDA 时间约为连续布局的 1.79 倍。实验尚未实现与正式 parser 完全相同的校验，因此没有替换生产路径。

这不等于 GPU 解析永远无效：当前正式端到端时间主要在 CPU 文本解析和 CSV 序列化。如果上游只能给 ASCII，融合换行定位、校验、解析和计算仍可能有收益；如果上游能直接给连续二进制，则应跳过文本。

### H2D/kernel overlap

实验实现了两条 non-blocking stream、两套 Device workspace 和双 pinned 输出 buffer。50 万快照连续提交 20 次时，真正隐藏的时间只有约 `0.057 ms/pass`，占 H2D、kernel、D2H 活动和的 `0.16%`，因此正式路径继续使用更简单的串行 batch。

本机 `asyncEngineCount=1`；H2D 与 Snapshot kernel 都大量访问显存，并发后的资源争用基本抵消时间重叠。

### 按深度分桶

把混合 batch 拆成多个 kernel 会增加启动次数，小批次利用率下降。实测多数比最佳单 kernel 慢，因此只在整批上选择一个加权档位。

### 数据常驻显存后重复计算

任务书必做路径一次运行只有一个 K 和一组固定指标，没有真实的第二次计算可复用输入。benchmark 的 repetitions 只是稳定计时，不能当作业务收益，因此没有为不存在的重复查询扩大正式接口。

## 正确性验证

- 端到端测试逐字节核对公开示例的 states、resolutions、snapshots 和 metrics；
- 单元测试覆盖同 level FIFO、跨 level、部分处理、队首/队中/队尾删除、非法事件、永久 ID、快照间隔和空侧指标；
- CUDA 测试覆盖 `K=1/3/16384`、batch `1/2/64`、`auto` 和旧 `thread`；
- 性能实验每轮都逐字段比较 CPU/CUDA，并记录 checksum；
- 没有 CUDA 时仍可构建和运行 replay、CPU metrics，CUDA 命令会给出明确错误。

## 复杂度

设当前 level 数为 `L`，一次 INSERT 实际处理 `R` 个 resident item：

- INSERT：`O((R + 被跨越的 level 数) log L)`；
- REMOVE：活跃 ID 定位均摊 `O(1)`，level 定位和删除 `O(log L)`；
- StateSummary：`O(1)`；
- Snapshot 导出：`O(L)`；
- 单份 SnapshotMetrics：`O(min(K,A_depth)+min(K,B_depth))`；
- 所有 SnapshotMetrics 总工作量不因 GPU 改变，只改变并行调度。

## 最终性能实验

环境：Intel Core i5-13500HX、NVIDIA GeForce RTX 4050 Laptop GPU（6141 MiB）、驱动 592.82、CUDA Toolkit 12.9.86、Windows、Visual Studio 2022 Release、C++17、`sm_89`、256 threads/block。

固定种子 `20260927`，名义单侧深度 50，`K=50`，batch 262,144。每个规模对 `thread/auto` 各做三次独立实验，每次内部 3 passes，并交替模式顺序。下表各字段取三次中位数；加速比取每次 `CPU/CUDA` 后的中位数。

| Snapshot 数 | 模式 | CPU / pass | CUDA / pass | H2D | kernel | D2H | CPU/CUDA |
|---:|---|---:|---:|---:|---:|---:|---:|
| 500,000 | thread | 135.309 ms | 46.237 ms | 28.095 ms | 5.282 ms | 5.294 ms | 2.842x |
| 500,000 | auto | 138.093 ms | 45.310 ms | 28.088 ms | 2.437 ms | 5.407 ms | 3.083x |
| 1,000,000 | thread | 325.417 ms | 283.113 ms | 181.392 ms | 50.400 ms | 24.168 ms | 1.149x |
| 1,000,000 | auto | 313.272 ms | 279.366 ms | 188.875 ms | 46.140 ms | 24.540 ms | 1.057x |
| 2,000,000 | thread | 537.262 ms | 637.468 ms | 429.311 ms | 110.336 ms | 51.262 ms | 0.866x |
| 2,000,000 | auto | 508.523 ms | 620.501 ms | 420.413 ms | 105.503 ms | 50.254 ms | 0.805x |

`auto` 相对 `thread` 的 kernel 中位时间分别下降约 53.9%、8.5% 和 4.4%，完整 CUDA 中位时间下降约 2.0%、1.3% 和 2.7%。所有输出一致，checksum 依次为：

```text
500,000    11174276585170874061
1,000,000  13908833524329689336
2,000,000  1926215580095944893
```

H2D 存在明显运行间波动，特别是 100 万和 200 万规模，因此不能用非配对的单次绝对值判断小优化。较大规模中 H2D 占主导，kernel 加速只能影响总时间的一小部分。

第一版 50 万 CUDA 完整计算路径为 `0.353 s/pass`，最终 auto 中位数为 `0.045310 s/pass`，约缩短到原来的 `12.8%`。第一版包含嵌套数据二次压平、重复 Device 分配和 pageable Host 输入；这项历史对比用于展示优化过程，最终 `thread/auto` 配对才用于隔离 kernel 分档收益。

### 正式文本到 CSV 的端到端时间

对 65,536 份、每侧深度 50、`K=50` 的 74.26 MiB Snapshot 文本，三种命令各预热一次并交替测三次：

| 路径 | 端到端中位数 |
|---|---:|
| CPU `metrics` | 2.335 s |
| CUDA `thread` | 2.393 s |
| CUDA `auto` | 2.432 s |

三份 CSV 的 SHA-256 完全相同。这个端到端范围包含文本解析、输入校验、计算和 CSV 序列化。它说明当前文本接口主要受共同的 CPU 解析和输出限制；prepared-data benchmark 中 GPU 计算更快，并不保证原始文本到文本输出也更快。

完整实验过程和失败方案见 `../docs/EXPERIMENTS.md`。

## 已知限制与下一步

- 自动阈值来自当前 GPU，需要在其他架构上重新标定；
- 2m 实验的 H2D 波动较大，最好结合运行期频率、功耗和 P-state 解释；
- 正式文本路径尚未并行化解析与 CSV 序列化；
- 没有实现可选 `MODIFY`、CUDA Graph 和大于 16384 的 K；
- 输出使用普通文件流，尚未专门优化一千万事件的磁盘吞吐。

如果继续优化，应优先改变输入输出边界，例如让上游提供连续二进制、并行解析文本或减少 CSV 序列化成本，而不是继续微调只占总时间少部分的 kernel。

## 参考资料

- NVIDIA, *CUDA C++ Programming Guide*：线程层级、warp shuffle、内存层级和并发执行；
- NVIDIA, *CUDA Runtime API*：pinned Host memory、stream、event 和异步复制；
- C++17 标准库容器语义：`map`、`list`、`unordered_map` 和 `unordered_set`；
- 任务书 `../docs/candidate_task.md`：业务规则、输出定义和评测范围。
