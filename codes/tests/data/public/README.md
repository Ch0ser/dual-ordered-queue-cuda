# 测试数据说明

本目录包含公开测试输入及对应的标准输出。操作规则和字段含义见 `candidate_task.md`。

## 文件说明

| 目录 | 内容 |
| --- | --- |
| `events/` | 事件输入，每个文件从空状态开始重放 |
| `snapshots/` | 独立快照输入，可直接计算指标 |
| `expected/` | 上述输入对应的标准输出 |
| `performance/` | 性能测试输入及标准输出，内部按 `events/`、`snapshots/`、`expected/` 划分 |

事件文件第一行为事件数 `N`；快照文件第一行为快照数 `S`。其余格式与题面一致。

## 参数与答案文件

`M` 为快照生成间隔，`K` 为指标计算深度。两者均为运行时参数，不写入输入文件。下表列出的每个 `M`、`K` 均有对应答案；事件数据中的 `M`、`K` 可交叉组合。

输入文件名去掉 `.txt` 后记为 `<name>`，其答案位于 `expected/<name>/`。`performance/` 下的输入对应 `performance/expected/<name>/`。

| 输入类型 | 输出内容 | 答案文件名 |
| --- | --- | --- |
| 事件 | Resolution | `resolutions.csv` |
| 事件 | StateSummary | `states.csv` |
| 事件 | 间隔为 M 的快照 | `snapshots.m{M}.txt` |
| 事件 | 上述快照在深度 K 下的指标 | `metrics.m{M}.k{K}.csv` |
| 独立快照 | 深度 K 下的指标 | `metrics.k{K}.csv` |

例如，`events/events_example.txt` 在 `M=3`、`K=1` 时，对应 `expected/events_example/snapshots.m3.txt` 和 `expected/events_example/metrics.m3.k1.csv`。

CSV 文件无表头，整数精确比较，记录顺序与题面一致。没有 Resolution 时，对应答案为空文件；双侧为空的快照仍保留 `event_seq 0 0` 记录。

## 数据列表

### 基础数据：事件

| 输入文件 | N | M | K |
| --- | ---: | --- | --- |
| `events/events_example.txt` | 6 | 1, 3, 4, 10 | 1, 5 |
| `events/events_minimal_a.txt` | 1 | 1, 2 | 1, 5 |
| `events/events_minimal_b.txt` | 1 | 1, 2 | 1, 5 |
| `events/events_minimal_remove.txt` | 1 | 1, 2 | 1, 5 |
| `events/events_minimal_invalid.txt` | 1 | 1, 2 | 1, 5 |
| `events/events_priority_a.txt` | 6 | 1, 7 | 1, 5 |
| `events/events_priority_b.txt` | 6 | 1 | 1, 5 |
| `events/events_fifo_a.txt` | 5 | 1, 6 | 1, 5 |
| `events/events_fifo_b.txt` | 5 | 1 | 1, 5 |
| `events/events_levels_a.txt` | 6 | 1, 7 | 1, 5 |
| `events/events_levels_b.txt` | 6 | 1 | 1, 5 |
| `events/events_amount_cases_a.txt` | 7 | 1, 8 | 1, 5 |
| `events/events_amount_cases_b.txt` | 7 | 1 | 1, 5 |
| `events/events_remove_a.txt` | 12 | 1, 13 | 1, 5 |
| `events/events_remove_b.txt` | 12 | 1 | 1, 5 |
| `events/events_id_lifecycle.txt` | 11 | 1 | 1, 5 |
| `events/events_invalid.txt` | 10 | 1 | 1, 5 |
| `events/events_snapshot_timing.txt` | 10 | 1, 3, 5, 20 | 1, 5 |
| `events/events_integer.txt` | 6 | 1 | 1, 5 |
| `events/events_mixed.txt` | 240 | 7, 31, 241 | 1, 5 |

### 基础数据：快照

| 输入文件 | S | 最大单侧 depth | K |
| --- | ---: | ---: | --- |
| `snapshots/snapshots_empty.txt` | 4 | 3 | 1, 5 |
| `snapshots/snapshots_small.txt` | 5 | 6 | 1, 3, 5, 10 |
| `snapshots/snapshots_metrics.txt` | 3 | 2 | 1, 2, 5 |
| `snapshots/snapshots_varied.txt` | 7 | 1025 | 1, 5, 257, 1025 |
| `snapshots/snapshots_boundaries.txt` | 9 | 1025 | 31, 32, 33, 255, 256, 257, 1023, 1024, 1025 |
| `snapshots/snapshots_deep.txt` | 3 | 16385 | 1025, 4096, 16383, 16384 |
| `snapshots/snapshots_integer.txt` | 2 | 1 | 1, 5 |
| `snapshots/snapshots_batch_1.txt` | 1 | 0 | 1, 5 |
| `snapshots/snapshots_batch_31.txt` | 31 | 6 | 1, 5 |
| `snapshots/snapshots_batch_32.txt` | 32 | 6 | 1, 5 |
| `snapshots/snapshots_batch_33.txt` | 33 | 6 | 1, 5 |
| `snapshots/snapshots_batch_257.txt` | 257 | 6 | 1, 5 |

### 性能数据：事件

| 输入文件 | N | M | K |
| --- | ---: | --- | --- |
| `performance/events/events_queue.txt` | 10000 | 1000 | 1, 5 |
| `performance/events/events_levels.txt` | 10000 | 1000 | 1, 5 |
| `performance/events/events_fanout.txt` | 10000 | 1000 | 1, 5 |
| `performance/events/events_churn.txt` | 10000 | 1000 | 1, 5 |
| `performance/events/events_mixed.txt` | 10000 | 1000 | 1, 5 |

### 性能数据：快照

| 输入文件 | S | 最大单侧 depth | K |
| --- | ---: | ---: | --- |
| `performance/snapshots/snapshots_launch.txt` | 64 | 32 | 1, 5, 32 |
| `performance/snapshots/snapshots_shallow.txt` | 1024 | 16 | 1, 5, 16 |
| `performance/snapshots/snapshots_medium.txt` | 128 | 256 | 1, 64, 255, 256 |
| `performance/snapshots/snapshots_deep.txt` | 4 | 16384 | 1, 1024, 4096, 16383, 16384 |
| `performance/snapshots/snapshots_varied.txt` | 128 | 16385 | 5, 257, 4096, 16384 |

所有输入均满足题面的数据范围。补充测试数据及其生成程序由候选人自行设计。
