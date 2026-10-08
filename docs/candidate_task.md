# 双侧有序队列与异构批量优化

## 1. 任务描述

实现一个可重放的双侧有序队列，并在功能正确的基础上选择合适的批量计算环节进行 CUDA 优化。

任务按以下三个阶段完成：

1. **完整功能**：实现事件重放、状态输出、快照导出和 CPU 批量计算，并通过正确性测试；
2. **GPU 批量计算**：根据数据依赖和性能测量，自行选择适合并行的环节，提供 CUDA 实现；
3. **极致优化**：以第一版正确实现为基线，持续 profiling、优化和复测，给出最终性能结果及限制。

请基于 C++17 和 CUDA Toolkit 12.x+ 实现。

## 2. 数据结构

双侧有序队列由 A、B 两侧组成。每次事件重放开始时，两侧均为空，尚无已使用的 `item_id`。

```text
Side = A | B
Op   = INSERT | REMOVE

Event {
    uint64_t seq;          // 全局唯一，严格递增
    uint64_t item_id;
    Side side;             // REMOVE 时不使用
    Op op;
    int64_t key;           // INSERT 时使用
    int64_t amount;        // INSERT 时使用
}
```

每个活跃 item 包含 `item_id`、`side`、`key`、剩余 `amount` 和插入时的 `seq`。

每一侧由若干个 level 组成：

- A 侧按 `key` 从大到小排列；
- B 侧按 `key` 从小到大排列；
- 相同 `key` 的 item 位于同一个 level；
- 同一 level 内按插入顺序排列，先插入的 item 先被处理。

`item_id` 是元素的唯一标识，用于 `REMOVE` 和 `Resolution` 引用，不参与 level 排序或 FIFO 排序。`key` 决定元素所在的 level，插入时的 `seq` 决定同一 level 内的先后顺序。一个成功使用过的 `item_id` 不能再次用于 `INSERT`，即使对应元素已经被删除或处理完。

## 3. 操作规则

### 3.1 `INSERT`

`INSERT` 创建一个当前 item `x`。`x` 先与另一侧已有 item 进行处理，剩余部分再进入本侧队列。

当 `x.side = A` 时：

1. 若 B 侧为空，停止处理；否则取 B 侧最小 `key` 的 level；
2. 若 `x.key < B.min_key`，停止处理；
3. 取该 level 的队首 item `y`；
4. 令 `v = min(x.amount, y.amount)`；
5. 生成一条 `Resolution`，其 `key` 使用 `y.key`，`amount` 为 `v`；
6. 分别从 `x.amount` 和 `y.amount` 中减去 `v`；
7. `y.amount` 变为 0 时，从 level 和 item 索引中删除 `y`；若该 level 变空，同时删除该 level；
8. `x.amount` 仍大于 0 时，从步骤 1 继续。

当 `x.side = B` 时：

1. 若 A 侧为空，停止处理；否则取 A 侧最大 `key` 的 level；
2. 若 `x.key > A.max_key`，停止处理；
3. 取该 level 的队首 item `y`；
4. 令 `v = min(x.amount, y.amount)`；
5. 生成一条 `Resolution`，其 `key` 使用 `y.key`，`amount` 为 `v`；
6. 分别从 `x.amount` 和 `y.amount` 中减去 `v`；
7. `y.amount` 变为 0 时，从 level 和 item 索引中删除 `y`；若该 level 变空，同时删除该 level；
8. `x.amount` 仍大于 0 时，从步骤 1 继续。

处理结束后，如果 `x.amount > 0`，将 `x` 放入本侧对应 level 的队尾。一个 `INSERT` 可以连续处理同一 level 内的多个 item，也可以跨越多个 level。

### 3.2 `REMOVE`

`REMOVE` 根据 `item_id` 查找活跃 item，并删除其全部剩余 amount。只有 item 仍然活跃时，删除才成功。

删除可能发生在 level 的队首、队中或队尾。删除后，同一 level 中其他 item 的相对顺序不得改变。level 为空时，应同时删除该 level。

### 3.3 非法事件

以下事件不得改变任何状态：

- `INSERT` 的 `key <= 0` 或 `amount <= 0`；
- `INSERT` 使用已经成功使用过的 `item_id`；
- `REMOVE` 指向不存在、已经删除或已经处理完的 item。

状态码按以下顺序判定：

1. `INSERT` 参数非法：`REJECTED_INVALID`；
2. `INSERT` 的 `item_id` 已经使用：`REJECTED_DUPLICATE`；
3. `REMOVE` 找不到活跃 item：`REJECTED_NOT_FOUND`；
4. 其余情况：`OK`。

非法 `INSERT` 不占用 `item_id`。成功插入但在当前事件中被全部处理的 item 仍视为已经使用过该 `item_id`。

### 3.4 状态约束

每个事件处理完成后，状态必须满足：

- A 侧和 B 侧同时非空时，`A.max_key < B.min_key`；
- 同一 level 内的 item 保持插入顺序；
- 活跃 item 的剩余 amount 大于 0；
- 每个活跃 `item_id` 只出现在一个 side 和一个 level 中。

## 4. 输出定义

### 4.1 事件处理结果

每个事件返回一个 `ApplyResult`。字段名称可以不同，但含义必须一致。

```text
ApplyStatus = OK
            | REJECTED_INVALID
            | REJECTED_DUPLICATE
            | REJECTED_NOT_FOUND

Resolution {
    uint64_t event_seq;
    uint64_t current_item_id;  // 当前 INSERT 创建的 item
    uint64_t resident_item_id; // 另一侧队列中已有的 item
    int64_t key;               // resident item 的 key
    int64_t amount;
}

StateSummary {
    uint64_t event_seq;
    int64_t a_extreme_key;     // A 侧最大 key；为空时为 -1
    int64_t b_extreme_key;     // B 侧最小 key；为空时为 -1
    int64_t a_amount;          // A 侧最大 key level 的 amount 总和
    int64_t b_amount;          // B 侧最小 key level 的 amount 总和
    uint64_t active_count;     // 当前活跃 item 数量
}

ApplyResult {
    ApplyStatus status;
    Resolution resolutions[];
    StateSummary state;         // 当前事件全部处理完成后的状态
}
```

`StateSummary` 表示当前事件的最终状态，生成时点位于该事件的所有 `Resolution` 之后。对于 `INSERT`，应先完成全部 amount 扣减、删除 amount 为 0 的 item 和空 level，并将当前 item 的剩余部分放入队列，再生成 `StateSummary`。不输出每条 `Resolution` 之间的中间状态。

同一个 `INSERT` 产生多条 `Resolution` 时，按实际处理顺序输出。事件之间按输入顺序输出。非法事件不产生 `Resolution`，其 `StateSummary` 与事件处理前相同，但 `event_seq` 使用当前非法事件的 `seq`。

某一侧为空时，`StateSummary` 中该侧的 extreme key 为 `-1`，amount 为 `0`。`active_count` 为两侧活跃 item 的总数；两侧均为空时为 `0`。

### 4.2 Snapshot 生成时点

`Snapshot` 在一个事件**全部处理完成后**生成。单个事件的顺序为：

```text
读取 Event
-> 按第 3 章执行 INSERT 或 REMOVE
   （处理过程中逐条生成 Resolution，并完成 amount 更新、删除和入队）
-> 生成 StateSummary
-> 按快照间隔 M 判断是否生成 Snapshot
```

快照间隔 `M` 由运行时指定，按已处理的输入事件数量计数，规则如下：

- 处理完第 `M`、`2M`、`3M` ... 个输入事件后生成快照；
- 非法事件也计入输入事件数量，可能生成一份状态未变化的快照；
- 快照的 `event_seq` 等于触发该快照的事件 `seq`；
- 最后一个事件不在间隔边界时，仍额外生成一份最终快照；
- 最后一个事件正好位于间隔边界时，不重复生成；
- 不生成任何事件之前的初始空状态快照；
- `M` 必须大于 0。

例如，共处理 10 个输入事件：

```text
M = 3  -> 在第 3、6、9、10 个事件后生成
M = 5  -> 在第 5、10 个事件后生成
M = 20 -> 只在第 10 个事件后生成
```

### 4.3 Snapshot 数据结构

状态快照使用聚合后的 level，不包含 level 内的单个 item：

```text
Snapshot {
    uint64_t event_seq;
    Level a_levels[]; // key 从大到小
    Level b_levels[]; // key 从小到大
}

Level {
    int64_t key;
    int64_t amount;   // 当前 level 内所有 item 的剩余 amount 总和
}
```

`a_levels` 和 `b_levels` 的长度分别称为该快照 A、B 两侧的 depth。depth 表示 level 数量，不表示 item 数量。

每份快照包含当时两侧的全部非空 level，不按后续计算使用的 `K` 截断。

### 4.4 SnapshotMetrics 批量计算

给定快照和计算深度 `K`，计算以下指标。`K` 为 `1` 到 `16384` 之间的整数，不要求是 2 的幂。两侧分别取前 `min(K, 本侧 depth)` 个 level；某一侧为空时，该侧的 amount 总和为 `0`。

```text
SnapshotMetrics {
    uint64_t event_seq;
    int64_t a_extreme_key;  // A 侧最大 key；为空时为 -1
    int64_t b_extreme_key;  // B 侧最小 key；为空时为 -1
    int64_t a_amount_top_k; // A 侧前 min(K, depth) 个 level 的 amount 总和
    int64_t b_amount_top_k; // B 侧前 min(K, depth) 个 level 的 amount 总和
    int64_t key_gap;
    int64_t balance_num;
    int64_t balance_den;
    int64_t weighted_key_num;
    int64_t weighted_key_den;
}
```

各字段按以下规则计算：

```text
key_gap = b_extreme_key - a_extreme_key
balance_num = a_amount_top_k - b_amount_top_k
balance_den = a_amount_top_k + b_amount_top_k
weighted_key_num = b_extreme_key * a_amount_top_k
                 + a_extreme_key * b_amount_top_k
weighted_key_den = a_amount_top_k + b_amount_top_k
```

任意一侧为空时，`key_gap=-1`、`weighted_key_num=-1`、`weighted_key_den=0`。`balance_num` 和 `balance_den` 仍按已有 level 计算。

`SnapshotMetrics` 计算不修改双侧队列或输入快照。不同 Snapshot 之间没有状态依赖，可以按批次处理；输出顺序与输入 Snapshot 顺序一致。

## 5. 操作示例

以下示例均省略 `seq`。方括号表示同一 level 内从队首到队尾的顺序。

### 示例一：两侧不重叠

初始状态：

```text
A: empty
B: key=105 [(item=201, amount=6)]
```

插入 `(item=101, side=A, key=100, amount=4)`。

由于 `100 < 105`，不生成 `Resolution`。新 item 直接进入 A 侧：

```text
A: key=100 [(item=101, amount=4)]
B: key=105 [(item=201, amount=6)]
```

### 示例二：双方同时归零

初始状态：

```text
A: key=100 [(item=101, amount=7)]
B: empty
```

插入 `(item=201, side=B, key=100, amount=7)`，生成：

```text
(current=201, resident=101, key=100, amount=7)
```

两个 item 的 amount 同时变为 0。操作结束后 A、B 两侧均为空。

### 示例三：同一 level 按 FIFO 处理

初始状态：

```text
A: key=100 [(item=101, amount=7), (item=102, amount=5)]
B: empty
```

插入 `(item=201, side=B, key=99, amount=10)`，依次生成：

```text
(current=201, resident=101, key=100, amount=7)
(current=201, resident=102, key=100, amount=3)
```

最终状态：

```text
A: key=100 [(item=102, amount=2)]
B: empty
```

item 101 被删除，item 102 剩余 2，item 201 不进入队列。

### 示例四：一次插入跨越多个 level

初始状态：

```text
A: key=103 [(item=101, amount=4)]
   key=101 [(item=102, amount=6), (item=103, amount=5)]
B: empty
```

插入 `(item=201, side=B, key=100, amount=12)`，依次生成：

```text
(current=201, resident=101, key=103, amount=4)
(current=201, resident=102, key=101, amount=6)
(current=201, resident=103, key=101, amount=2)
```

最终状态：

```text
A: key=101 [(item=103, amount=3)]
B: empty
```

### 示例五：处理一部分后，当前 item 进入本侧

初始状态：

```text
A: empty
B: key=99  [(item=201, amount=3)]
   key=102 [(item=202, amount=5)]
```

插入 `(item=101, side=A, key=100, amount=6)`。它只能处理 B 侧的 `key=99`：

```text
(current=101, resident=201, key=99, amount=3)
```

此时当前 item 剩余 3，而下一个 B 侧 level 的 key 为 102。由于 `100 < 102`，处理停止，剩余部分进入 A 侧：

```text
A: key=100 [(item=101, amount=3)]
B: key=102 [(item=202, amount=5)]
```

### 示例六：删除同一 level 的队中元素

初始状态：

```text
A: key=100 [(item=101, amount=2),
            (item=102, amount=3),
            (item=103, amount=4)]
B: empty
```

执行 `REMOVE item=102` 后：

```text
A: key=100 [(item=101, amount=2),
            (item=103, amount=4)]
B: empty
```

随后插入 `(item=201, side=B, key=100, amount=5)`，依次生成：

```text
(current=201, resident=101, key=100, amount=2)
(current=201, resident=103, key=100, amount=3)
```

item 103 剩余 1。被删除的 item 102 不会再次出现。

### 示例七：非法插入与 item_id 复用

依次执行：

```text
INSERT (item=101, side=A, key=0,   amount=5)
INSERT (item=101, side=A, key=100, amount=5)
REMOVE (item=101)
INSERT (item=101, side=B, key=110, amount=2)
```

结果依次为：

```text
REJECTED_INVALID
OK
OK
REJECTED_DUPLICATE
```

第一次插入参数非法，不占用 item 101。第二次插入成功后，该编号永久视为已使用；即使随后删除，也不能再次插入。

### 示例八：已处理完的 item 不再活跃

初始状态：

```text
A: key=100 [(item=101, amount=4)]
B: empty
```

插入 `(item=201, side=B, key=99, amount=6)`，生成：

```text
(current=201, resident=101, key=100, amount=4)
```

item 101 被处理完并从队列中删除；item 201 剩余 2，进入 B 侧 `key=99` 的 level。此后执行 `REMOVE item=101` 返回 `REJECTED_NOT_FOUND`，再次使用 item 101 执行 `INSERT` 返回 `REJECTED_DUPLICATE`。

## 6. 实现要求

> 提示：可以在 `DESIGN.md` 或 README 中持续记录实现过程，包括对题目的理解、做出的假设、遇到的问题、查阅的资料、尝试过的方案、解决方式、失败结果以及尚未解决的内容。记录形式和篇幅不限，不要求只保留最终成功的方案。

### 6.1 阶段一：完整功能

先完成 CPU 版本的参考实现：

- 使用 CPU 按 `seq` 顺序处理事件，维护队列状态并生成 `ApplyResult`；
- 支持同一 level FIFO、完全处理、部分处理和跨 level 处理；
- 支持删除队首、队中和队尾 item；
- 支持 A、B 两侧交替插入和删除；
- 正确处理非法参数、重复标识和未知删除；
- 按指定间隔导出 `Snapshot`；
- 使用 CPU 批量计算 `SnapshotMetrics`；
- 提供固定输入的测试、端到端测试和可重复实验。

### 6.2 阶段二：选择并实现 GPU 批量计算

完成阶段一后，根据数据依赖和计算形式，自行选择至少一个适合批量并行的计算环节实现 CUDA 版本。**不要求将完整事件重放迁移到 GPU。**

可选目标包括但不限于：

- 对大量 `Snapshot` 批量计算 `SnapshotMetrics`；
- 对同一批快照计算额外的深度统计或多个 `K`；
- 对多组相互独立的事件流进行批量处理；
- 其他能够复用本题输入、状态或输出的数据并行计算。

对所选的计算环节，需满足以下要求：

- 明确输入、输出和一个并行工作单元的含义；
- 提供可独立运行的 CPU 参考实现；
- CPU 与 CUDA 输出结果一致；
- 批次大小可配置，支持在单个批次中处理多个独立工作单元。

如果选择 `SnapshotMetrics`，应说明 Host 数据如何转换为设备布局、可变深度如何表示，以及 grid、block、thread 与 snapshot/level 的对应关系。

### 6.3 阶段三：极致优化

保留阶段一的 CPU baseline 和阶段二第一版正确 CUDA 实现，思考如何进行更进一步的优化，主要记录优化尝试。

自行设计补充测试数据、数据生成程序及生成参数，用于正确性验证和性能实验。

优化可以涉及：

- 数据重新布局；
- Kernel 参数的选择；
- 合并访存、共享内存、寄存器使用和分支发散；
- 内存复用、pinned memory、异步拷贝和 CUDA streams；
- CPU 预处理、序列化、数据传输和 GPU 计算之间的流水线；
- ......

最终结果包含：

- CPU reference 总时间和吞吐；
- 第一版和最终 CUDA 实现的端到端时间；
- 使用的数据规模、硬件、CUDA Toolkit 和主要编译选项；
- 对优化结果、当前瓶颈和下一步方案的简短说明。

可以额外报告 Host-to-Device、kernel、Device-to-Host、不同 batch size 或 profiling 结果。GPU 不要求快于 CPU；没有获得加速时，给出测量结果和原因分析即可。

### 6.4 可选扩展

#### `MODIFY`

可增加以下事件格式：

```text
M seq item_id new_key new_amount
```

`MODIFY` 只作用于当前仍然活跃的 item，不改变其 `item_id` 和 `side`。`new_key` 和 `new_amount` 表示修改后的完整值，不是增量。

可按以下规则实现：

- `new_key <= 0` 或 `new_amount <= 0`：返回 `REJECTED_INVALID`，状态不变；
- `item_id` 不活跃：返回 `REJECTED_NOT_FOUND`，状态不变；
- `new_key` 不变且 `new_amount` 小于等于当前剩余 amount：直接更新 amount，保留该 item 在 level 内的原有位置；
- `new_key` 改变，或 `new_amount` 大于当前剩余 amount：先从原 level 删除该 item，再以当前 `MODIFY.seq` 作为新的顺序重新执行同侧插入流程；
- 重新执行插入流程时可能产生 `Resolution`；剩余部分进入新 level 的队尾；
- 原 level 变空时删除原 level；`item_id` 始终保持不变，也不触发 `REJECTED_DUPLICATE`。

实现 `MODIFY` 时，提供能够体现保留 FIFO、重新入队和非法修改行为的测试。

#### 其他扩展

- 在保持事件顺序的前提下优化大规模事件重放；
- 使用 pinned memory、CUDA streams 或 CUDA Graph；
- 使用 Nsight Compute 或 Nsight Systems 分析瓶颈；
- 支持大于 16384 的 `K`。

## 7. 运行与文件格式

程序的组织形式和运行接口自行设计，并在 README 中说明。基础数据流如下：

```text
events.txt -> 事件重放
               |-> resolutions.csv
               |-> states.csv
               `-> snapshots.txt -> 快照批量计算 -> metrics.csv
```

### 7.1 事件重放

#### 输入：events.txt

第一行是事件数量 `N`，随后每行是以下格式之一：

```text
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

#### 输出：resolutions.csv

无表头，每条 `Resolution` 输出一行：

```text
event_seq,current_item_id,resident_item_id,key,amount
```

上述示例对应：

```text
3,201,101,100,10
3,201,102,100,2
6,401,301,105,2
```

#### 输出：states.csv

无表头，每个输入事件输出一行：

```text
event_seq,a_extreme_key,b_extreme_key,a_amount,b_amount,active_count,status
```

上述示例对应：

```text
1,100,-1,10,0,1,OK
2,100,-1,15,0,2,OK
3,100,-1,3,0,1,OK
4,-1,-1,0,0,0,OK
5,-1,105,0,4,1,OK
6,-1,105,0,2,1,OK
```

#### 输出：snapshots.txt

第一行是快照数量 `S`：

```text
S
```

每个快照依次写入：

```text
event_seq NA NB
<NA 行 A 侧 level>
<NB 行 B 侧 level>
```

每个 level 的格式为：

```text
key amount
```

A 侧 level 按 key 严格递减，B 侧 level 按 key 严格递增。`NA` 或 `NB` 可以为 0。Snapshot 的生成时点按第 4.2 节执行。

### 7.2 快照批量计算

输入为 `snapshots.txt`，同时指定第 4.4 节中的计算深度 `K`。输出为无表头 CSV：

```text
event_seq,a_extreme_key,b_extreme_key,a_amount_top_k,b_amount_top_k,key_gap,balance_num,balance_den,weighted_key_num,weighted_key_den
```

输出顺序与 `snapshots.txt` 中的 Snapshot 顺序一致。

选择 `SnapshotMetrics` 作为 GPU 目标时，CPU 和 CUDA 实现使用相同的输入与 `K`，并分别输出计算结果。选择其他 GPU 目标时，自行定义输入和输出，并在 README 中说明。

没有可用 CUDA 设备时，事件重放和 CPU 快照批量计算仍应能够运行。运行 CUDA 路径但设备不可用时，应返回明确的错误信息，或明确说明正在使用 CPU fallback。

### 7.3 性能测试

性能测试的入口、参数和数据规模可以根据所选 GPU 目标自行设计。测试结果按第 6.3 节记录。

### 7.4 数据范围

```text
1 <= N <= 10,000,000
1 <= K <= 16384
1 <= item_id <= 2^64 - 1
-10^9 <= event.key <= 10^9
-10^9 <= event.amount <= 10^9
```

输入保证所有无符号字段和计数均可用 `uint64_t` 表示；所有有符号字段、聚合值，以及计算过程中的中间结果和最终结果均可用 `int64_t` 表示。

## 8. 提交内容

仓库包含：

1. 源代码，以及能够复现构建和运行过程的命令或脚本；
2. `README.md`：构建、运行、测试和 benchmark 命令；
3. 测试代码或测试脚本，包含 CPU 基本功能和所选 CPU/CUDA 一致性测试；
4. `DESIGN.md`：说明数据结构、GPU 目标、主要实现思路、做过的优化、测量结果、参考资料和已知限制。

## 9. 评测范围

- 是否完成可运行的 CPU 基础功能；
- GPU 目标的选择、数据布局和并行方式是否合理；
- 是否使用 CPU reference 验证 CUDA 结果；
- 实际测量并能解释优化有效或无效的原因（无论何种优化都可以计入讨论）；
- 说明当前限制和下一步改进方向。

不要求实现所有可选扩展，不要求 GPU 必须快于 CPU，也不以固定吞吐、固定测试数量或复杂优化技巧作为通过条件。
