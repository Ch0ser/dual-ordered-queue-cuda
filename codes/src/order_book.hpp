#pragma once

#include "types.hpp"

#include <cstdint>
#include <functional>
#include <list>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace doq {

// ---------- 阶段一的内存状态机 ----------
// 对外只接收 Event 并导出结果/快照；具体容器和辅助操作全部封装在 private。
class OrderBook {
public:
    // 对外提供的三个功能
    // 处理一个事件并返回本次产生的记录及事件结束后的状态。
    ApplyResult apply(const Event& event);
    // 将当前队列聚合成只含 level 的快照，不暴露内部 item/FIFO 结构。
    Snapshot make_snapshot(std::uint64_t event_seq) const;
    // 测试和调试用的内部一致性检查，检查内部数据是否符合规则；返回 false 时 error 说明不符合的规则。
    bool validate_invariants(std::string* error = nullptr) const;

private:
    // 内部节点：Item 是单条活跃记录，Level 是同 key 的 FIFO 队列。
    // 队列中真正保存的活跃 item。insertion_seq 仅用于检查同档 FIFO 不变量。
    struct Item {
        std::uint64_t item_id{};
        Side side{Side::A};
        std::int64_t key{};
        std::int64_t amount{};
        //导致其入队的事件的 seq。仅用于检查同档 FIFO 不变量。
        std::uint64_t insertion_seq{};
    };

    // 一个 key 对应一个 level。list 的逻辑用法是 FIFO（尾插、头取）；
    // 之所以不用 std::queue，是因为 REMOVE 还需要通过迭代器删除中间元素。
    struct Level {
        std::int64_t total_amount{};
        std::list<Item> items;
    };

    // map.begin() 始终就是该侧最优档：A 递减得到最大 key，B 递增得到最小 key。
    using ALevels = std::map<std::int64_t, Level, std::greater<std::int64_t>>;
    using BLevels = std::map<std::int64_t, Level, std::less<std::int64_t>>;

    // 功能：从 item_id 直接定位到 map 中的 level 和 list 节点。
    // active_items_ 的 value：记录 item 位于哪一侧、哪个 level、list 的哪个节点。
    // std::list 在删除其他节点时不会让该迭代器失效。
    struct ActiveLocation {
        Side side{Side::A};
        std::int64_t key{};
        std::list<Item>::iterator item;
    };

    // 业务步骤拆分：事件分派后分别进入 INSERT、REMOVE、item剩余部分入队和摘要生成。
    ApplyResult apply_insert(const Event& event);
    ApplyResult apply_remove(const Event& event);
    StateSummary state_summary(std::uint64_t event_seq) const;
    void enqueue(const Event& event, std::int64_t remaining_amount);

    // 持久状态容器。
    // 四个容器分工：两侧 map 管排序和 FIFO，active_items_ 管“现在在哪”，
    // used_item_ids_ 管“历史上是否成功使用过”。
    ALevels a_levels_;
    BLevels b_levels_;
    std::unordered_map<std::uint64_t, ActiveLocation> active_items_;
    std::unordered_set<std::uint64_t> used_item_ids_;
};

} // namespace doq
