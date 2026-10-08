#include "order_book.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace doq {
namespace {

void set_error(std::string* destination, const std::string& message) {
    if (destination != nullptr) {
        *destination = message;
    }
}

} // namespace

// ---------- 对外统一入口 ----------
// 根据事件类型分发给 INSERT 或 REMOVE；调用者不需要了解两个实现函数。
ApplyResult OrderBook::apply(const Event& event) {
    // 对外只暴露一个入口，再按操作类型分发；事件顺序由 replay_file 保证。
    if (event.op == Op::Insert) {
        return apply_insert(event);
    }
    return apply_remove(event);
}

// ---------- INSERT：校验、对侧处理、剩余量入队 ----------
ApplyResult OrderBook::apply_insert(const Event& event) {
    ApplyResult result;

    // 检查key和amount
    // 被拒绝的事件不改队列、不产生 Resolution，但仍要生成当前 seq 的状态。
    if (event.key <= 0 || event.amount <= 0) {
        result.status = ApplyStatus::RejectedInvalid;
        result.state = state_summary(event.seq);
        return result;
    }

    //检查item_id是否重复
    if (used_item_ids_.find(event.item_id) != used_item_ids_.end()) {
        result.status = ApplyStatus::RejectedDuplicate;
        result.state = state_summary(event.seq);
        return result;
    }

    // 标记 item_id 已经使用过，避免后续重复使用。
    used_item_ids_.insert(event.item_id);

    // current item 的剩余量会在循环中逐步减少；只有最后仍大于 0 才入本侧。
    std::int64_t remaining = event.amount;

    //判断在哪一侧，处理对侧的最优档，直到对侧空或不再满足 key 条件。
    if (event.side == Side::A) {
        while (remaining > 0 && !b_levels_.empty()) {
            // B 按 key 递增，因此 begin() 永远是 B 的最小 key（最先处理的档）。
            auto level_it = b_levels_.begin();
            // A.key 必须 >= B.min_key；不满足时，后面的 B key 更大，也都不能处理。
            if (event.key < level_it->first) {
                break;
            }
            //level_it->first   key
            //level_it->second  Level
            Level& level = level_it->second; 
            // 同一 level 严格 FIFO，只能先处理 list.front()。
            Item& resident = level.items.front();
            // 双方本次最多处理较小的剩余量。
            const std::int64_t resolved = std::min(remaining, resident.amount);

            result.resolutions.push_back(Resolution{
                event.seq,
                event.item_id,
                resident.item_id,
                resident.key,
                resolved,
            });
            //current的剩余量；resident的剩余量；
            remaining -= resolved;
            resident.amount -= resolved;
            // total_amount 是 level 的缓存总量，每次修改 item 时必须同步维护。
            level.total_amount -= resolved;

            // resident 归零后，同时从活动索引和 FIFO 中移除。
            if (resident.amount == 0) {
                active_items_.erase(resident.item_id);
                level.items.pop_front();
            }
            // 最后一个 item 离开后，空 level 也必须从有序 map 中删除。
            if (level.items.empty()) {
                b_levels_.erase(level_it);
            }
        }
    } else {
        // event.side == Side::B
        while (remaining > 0 && !a_levels_.empty()) {
            // A 按 key 递减，因此 begin() 永远是 A 的最大 key。
            auto level_it = a_levels_.begin();
            // B.key 必须 <= A.max_key；不满足时，后面的 A key 更小，也都不能处理。
            if (event.key > level_it->first) {
                break;
            }

            Level& level = level_it->second;
            // B 插入时同样只处理对侧最优档的队首 item。
            Item& resident = level.items.front();
            const std::int64_t resolved = std::min(remaining, resident.amount);

            result.resolutions.push_back(Resolution{
                event.seq,
                event.item_id,
                resident.item_id,
                resident.key,
                resolved,
            });

            remaining -= resolved;
            resident.amount -= resolved;
            level.total_amount -= resolved;

            if (resident.amount == 0) {
                active_items_.erase(resident.item_id);
                level.items.pop_front();
            }
            if (level.items.empty()) {
                a_levels_.erase(level_it);
            }
        }
    }

    if (remaining > 0) {
        // 对侧已经为空或不再满足 key 条件，current 的剩余部分进入本侧队尾。
        enqueue(event, remaining);
    }

    result.status = ApplyStatus::Ok;
    result.state = state_summary(event.seq);
    return result;
}

// ---------- REMOVE：通过 item_id 索引直接删除活跃 item ----------
ApplyResult OrderBook::apply_remove(const Event& event) {
    ApplyResult result;
    // REMOVE 只给 item_id，先通过哈希索引直接定位；找不到就按业务规则拒绝。
    const auto active_it = active_items_.find(event.item_id);
    if (active_it == active_items_.end()) {
        result.status = ApplyStatus::RejectedNotFound;
        result.state = state_summary(event.seq);
        return result;
    }

    // 复制位置描述后再删除索引，location 内的 list 迭代器仍指向目标节点。
    //location包含 side key 和 list迭代器
    const ActiveLocation location = active_it->second;
    if (location.side == Side::A) {
        // 正常情况下 level 一定存在；找不到说明内部索引已经损坏，所以抛逻辑错误。
        const auto level_it = a_levels_.find(location.key);
        if (level_it == a_levels_.end()) {
            throw std::logic_error("active item points to a missing A level");
        }
        Level& level = level_it->second;
        // REMOVE 一次删掉该 item 的全部剩余量，不产生 Resolution。
        level.total_amount -= location.item->amount;
        // 直接通过 list 迭代器删除节点
        level.items.erase(location.item);
        // 删除后如果该 level 为空，也必须从 map 中删除。
        if (level.items.empty()) {
            a_levels_.erase(level_it);
        }
    } else {
        const auto level_it = b_levels_.find(location.key);
        if (level_it == b_levels_.end()) {
            throw std::logic_error("active item points to a missing B level");
        }
        Level& level = level_it->second;
        level.total_amount -= location.item->amount;
        level.items.erase(location.item);
        if (level.items.empty()) {
            b_levels_.erase(level_it);
        }
    }

    active_items_.erase(active_it);
    result.status = ApplyStatus::Ok;
    result.state = state_summary(event.seq);
    return result;
}

// ---------- 入队：维护 level、FIFO list 和 item_id 索引 ----------
void OrderBook::enqueue(const Event& event, std::int64_t remaining_amount) {
    // 这里只会接收已经通过校验、且尚有剩余 amount 的 current item。
    Item item{event.item_id, event.side, event.key, remaining_amount, event.seq};

    if (event.side == Side::A) {
        // level 不存在时创建；存在时直接取得原 level。
        auto [level_it, inserted] = a_levels_.try_emplace(event.key);
        (void)inserted;
        Level& level = level_it->second;
        // 尾插保证同 key 内 FIFO。
        // std::prev(end) 得到刚插入节点的稳定迭代器，也就是end的前一个
        level.items.push_back(item);
        auto item_it = std::prev(level.items.end());

        level.total_amount += remaining_amount;
        // 保存精确位置后，REMOVE 无需遍历 map 和 list。
        active_items_.emplace(event.item_id,
                              ActiveLocation{Side::A, event.key, item_it});
    } else {
        auto [level_it, inserted] = b_levels_.try_emplace(event.key);
        (void)inserted;
        Level& level = level_it->second;
        level.items.push_back(item);
        auto item_it = std::prev(level.items.end());
        level.total_amount += remaining_amount;
        active_items_.emplace(event.item_id,
                              ActiveLocation{Side::B, event.key, item_it});
    }
}

// ---------- 只提取当前两侧最优档的状态摘要 ----------
StateSummary OrderBook::state_summary(std::uint64_t event_seq) const {
    StateSummary state;
    state.event_seq = event_seq;
    state.active_count = static_cast<std::uint64_t>(active_items_.size());

    // 空侧保留结构体默认值：extreme_key=-1、amount=0。
    // 非空侧只汇总 begin() 指向的 extreme level。
    if (!a_levels_.empty()) {
        state.a_extreme_key = a_levels_.begin()->first;
        state.a_amount = a_levels_.begin()->second.total_amount;
    }
    if (!b_levels_.empty()) {
        state.b_extreme_key = b_levels_.begin()->first;
        state.b_amount = b_levels_.begin()->second.total_amount;
    }
    return state;
}

// ---------- 将内部 item 队列聚合成只含 level 的 Snapshot ----------
Snapshot OrderBook::make_snapshot(std::uint64_t event_seq) const {
    Snapshot snapshot;
    snapshot.event_seq = event_seq;
    //提前分配容量，减少vector扩容次数
    snapshot.a_levels.reserve(a_levels_.size());
    snapshot.b_levels.reserve(b_levels_.size());

    // Snapshot 保存全部非空 level，但每档只保存 key 和聚合后的 total_amount。
    // map 已经维持正确顺序，因此按迭代顺序写入即可。
    for (const auto& [key, level] : a_levels_) {
        snapshot.a_levels.push_back(LevelSnapshot{key, level.total_amount});
    }
    for (const auto& [key, level] : b_levels_) {
        snapshot.b_levels.push_back(LevelSnapshot{key, level.total_amount});
    }
    return snapshot;
}




// ---------- 测试和调试用的内部一致性检查 ----------
bool OrderBook::validate_invariants(std::string* error) const {
    // 事件完全处理后，两侧不能还存在可继续处理的 extreme level。
    if (!a_levels_.empty() && !b_levels_.empty() &&
        a_levels_.begin()->first >= b_levels_.begin()->first) {
        set_error(error, "A.max_key must be smaller than B.min_key");
        return false;
    }

    std::unordered_set<std::uint64_t> seen_ids;
    std::size_t counted_items = 0;

    // 这段只用于测试/调试：重新遍历真实容器，交叉检查 level 总量、FIFO、
    // used-id 集合和 active_items_ 索引是否彼此一致。
    const auto validate_side = [&](const auto& levels, Side expected_side) {
        for (const auto& [key, level] : levels) {
            if (level.items.empty()) {
                set_error(error, "an empty level is still present");
                return false;
            }

            std::int64_t amount_sum = 0;
            std::uint64_t previous_seq = 0;
            bool first = true;
            for (const Item& item : level.items) {
                ++counted_items;
                if (item.side != expected_side || item.key != key || item.amount <= 0) {
                    set_error(error, "an active item has inconsistent side/key/amount");
                    return false;
                }
                if (!first && item.insertion_seq <= previous_seq) {
                    set_error(error, "FIFO insertion sequence is not strictly increasing");
                    return false;
                }
                first = false;
                previous_seq = item.insertion_seq;
                amount_sum += item.amount;

                if (!seen_ids.insert(item.item_id).second) {
                    set_error(error, "an active item_id appears more than once");
                    return false;
                }
                if (used_item_ids_.find(item.item_id) == used_item_ids_.end()) {
                    set_error(error, "an active item_id is absent from the used-id set");
                    return false;
                }

                const auto location_it = active_items_.find(item.item_id);
                if (location_it == active_items_.end()) {
                    set_error(error, "an active item is absent from the item index");
                    return false;
                }
                const ActiveLocation& location = location_it->second;
                if (location.side != expected_side || location.key != key ||
                    &(*location.item) != &item) {
                    set_error(error, "an item-index entry points to the wrong item");
                    return false;
                }
            }
            if (amount_sum != level.total_amount || level.total_amount <= 0) {
                set_error(error, "a level total does not match its item amounts");
                return false;
            }
        }
        return true;
    };

    if (!validate_side(a_levels_, Side::A) || !validate_side(b_levels_, Side::B)) {
        return false;
    }
    if (counted_items != active_items_.size()) {
        set_error(error, "the active item count does not match the index size");
        return false;
    }
    return true;
}

} // namespace doq
