#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace doq {

// ---------- 事件和处理状态枚举 ----------
// Side 和 Op 直接对应题面中的两种队列方向与两种输入操作。
// enum class 只能从定义好的值中选择，而且具有类型检查。
enum class Side {
    A,
    B,
};

enum class Op {
    Insert,
    Remove,
};

enum class ApplyStatus {
    Ok,
    RejectedInvalid,
    RejectedDuplicate,
    RejectedNotFound,
};

// inline 允许相同函数定义出现在多个编译单元中，最终仍被视为同一个函数。
// 将内部枚举转换成输出文件要求的状态字符串。
inline std::string_view to_string(ApplyStatus status) {
    switch (status) {
    case ApplyStatus::Ok:
        return "OK";
    case ApplyStatus::RejectedInvalid:
        return "REJECTED_INVALID";
    case ApplyStatus::RejectedDuplicate:
        return "REJECTED_DUPLICATE";
    case ApplyStatus::RejectedNotFound:
        return "REJECTED_NOT_FOUND";
    }
    return "UNKNOWN";
}

// ---------- 事件回放的输入与输出 ----------
struct Event {
    // seq 是事件顺序号；item_id 是业务对象的永久标识。
    std::uint64_t seq{};
    std::uint64_t item_id{};
    Side side{Side::A};
    Op op{Op::Insert};
    std::int64_t key{};
    std::int64_t amount{};
};

// 一次 INSERT 可能连续处理多个 resident item，因此一个事件可以产生多条
// Resolution。key 按题意记录 resident item 所在档位的 key。
struct Resolution {
    std::uint64_t event_seq{};
    std::uint64_t current_item_id{};
    std::uint64_t resident_item_id{};
    std::int64_t key{};
    std::int64_t amount{};
};

// 每个输入事件处理结束后都生成一条 StateSummary，非法/拒绝事件也不例外。
// a_amount/b_amount 只统计各自 extreme level 的总量，不是整侧总量。
struct StateSummary {
    std::uint64_t event_seq{};
    std::int64_t a_extreme_key{-1};
    std::int64_t b_extreme_key{-1};
    std::int64_t a_amount{};
    std::int64_t b_amount{};
    std::uint64_t active_count{};
};

// apply() 的完整返回值：处理状态、0 到多条 Resolution，以及事件后的状态。
struct ApplyResult {
    ApplyStatus status{ApplyStatus::Ok};
    std::vector<Resolution> resolutions;
    StateSummary state;
};

// ---------- Snapshot 及其派生指标 ----------
// Snapshot 不保留单个 item，只保留每个非空 level 的 key 和剩余 amount 总和。
struct LevelSnapshot {
    std::int64_t key{};
    std::int64_t amount{};
};

struct Snapshot {
    // A 已按 key 递减，B 已按 key 递增；数组长度就是该侧 depth。
    std::uint64_t event_seq{};
    std::vector<LevelSnapshot> a_levels;
    std::vector<LevelSnapshot> b_levels;
};

// 对一份 Snapshot 的派生统计。任务书直接要求输出 num/den 四个整数字段，
// 程序只计算这些字段，不执行除法。
struct SnapshotMetrics {
    std::uint64_t event_seq{};
    std::int64_t a_extreme_key{-1};
    std::int64_t b_extreme_key{-1};
    std::int64_t a_amount_top_k{};
    std::int64_t b_amount_top_k{};
    std::int64_t key_gap{-1};
    std::int64_t balance_num{};
    std::int64_t balance_den{};
    std::int64_t weighted_key_num{-1};
    std::int64_t weighted_key_den{};
};

} // namespace doq
