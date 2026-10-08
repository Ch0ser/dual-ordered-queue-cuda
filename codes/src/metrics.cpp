#include "metrics.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace doq {
namespace {

// 指标全部使用 int64_t。下面三个辅助函数在运算前检查边界，避免 C++
// 有符号整数溢出导致未定义行为，也让非法大输入得到明确错误。
// 安全加法：计算 left + right；若结果超出 int64_t 范围则抛出异常。
std::int64_t checked_add(std::int64_t left, std::int64_t right, const char* field) {
    if ((right > 0 && left > std::numeric_limits<std::int64_t>::max() - right) ||
        (right < 0 && left < std::numeric_limits<std::int64_t>::min() - right)) {
        throw std::overflow_error(std::string(field) + " exceeds int64_t");
    }
    return left + right;
}

// 安全乘法：计算 left * right。
// 单独处理 0 和 -1 等边界，再根据操作数符号判断是否会溢出。
std::int64_t checked_multiply(std::int64_t left,
                              std::int64_t right,
                              const char* field) {
    if (left == 0 || right == 0) {
        return 0;
    }
    if (left == -1) {
        if (right == std::numeric_limits<std::int64_t>::min()) {
            throw std::overflow_error(std::string(field) + " exceeds int64_t");
        }
        return -right;
    }
    if (right == -1) {
        if (left == std::numeric_limits<std::int64_t>::min()) {
            throw std::overflow_error(std::string(field) + " exceeds int64_t");
        }
        return -left;
    }

    const auto minimum = std::numeric_limits<std::int64_t>::min();
    const auto maximum = std::numeric_limits<std::int64_t>::max();
    const bool overflow =
        (left > 0 && right > 0 && left > maximum / right) ||
        (left > 0 && right < 0 && right < minimum / left) ||
        (left < 0 && right > 0 && left < minimum / right) ||
        (left < 0 && right < 0 && left < maximum / right);
    if (overflow) {
        throw std::overflow_error(std::string(field) + " exceeds int64_t");
    }
    return left * right;
}

// 安全减法：计算 left - right；主要用于 balance_num 和 key_gap。
std::int64_t checked_subtract(std::int64_t left,
                              std::int64_t right,
                              const char* field) {
    if ((right < 0 && left > std::numeric_limits<std::int64_t>::max() + right) ||
        (right > 0 && left < std::numeric_limits<std::int64_t>::min() + right)) {
        throw std::overflow_error(std::string(field) + " exceeds int64_t");
    }
    return left - right;
}

// 加权key分子：分别安全计算 key*amount，再安全相加。
// 公式为 first_key*first_amount + second_key*second_amount。
std::int64_t checked_weighted_sum(std::int64_t first_key,
                                  std::int64_t first_amount,
                                  std::int64_t second_key,
                                  std::int64_t second_amount) {
    const std::int64_t first =
        checked_multiply(first_key, first_amount, "weighted_key_num");
    const std::int64_t second =
        checked_multiply(second_key, second_amount, "weighted_key_num");
    return checked_add(first, second, "weighted_key_num");
}

} // namespace

// ---------- CPU SnapshotMetrics 主流程 ----------
// 依次确定 extreme key、累加两侧前 K 档，再计算 balance、gap 和加权 key。
// 输入Snapshot只读，返回一份新的SnapshotMetrics，不修改订单簿或原快照。
SnapshotMetrics compute_snapshot_metrics(const Snapshot& snapshot, std::size_t k) {
    // K 是每侧最多统计的 level 数，不是 item 数。
    if (k == 0 || k > 16384) {
        throw std::invalid_argument("K must be in [1, 16384]");
    }

    SnapshotMetrics metrics;
    metrics.event_seq = snapshot.event_seq;

    // Snapshot 已按题目顺序保存，因此 front() 就是两侧的 extreme key。
    // 某侧为空时保留默认值 -1。
    if (!snapshot.a_levels.empty()) {
        metrics.a_extreme_key = snapshot.a_levels.front().key;
    }
    if (!snapshot.b_levels.empty()) {
        metrics.b_extreme_key = snapshot.b_levels.front().key;
    }
    // 实际要计算的level数可能小于 K，取两侧各自的 min(K, depth)。
    const std::size_t a_count = std::min(k, snapshot.a_levels.size());
    const std::size_t b_count = std::min(k, snapshot.b_levels.size());
    // 两侧独立累加前 min(K, depth) 档的聚合 amount。
    for (std::size_t i = 0; i < a_count; ++i) {
        metrics.a_amount_top_k = checked_add(
            metrics.a_amount_top_k, snapshot.a_levels[i].amount, "a_amount_top_k");
    }
    for (std::size_t i = 0; i < b_count; ++i) {
        metrics.b_amount_top_k = checked_add(
            metrics.b_amount_top_k, snapshot.b_levels[i].amount, "b_amount_top_k");
    }

    metrics.balance_num = checked_subtract(
        metrics.a_amount_top_k, metrics.b_amount_top_k, "balance_num");
    metrics.balance_den = checked_add(
        metrics.a_amount_top_k, metrics.b_amount_top_k, "balance_den");

    // balance 即使在单侧为空时也有定义；gap 和 weighted key 则按题意
    // 使用 -1/0 哨兵值，并提前返回，避免拿空侧的 -1 key 参与公式。
    if (snapshot.a_levels.empty() || snapshot.b_levels.empty()) {
        metrics.key_gap = -1;
        metrics.weighted_key_num = -1;
        metrics.weighted_key_den = 0;
        return metrics;
    }

    // 任务书要求分别输出 weighted_key_num 和 weighted_key_den；
    // 这里只计算这两个整数字段，不需要执行除法。
    metrics.key_gap = checked_subtract(
        metrics.b_extreme_key, metrics.a_extreme_key, "key_gap");
        
    metrics.weighted_key_num = checked_weighted_sum(
        metrics.b_extreme_key,
        metrics.a_amount_top_k,
        metrics.a_extreme_key,
        metrics.b_amount_top_k);
    metrics.weighted_key_den = metrics.balance_den;
    return metrics;
}

} // namespace doq
