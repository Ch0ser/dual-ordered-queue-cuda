#pragma once

#include "types.hpp"

#include <cstddef>

namespace doq {

// CPU reference：读取一份 Snapshot，统计两侧前 K 个 level 的指标。
SnapshotMetrics compute_snapshot_metrics(const Snapshot& snapshot, std::size_t k);

} // namespace doq
