#pragma once

#include "metrics_cuda.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace doq {

// ---------- 面向命令行的三个文件处理入口 ----------
// 阶段一入口：按 seq 顺序重放事件，并输出 Resolution、StateSummary、Snapshot。
void replay_file(const std::filesystem::path& events_path,
                 std::uint64_t snapshot_interval,
                 const std::filesystem::path& output_directory);

// CPU SnapshotMetrics 入口：读取快照，逐份调用 CPU reference 并写出 CSV。
void compute_metrics_file(const std::filesystem::path& snapshots_path,
                          std::size_t k,
                          const std::filesystem::path& metrics_path);

// CUDA 入口保留在同一 I/O 层，方便三个命令共享输入校验和输出格式。
void compute_metrics_cuda_file(const std::filesystem::path& snapshots_path,
                               std::size_t k,
                               std::size_t batch_size,
                               const std::filesystem::path& metrics_path,
                               CudaSnapshotKernelMode kernel_mode =
                                   CudaSnapshotKernelMode::automatic);

} // namespace doq
