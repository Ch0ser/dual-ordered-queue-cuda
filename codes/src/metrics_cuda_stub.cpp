#include "metrics_cuda.hpp"

#include <new>
#include <stdexcept>

namespace doq {

// ---------- CPU-only 构建的内存替代 ----------
// 没有 CUDA Runtime 时退化为普通 new/delete，使共享的数据容器仍可编译。
void* allocate_cuda_pinned_bytes(std::size_t bytes) {
    return ::operator new(bytes);
}

void free_cuda_pinned_bytes(void* pointer) noexcept {
    ::operator delete(pointer);
}

// ---------- CUDA 能力查询和纯 CPU 可执行的辅助逻辑 ----------
bool cuda_device_available(std::string* reason) {
    if (reason != nullptr) {
        *reason = "CUDA support was not built; configure with CUDA Toolkit 12.x+";
    }
    return false;
}

const char* cuda_snapshot_kernel_mode_name(CudaSnapshotKernelMode mode) noexcept {
    switch (mode) {
    case CudaSnapshotKernelMode::automatic:
        return "auto";
    case CudaSnapshotKernelMode::thread_per_snapshot:
        return "thread";
    case CudaSnapshotKernelMode::four_threads_per_snapshot:
        return "group4";
    case CudaSnapshotKernelMode::eight_threads_per_snapshot:
        return "group8";
    case CudaSnapshotKernelMode::sixteen_threads_per_snapshot:
        return "group16";
    case CudaSnapshotKernelMode::warp_per_snapshot:
        return "warp";
    }
    return "unknown";
}

CudaSnapshotKernelMode select_cuda_snapshot_kernel_mode(
    const CudaMetricsInputBatch& input,
    double* weighted_work) {
    long double work_sum = input.work_sum;
    long double squared_work_sum = input.squared_work_sum;
    if (input.summarized_snapshot_count != input.snapshots.size()) {
        work_sum = 0.0L;
        squared_work_sum = 0.0L;
        for (std::size_t index = 0; index < input.snapshots.size(); ++index) {
            const CudaSnapshotDescriptor& snapshot = input.snapshots.data()[index];
            const long double work =
                static_cast<long double>(snapshot.a_count) + snapshot.b_count;
            work_sum += work;
            squared_work_sum += work * work;
        }
    }
    const double value = work_sum == 0.0L
        ? 0.0
        : static_cast<double>(squared_work_sum / work_sum);
    if (weighted_work != nullptr) {
        *weighted_work = value;
    }
    if (value <= 96.0) {
        return CudaSnapshotKernelMode::four_threads_per_snapshot;
    }
    if (value <= 448.0) {
        return CudaSnapshotKernelMode::eight_threads_per_snapshot;
    }
    if (value < 1024.0) {
        return CudaSnapshotKernelMode::sixteen_threads_per_snapshot;
    }
    return CudaSnapshotKernelMode::warp_per_snapshot;
}

// ---------- CUDA 计算接口的明确失败实现 ----------
// 这些函数保持与 .cu 相同的符号，避免 CPU-only 构建链接失败；真正调用时
// 抛出清晰错误，而不是静默改用 CPU 造成性能结论混淆。
std::vector<SnapshotMetrics> compute_snapshot_metrics_cuda(
    const std::vector<Snapshot>&,
    std::size_t,
    CudaMetricsTimings*,
    CudaSnapshotKernelMode) {
    throw std::runtime_error(
        "CUDA support was not built; configure with CUDA Toolkit 12.x+");
}

std::vector<SnapshotMetrics> compute_snapshot_metrics_cuda_flat(
    const CudaMetricsInputBatch&,
    CudaMetricsTimings*,
    CudaSnapshotKernelMode) {
    throw std::runtime_error(
        "CUDA support was not built; configure with CUDA Toolkit 12.x+");
}

CudaPipelineBatchResult compute_snapshot_metrics_cuda_flat_pipeline(
    const CudaMetricsInputBatch&,
    std::size_t,
    CudaSnapshotKernelMode) {
    throw std::runtime_error(
        "CUDA support was not built; configure with CUDA Toolkit 12.x+");
}

CudaKernelBenchmarkStats benchmark_snapshot_metrics_cuda_kernel(
    const CudaMetricsInputBatch&,
    CudaSnapshotKernelMode,
    std::size_t,
    std::size_t,
    std::size_t) {
    throw std::runtime_error(
        "CUDA support was not built; configure with CUDA Toolkit 12.x+");
}

} // namespace doq
