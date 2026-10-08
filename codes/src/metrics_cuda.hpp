#pragma once

#include "types.hpp"

#include <cstddef>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace doq {

// ---------- 可复用的页锁定 Host 内存 ----------
// CUDA 构建使用 cudaMallocHost；CPU-only 构建使用普通内存，接口保持一致。
void* allocate_cuda_pinned_bytes(std::size_t bytes);
// 对象销毁期间如果释放内存又抛异常，很容易导致程序直接终止。
void free_cuda_pinned_bytes(void* pointer) noexcept;

// 用Pinned memory实现的精简版的vector：
// 1. 只支持push_back、reserve、clear、release、swap等最基本操作
// 2. 不支持迭代器、insert、erase等复杂操作
template <typename T>
class CudaPinnedBuffer {
    // 只允许trivially copyable类型，避免复杂的构造/析构/拷贝语义。普通字节可以，指针等也可以，但不允许有自定义构造函数或析构函数的类型。
    static_assert(std::is_trivially_copyable<T>::value,
                  "CudaPinnedBuffer requires a trivially copyable type");

public:
    CudaPinnedBuffer() = default;

    ~CudaPinnedBuffer() {
        release();
    }

    // 禁止拷贝构造和拷贝赋值，避免意外的深拷贝和资源管理问题。
    CudaPinnedBuffer(const CudaPinnedBuffer&) = delete;
    CudaPinnedBuffer& operator=(const CudaPinnedBuffer&) = delete;

    // 允许移动构造和移动赋值，方便资源转移。
    CudaPinnedBuffer(CudaPinnedBuffer&& other) noexcept {
        swap(other);
    }

    CudaPinnedBuffer& operator=(CudaPinnedBuffer&& other) noexcept {
        if (this != &other) {
            release();
            swap(other);
        }
        return *this;
    }

    // 预留指定容量的页锁定内存，如果当前容量不足则重新分配。
    void reserve(std::size_t requested_capacity) {
        if (requested_capacity <= capacity_) {
            return;
        }
        if (requested_capacity >
            std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::length_error("CUDA pinned buffer is too large");
        }
        T* replacement = static_cast<T*>(
            allocate_cuda_pinned_bytes(requested_capacity * sizeof(T)));
        if (size_ != 0) {
            std::memcpy(replacement, data_, size_ * sizeof(T));
        }
        free_cuda_pinned_bytes(data_);
        data_ = replacement;
        capacity_ = requested_capacity;
    }

    // 向缓冲区添加一个元素，如果缓冲区已满则自动扩容。
    void push_back(const T& value) {
        if (size_ == capacity_) {
            if (capacity_ == std::numeric_limits<std::size_t>::max()) {
                throw std::length_error("CUDA pinned buffer is too large");
            }
            std::size_t next_capacity;
            if (capacity_ == 0) {
                // 第一次添加元素时，先申请一小块初始空间。
                next_capacity = 64;
            } else if (capacity_ >
                       std::numeric_limits<std::size_t>::max() / 2) {
                // 当前容量已经无法安全乘 2，避免整数溢出。
                next_capacity = std::numeric_limits<std::size_t>::max();
            } else {
                // 后续空间不足时将容量翻倍，减少反复申请和复制。
                next_capacity = capacity_ * 2;
            }
            reserve(next_capacity);
        }
        data_[size_++] = value;
    }

    void clear() noexcept {
        size_ = 0;
    }

    void release() noexcept {
        free_cuda_pinned_bytes(data_);
        data_ = nullptr;
        size_ = 0;
        capacity_ = 0;
    }

    bool empty() const noexcept {
        return size_ == 0;
    }

    std::size_t size() const noexcept {
        return size_;
    }

    // 返回底层连续内存地址。
    T* data() noexcept {
        return data_;
    }

    const T* data() const noexcept {
        return data_;
    }

    // 移动构造和移动赋值的辅助函数，交换两个缓冲区的内容。
    void swap(CudaPinnedBuffer& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
        std::swap(capacity_, other.capacity_);
    }

private:
    // T* data_ 是指向页锁定内存的指针，size_ 是当前元素数量，capacity_ 是分配的容量。  
    T* data_{nullptr};
    std::size_t size_{};
    std::size_t capacity_{};
};



// ---------- GPU 使用的连续输入布局 ----------
// 每份 descriptor 记录元数据，记录位置、数量和关键值
struct CudaSnapshotDescriptor {
    std::uint64_t event_seq{};
    std::uint64_t a_offset{};
    std::uint64_t b_offset{};
    std::uint32_t a_count{};
    std::uint32_t b_count{};
    std::int64_t a_extreme_key{-1};
    std::int64_t b_extreme_key{-1};
};

// 记录整个批次的输入
struct CudaMetricsInputBatch {
    CudaPinnedBuffer<CudaSnapshotDescriptor> snapshots;
    // A/B 共用一块页锁定数组，offset 都相对这块数组计算，使每批 H2D
    // 从三次拷贝减少为 descriptor 和 amounts 两次拷贝。
    CudaPinnedBuffer<std::int64_t> amounts;
    // 记录整个批次的工作量统计，便于选择 kernel 档位。
    std::size_t summarized_snapshot_count{};
    long double work_sum{};
    long double squared_work_sum{};

    // 记录每个 snapshot 的工作量，便于后续计算加权平均。
    void record_snapshot_work(std::uint32_t a_count,
                              std::uint32_t b_count) noexcept {
        const long double work =
            static_cast<long double>(a_count) + b_count;
        ++summarized_snapshot_count;
        work_sum += work;
        squared_work_sum += work * work;
    }
    
    // 开始构造一个新批次时，需要把上一批的统计信息清零。
    void clear_work_summary() noexcept {
        summarized_snapshot_count = 0;
        work_sum = 0.0L;
        squared_work_sum = 0.0L;
    }
};

// 拆分整个 CUDA 流程的耗时，所有字段单位都是秒。
struct CudaMetricsTimings {
    // 记录整个批次的 GPU 流程耗时，便于分析瓶颈。
    double device_query_seconds{};
    double flatten_seconds{};
    double device_allocation_seconds{};
    double h2d_seconds{};
    double kernel_seconds{};
    double host_output_allocation_seconds{};
    double d2h_seconds{};
    double device_free_seconds{};
    double host_input_free_seconds{};
};

// CUDA kernel 档位选择和性能分析 
enum class CudaSnapshotKernelMode {
    automatic,
    thread_per_snapshot,
    four_threads_per_snapshot,
    eight_threads_per_snapshot,
    sixteen_threads_per_snapshot,
    warp_per_snapshot,
};











// 下面是 CUDA kernel 性能分析的结果结构体，记录了每次 kernel 启动的耗时统计信息。
struct CudaKernelBenchmarkStats {
    double median_milliseconds_per_launch{};
    double minimum_milliseconds_per_launch{};
    double maximum_milliseconds_per_launch{};
    std::size_t launches_per_sample{};
    std::size_t sample_count{};
};
// CUDA pipeline 批次执行的结果结构体，包含输出、耗时和重复次数。
struct CudaPipelineBatchResult {
    std::vector<SnapshotMetrics> output;
    CudaMetricsTimings activity;
    double wall_seconds{};
    std::size_t repetitions{};
};

// ---------- 对外 CUDA 接口 ----------
const char* cuda_snapshot_kernel_mode_name(CudaSnapshotKernelMode mode) noexcept;

// 用工作量加权深度 sum(work^2)/sum(work) 和实测阈值 96/448/1024，
// 为整批 Snapshot 选择协作线程数。
CudaSnapshotKernelMode select_cuda_snapshot_kernel_mode(
    const CudaMetricsInputBatch& input,
    double* weighted_work = nullptr);

// 未编译 CUDA 或没有可用设备时返回 false，并通过 reason 说明原因。
bool cuda_device_available(std::string* reason = nullptr);

// 接收普通嵌套 Snapshot；内部先压平，再计算一批，输出顺序与输入一致。
std::vector<SnapshotMetrics> compute_snapshot_metrics_cuda(
    const std::vector<Snapshot>& snapshots,
    std::size_t k,
    CudaMetricsTimings* timings = nullptr,
    CudaSnapshotKernelMode mode =
        CudaSnapshotKernelMode::thread_per_snapshot);

// 接收已经压平的连续布局；输入准备不计入此函数的 GPU 数据路径计时。
std::vector<SnapshotMetrics> compute_snapshot_metrics_cuda_flat(
    const CudaMetricsInputBatch& input,
    CudaMetricsTimings* timings = nullptr,
    CudaSnapshotKernelMode mode =
        CudaSnapshotKernelMode::thread_per_snapshot);

// 实验接口：使用两条 stream 和两套 workspace，让重复批次尝试重叠执行。
CudaPipelineBatchResult compute_snapshot_metrics_cuda_flat_pipeline(
    const CudaMetricsInputBatch& input,
    std::size_t repetitions,
    CudaSnapshotKernelMode mode = CudaSnapshotKernelMode::automatic);

// 实验接口：输入只上传一次，用 CUDA event 测量重复 kernel，不含传输和分配。
CudaKernelBenchmarkStats benchmark_snapshot_metrics_cuda_kernel(
    const CudaMetricsInputBatch& input,
    CudaSnapshotKernelMode mode,
    std::size_t warmup_launches,
    std::size_t launches_per_sample,
    std::size_t sample_count);

} // namespace doq
