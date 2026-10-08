#include "metrics_cuda.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace doq {
namespace {

// 本匿名命名空间中的类和辅助函数只在当前 .cu 文件内可见，
// 避免把 CUDA 实现细节暴露给其他编译单元。

// ---------- CUDA 错误处理 ----------
// 作用：统一检查 CUDA Runtime API 的返回值。
// 成功时直接返回；失败时把“哪项操作失败”和 CUDA 错误信息一起转成 C++ 异常。
void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + " failed: " + cudaGetErrorString(status));
    }
}

// ---------- 可增长、可跨批复用的 Device buffer ----------
// 容量足够时不再 cudaMalloc/cudaFree，降低多批处理的重复分配成本。
template <typename T>
class ReusableDeviceBuffer {
public:
    // 作用：对象生命周期结束时释放其拥有的显存。
    // 析构函数不能再向外抛异常，因此这里不调用会抛异常的 check_cuda。
    ~ReusableDeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    ReusableDeviceBuffer() = default;
    ReusableDeviceBuffer(const ReusableDeviceBuffer&) = delete;
    ReusableDeviceBuffer& operator=(const ReusableDeviceBuffer&) = delete;

    // 作用：保证显存至少能容纳 count 个 T。
    // 已有容量足够时什么也不做；只有容量不足时才释放旧块并重新申请。
    // 旧数据不需要保留，因为调用方随后会用新一批输入完全覆盖它。
    void ensure_capacity(std::size_t count) {
        if (count <= capacity_) {
            return;
        }
        if (data_ != nullptr) {
            check_cuda(cudaFree(data_), "cudaFree while growing reusable buffer");
            data_ = nullptr;
            capacity_ = 0;
        }
        if (count != 0) {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_),
                                  count * sizeof(T)),
                       "cudaMalloc reusable buffer");
            capacity_ = count;
        }
    }

    // 返回原始 Device 指针，供 cudaMemcpy 和 kernel 启动参数使用。
    T* get() {
        return data_;
    }

private:
    T* data_{nullptr};
    std::size_t capacity_{};
};

struct CudaMetricsWorkspace {
    // 一次计算需要三块显存：快照描述符、统一 amounts 和计算结果。
    // 三者都可以在后续批次中复用。
    ReusableDeviceBuffer<CudaSnapshotDescriptor> snapshots;
    ReusableDeviceBuffer<std::int64_t> amounts;
    ReusableDeviceBuffer<SnapshotMetrics> output;
};

// ---------- overlap 实验使用的页锁定输出和双缓冲 workspace ----------
class ReusablePinnedMetricsBuffer {
public:
    // 作用：释放流水线 D2H 使用的页锁定 Host 内存。
    ~ReusablePinnedMetricsBuffer() {
        if (data_ != nullptr) {
            cudaFreeHost(data_);
        }
    }

    ReusablePinnedMetricsBuffer() = default;
    ReusablePinnedMetricsBuffer(const ReusablePinnedMetricsBuffer&) = delete;
    ReusablePinnedMetricsBuffer& operator=(const ReusablePinnedMetricsBuffer&) = delete;

    // 作用：保证页锁定输出区至少能保存 count 个 SnapshotMetrics。
    // cudaMemcpyAsync 要实现真正的异步 Host 传输，需要使用页锁定内存。
    void ensure_capacity(std::size_t count) {
        if (count <= capacity_) {
            return;
        }
        if (data_ != nullptr) {
            check_cuda(cudaFreeHost(data_),
                       "cudaFreeHost while growing pipeline output");
            data_ = nullptr;
            capacity_ = 0;
        }
        check_cuda(cudaMallocHost(reinterpret_cast<void**>(&data_),
                                  count * sizeof(SnapshotMetrics)),
                   "cudaMallocHost pipeline output");
        capacity_ = count;
    }

    // 可写指针用于 D2H，const 指针用于把最后一批结果复制进 vector。
    SnapshotMetrics* get() { return data_; }
    const SnapshotMetrics* get() const { return data_; }

private:
    SnapshotMetrics* data_{};
    std::size_t capacity_{};
};

struct CudaPipelineSlot {
    // 每个 slot 都有独立的显存、页锁定输出、stream 和计时 event。
    // 两个 slot 交替使用，避免下一批覆盖仍在执行的上一批数据。
    CudaMetricsWorkspace workspace;
    ReusablePinnedMetricsBuffer host_output;
    cudaStream_t stream{};
    cudaEvent_t h2d_start{};
    cudaEvent_t h2d_end{};
    cudaEvent_t kernel_end{};
    cudaEvent_t d2h_end{};
    bool pending{};

    // 作用：退出线程时清理该 slot 的 CUDA 资源。
    // 先等待 stream 完成，再销毁 event 和 stream，防止资源仍被异步任务使用。
    ~CudaPipelineSlot() {
        if (stream != nullptr) cudaStreamSynchronize(stream);
        if (d2h_end != nullptr) cudaEventDestroy(d2h_end);
        if (kernel_end != nullptr) cudaEventDestroy(kernel_end);
        if (h2d_end != nullptr) cudaEventDestroy(h2d_end);
        if (h2d_start != nullptr) cudaEventDestroy(h2d_start);
        if (stream != nullptr) cudaStreamDestroy(stream);
    }
};

struct CudaPipelineWorkspace {
    // 双缓冲：第 0、1、0、1……个批次轮流进入两个 slot。
    CudaPipelineSlot slots[2];
};

// 作用：取得当前 CPU 线程专用的普通计算 workspace。
// thread_local 让同一线程的多批计算复用显存，同时避免不同 CPU 线程互相覆盖。
CudaMetricsWorkspace& current_workspace() {
    thread_local CudaMetricsWorkspace workspace;
    return workspace;
}

// 作用：取得当前 CPU 线程专用的双缓冲流水线 workspace。
CudaPipelineWorkspace& current_pipeline_workspace() {
    thread_local CudaPipelineWorkspace workspace;
    return workspace;
}

// ---------- Device 端公共结果组装 ----------
// 作用：根据已经求出的 A/B amount 总和，组装一份最终 SnapshotMetrics。
// 两种 kernel 都调用它，从而保证字段公式和空侧哨兵完全一致。
// __device__ 表示只能由 GPU 代码调用；__forceinline__ 建议编译器直接内联，
// 避免每份快照额外发生一次设备函数调用。
__device__ __forceinline__ void write_snapshot_metrics(
    const CudaSnapshotDescriptor& snapshot,
    std::int64_t a_sum,
    std::int64_t b_sum,
    SnapshotMetrics* destination) {
    // 先在当前线程的局部变量中完成所有字段，再一次性写入全局显存。
    SnapshotMetrics metrics{};
    metrics.event_seq = snapshot.event_seq;
    metrics.a_extreme_key = snapshot.a_extreme_key;
    metrics.b_extreme_key = snapshot.b_extreme_key;
    metrics.a_amount_top_k = a_sum;
    metrics.b_amount_top_k = b_sum;
    metrics.balance_num = a_sum - b_sum;
    metrics.balance_den = a_sum + b_sum;

    // 任一侧为空时，价差和加权 key 没有定义，按任务约定写入哨兵值。
    if (snapshot.a_extreme_key == -1 || snapshot.b_extreme_key == -1) {
        metrics.key_gap = -1;
        metrics.weighted_key_num = -1;
        metrics.weighted_key_den = 0;
    } else {
        metrics.key_gap = snapshot.b_extreme_key - snapshot.a_extreme_key;
        metrics.weighted_key_num =
            snapshot.b_extreme_key * a_sum + snapshot.a_extreme_key * b_sum;
        metrics.weighted_key_den = metrics.balance_den;
    }
    // 每份快照只写自己的 output[index]，不同线程之间不存在写冲突。
    *destination = metrics;
}

// ---------- 一线程处理一份 Snapshot 的基线 kernel ----------
// 作用：每个 CUDA 线程独立累加一份快照的 A/B 前 K 档 amount。
// 适合每份快照较浅、单线程循环不长的情况。
__global__ void snapshot_metrics_thread_kernel(
    const CudaSnapshotDescriptor* snapshots,
    const std::int64_t* a_amounts,
    const std::int64_t* b_amounts,
    std::uint64_t snapshot_count,
    SnapshotMetrics* output) {
    // 当前线程最先负责的快照编号，以及整个 grid 的线程总数。
    // grid-stride loop 让固定大小的 grid 也能覆盖任意多份快照。
    const std::uint64_t first =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t stride =
        static_cast<std::uint64_t>(blockDim.x) * gridDim.x;

    for (std::uint64_t index = first; index < snapshot_count; index += stride) {
        // 描述符按值读入局部变量，后续用 offset/count 定位统一 amounts 数组。
        const CudaSnapshotDescriptor snapshot = snapshots[index];
        std::int64_t a_sum = 0;
        std::int64_t b_sum = 0;

        // 当前线程串行累加这一份快照的 A、B 两段数据。
        for (std::uint32_t i = 0; i < snapshot.a_count; ++i) {
            a_sum += a_amounts[snapshot.a_offset + i];
        }
        for (std::uint32_t i = 0; i < snapshot.b_count; ++i) {
            b_sum += b_amounts[snapshot.b_offset + i];
        }

        // 求和结束后，由同一线程计算并写出其余指标字段。
        write_snapshot_metrics(snapshot, a_sum, b_sum, &output[index]);
    }
}

// ---------- 多线程协作处理一份 Snapshot 的分档 kernel ----------
// 作用：让 GroupSize 个线程共同处理一份较深的快照。
// GroupSize 在编译期取 4/8/16/32；各线程分担 level，再用 shuffle 完成组内归约。
template <unsigned int GroupSize>
__global__ void snapshot_metrics_group_kernel(
    const CudaSnapshotDescriptor* snapshots,
    const std::int64_t* a_amounts,
    const std::int64_t* b_amounts,
    std::uint64_t snapshot_count,
    SnapshotMetrics* output) {
    // 编译期限制线程组必须是 4、8、16、32 之一。
    // 这些条件保证一个小组完整位于一个硬件 warp 中，可安全使用 warp shuffle。
    static_assert(GroupSize >= 4 && GroupSize <= 32,
                  "group size must be between 4 and 32");
    static_assert((GroupSize & (GroupSize - 1)) == 0,
                  "group size must be a power of two");
    static_assert(32 % GroupSize == 0,
                  "group size must divide the hardware warp");

    // warp_lane：线程在所属 32 线程 warp 中的位置，范围为 0~31。
    // lane：线程在所属协作小组中的位置，范围为 0~GroupSize-1。
    // group_base：这个小组在 warp 中的起始 lane。
    const unsigned int warp_lane = threadIdx.x & (warpSize - 1);
    const unsigned int lane = warp_lane & (GroupSize - 1);
    const unsigned int group_base = warp_lane - lane;
    // group_mask 只标记当前协作小组的线程。
    // 例如 GroupSize=8 且 group_base=8 时，mask 覆盖 warp lane 8~15。
    constexpr unsigned int group_bits =
        GroupSize == 32 ? 0xffffffffU : (1U << GroupSize) - 1U;
    const unsigned int group_mask = group_bits << group_base;
    // 每 GroupSize 个连续线程组成一个逻辑工作项，对应一份快照。
    // group_stride 是整个 grid 一轮能同时处理的快照数。
    const std::uint64_t global_thread =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t first_group = global_thread / GroupSize;
    const std::uint64_t group_stride =
        static_cast<std::uint64_t>(blockDim.x) * gridDim.x / GroupSize;

    // 线程组级别的 grid-stride loop：同一小组可以继续处理后续快照。
    for (std::uint64_t index = first_group;
         index < snapshot_count;
         index += group_stride) {
        const CudaSnapshotDescriptor snapshot = snapshots[index];
        std::int64_t a_sum = 0;
        std::int64_t b_sum = 0;
        // 小组内按 lane 交错分工。
        // 例如 GroupSize=4 时，lane 0 处理 0/4/8...，lane 1 处理 1/5/9...。
        for (std::uint32_t i = lane; i < snapshot.a_count; i += GroupSize) {
            a_sum += a_amounts[snapshot.a_offset + i];
        }
        for (std::uint32_t i = lane; i < snapshot.b_count; i += GroupSize) {
            b_sum += b_amounts[snapshot.b_offset + i];
        }
        // 树形归约：每轮从相距 offset 的线程读取部分和。
        // offset 依次减半，最终 lane 0 得到整个小组的 A/B 总和。
        for (int offset = GroupSize / 2; offset > 0; offset /= 2) {
            a_sum += __shfl_down_sync(group_mask, a_sum, offset, GroupSize);
            b_sum += __shfl_down_sync(group_mask, b_sum, offset, GroupSize);
        }
        // 只有 lane 0 拥有完整归约结果，也只让它写出这一份快照，避免写冲突。
        if (lane == 0) {
            write_snapshot_metrics(snapshot, a_sum, b_sum, &output[index]);
        }
    }
}

constexpr unsigned int kThreadsPerBlock = 256;
constexpr std::uint64_t kMaximumGridBlocks = 65535;

// ---------- kernel 启动配置和统一分发 ----------
// 作用：根据快照数和每份快照占用的线程数，计算需要启动多少个 block。
// 每个 block 固定 256 个线程；结果向上取整，并限制为最多 65535 个 block。
// 超出一次覆盖范围的数据仍由 kernel 内的 grid-stride loop 继续处理。
unsigned int kernel_block_count(std::size_t snapshot_count,
                                CudaSnapshotKernelMode mode) {
    unsigned int group_size = 1;
    switch (mode) {
    case CudaSnapshotKernelMode::automatic:
        throw std::invalid_argument(
            "automatic kernel mode must be resolved before launch");
    case CudaSnapshotKernelMode::thread_per_snapshot:
        group_size = 1;
        break;
    case CudaSnapshotKernelMode::four_threads_per_snapshot:
        group_size = 4;
        break;
    case CudaSnapshotKernelMode::eight_threads_per_snapshot:
        group_size = 8;
        break;
    case CudaSnapshotKernelMode::sixteen_threads_per_snapshot:
        group_size = 16;
        break;
    case CudaSnapshotKernelMode::warp_per_snapshot:
        group_size = 32;
        break;
    }
    // 一线程档位每个 block 可处理 256 份快照；group32 则只能处理 8 份。
    const std::uint64_t work_items_per_block = kThreadsPerBlock / group_size;
    // (n + d - 1) / d 是整数除法下的向上取整。
    const std::uint64_t needed_blocks =
        (snapshot_count + work_items_per_block - 1) / work_items_per_block;
    return static_cast<unsigned int>(
        std::min<std::uint64_t>(needed_blocks, kMaximumGridBlocks));
}

// 作用：把统一的 mode 枚举分发到对应的 kernel 或模板实例。
// stream 为空时进入默认 stream；流水线实验会传入自己的非阻塞 stream。
void launch_snapshot_metrics_kernel(CudaMetricsWorkspace& workspace,
                                    std::size_t snapshot_count,
                                    CudaSnapshotKernelMode mode,
                                    cudaStream_t stream = nullptr) {
    const unsigned int block_count = kernel_block_count(snapshot_count, mode);
    switch (mode) {
    case CudaSnapshotKernelMode::automatic:
        throw std::invalid_argument(
            "automatic kernel mode must be resolved before launch");
    case CudaSnapshotKernelMode::thread_per_snapshot:
        // <<<block 数, 每 block 线程数, 动态共享内存字节数, stream>>>。
        snapshot_metrics_thread_kernel<<<block_count, kThreadsPerBlock, 0, stream>>>(
            workspace.snapshots.get(),
            // A/B 共用同一块 amounts，二者用各自 offset 定位，所以传入同一基址。
            workspace.amounts.get(),
            workspace.amounts.get(),
            static_cast<std::uint64_t>(snapshot_count),
            workspace.output.get());
        break;
    case CudaSnapshotKernelMode::four_threads_per_snapshot:
        snapshot_metrics_group_kernel<4><<<block_count, kThreadsPerBlock, 0, stream>>>(
            workspace.snapshots.get(),
            workspace.amounts.get(),
            workspace.amounts.get(),
            static_cast<std::uint64_t>(snapshot_count),
            workspace.output.get());
        break;
    case CudaSnapshotKernelMode::eight_threads_per_snapshot:
        snapshot_metrics_group_kernel<8><<<block_count, kThreadsPerBlock, 0, stream>>>(
            workspace.snapshots.get(),
            workspace.amounts.get(),
            workspace.amounts.get(),
            static_cast<std::uint64_t>(snapshot_count),
            workspace.output.get());
        break;
    case CudaSnapshotKernelMode::sixteen_threads_per_snapshot:
        snapshot_metrics_group_kernel<16><<<block_count, kThreadsPerBlock, 0, stream>>>(
            workspace.snapshots.get(),
            workspace.amounts.get(),
            workspace.amounts.get(),
            static_cast<std::uint64_t>(snapshot_count),
            workspace.output.get());
        break;
    case CudaSnapshotKernelMode::warp_per_snapshot:
        snapshot_metrics_group_kernel<32><<<block_count, kThreadsPerBlock, 0, stream>>>(
            workspace.snapshots.get(),
            workspace.amounts.get(),
            workspace.amounts.get(),
            static_cast<std::uint64_t>(snapshot_count),
            workspace.output.get());
        break;
    }
}

// ---------- 纯 kernel benchmark 的输入预上传 ----------
// 作用：在 benchmark 正式计时前完成显存扩容和 H2D。
// 这样测到的是纯 kernel 时间，不会混入 cudaMalloc 和数据传输。
void prepare_workspace(const CudaMetricsInputBatch& input,
                       CudaMetricsWorkspace& workspace) {
    // 先保证三块显存容量足够。
    workspace.snapshots.ensure_capacity(input.snapshots.size());
    workspace.amounts.ensure_capacity(input.amounts.size());
    workspace.output.ensure_capacity(input.snapshots.size());

    // 再同步上传描述符和统一 amounts；输出区只需分配，不需要初始化。
    check_cuda(cudaMemcpy(workspace.snapshots.get(),
                          input.snapshots.data(),
                          input.snapshots.size() * sizeof(CudaSnapshotDescriptor),
                          cudaMemcpyHostToDevice),
               "copy snapshot metadata to device");
    if (!input.amounts.empty()) {
        check_cuda(cudaMemcpy(workspace.amounts.get(),
                              input.amounts.data(),
                              input.amounts.size() * sizeof(std::int64_t),
                              cudaMemcpyHostToDevice),
                   "copy unified amounts to device");
    }
}

} // namespace

// ---------- 页锁定 Host 内存的 CUDA 实现 ----------
// 作用：为 CudaPinnedBuffer 申请指定字节数的页锁定 Host 内存。
// 页锁定内存不会被操作系统换出，是稳定高速 H2D 和真正异步拷贝的前提。
void* allocate_cuda_pinned_bytes(std::size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    void* pointer = nullptr;
    check_cuda(cudaMallocHost(&pointer, bytes), "cudaMallocHost");
    return pointer;
}

// 作用：释放 allocate_cuda_pinned_bytes 申请的页锁定内存。
// 接收空指针时什么也不做；noexcept 保证它可安全用于析构和清理路径。
void free_cuda_pinned_bytes(void* pointer) noexcept {
    if (pointer != nullptr) {
        cudaFreeHost(pointer);
    }
}

// ---------- 运行环境检查 ----------
// 作用：查询当前进程是否能看到至少一块 CUDA 设备。
// 返回 false 时，如果 reason 非空，就写入可读的失败原因；成功时清空旧原因。
bool cuda_device_available(std::string* reason) {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess) {
        if (reason != nullptr) {
            *reason = std::string("CUDA device query failed: ") +
                      cudaGetErrorString(status);
        }
        return false;
    }
    if (device_count == 0) {
        if (reason != nullptr) {
            *reason = "no CUDA device is available";
        }
        return false;
    }
    if (reason != nullptr) {
        reason->clear();
    }
    return true;
}

// ---------- kernel 档位名称和自动选择 ----------
// 作用：把枚举转换成命令行输出和实验表格使用的短名称。
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

// 自动选档
// 作用：根据整批快照的工作深度，自动选择 4/8/16/32 线程协作档位。
// work 定义为一份快照的 a_count + b_count。
// 加权工作深度 sum(work^2)/sum(work) 会让少量深快照获得更高权重，
// 避免它们被大量浅快照的普通平均值稀释。
CudaSnapshotKernelMode select_cuda_snapshot_kernel_mode(
    const CudaMetricsInputBatch& input,
    double* weighted_work) {

    // 优先使用构造输入时顺手累计的统计量，避免再次扫描 descriptors。
    long double work_sum = input.work_sum;
    long double squared_work_sum = input.squared_work_sum;

    // 如果统计数量和实际快照数不一致，说明调用方没有完整维护缓存统计；
    // 此时从 descriptors 重新计算，保证选档正确。
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
    // 全部快照均为空时令 weighted_work=0，避免除以零。
    const double value = work_sum == 0.0L
        ? 0.0
        : static_cast<double>(squared_work_sum / work_sum);
    // 这个输出参数是可选的，主要用于日志和实验记录。
    if (weighted_work != nullptr) {
        *weighted_work = value;
    }
    // 96、448、1024 来自对合成深度矩阵的实际 benchmark，
    // 分别是相邻协作档位的经验切换边界。
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

// ---------- 正式连续批数据路径 ----------
// 作用：接收已经压平的页锁定输入，完成一次同步 CUDA 计算并返回结果。
// 依次执行设备查询、workspace 扩容、H2D、kernel 和 D2H，并分段计时。
// 本函数不负责把 vector<Snapshot> 压平，因此 flatten 时间不包含在这里。
std::vector<SnapshotMetrics> compute_snapshot_metrics_cuda_flat(
    const CudaMetricsInputBatch& input,
    CudaMetricsTimings* timings,
    CudaSnapshotKernelMode mode) {
    using Clock = std::chrono::steady_clock;
    // 调用方要求计时时，先清零全部字段，避免残留上一次调用的数据。
    if (timings != nullptr) {
        *timings = {};
    }
    if (input.snapshots.empty()) {
        return {};
    }
    // automatic 必须在 kernel 启动前解析成一个具体档位。
    const CudaSnapshotKernelMode resolved_mode =
        mode == CudaSnapshotKernelMode::automatic
            ? select_cuda_snapshot_kernel_mode(input)
            : mode;

    // 1. 查询设备，单独记录 Runtime 初始化和设备检查时间。
    const auto query_start = Clock::now();
    std::string unavailable_reason;
    if (!cuda_device_available(&unavailable_reason)) {
        throw std::runtime_error(unavailable_reason);
    }
    const auto query_end = Clock::now();
    if (timings != nullptr) {
        timings->device_query_seconds =
            std::chrono::duration<double>(query_end - query_start).count();
    }

    // 2. 确保三个可复用 Device buffer 容量足够。
    // 第一次或批次变大时可能触发 cudaMalloc；容量足够时基本只做整数比较。
    const auto allocation_start = Clock::now();
    CudaMetricsWorkspace& workspace = current_workspace();
    workspace.snapshots.ensure_capacity(input.snapshots.size());
    workspace.amounts.ensure_capacity(input.amounts.size());
    workspace.output.ensure_capacity(input.snapshots.size());
    const auto allocation_end = Clock::now();
    if (timings != nullptr) {
        timings->device_allocation_seconds =
            std::chrono::duration<double>(allocation_end - allocation_start).count();
    }

    // 3. 同步上传 descriptor 和统一 amounts 数组。
    // input 使用页锁定 Host 内存；A/B amounts 已合并，所以总共两次 H2D 调用。
    const auto h2d_start = Clock::now();
    check_cuda(cudaMemcpy(workspace.snapshots.get(),
                          input.snapshots.data(),
                          input.snapshots.size() * sizeof(CudaSnapshotDescriptor),
                          cudaMemcpyHostToDevice),
               "copy snapshot metadata to device");
    if (!input.amounts.empty()) {
        check_cuda(cudaMemcpy(workspace.amounts.get(),
                              input.amounts.data(),
                              input.amounts.size() * sizeof(std::int64_t),
                              cudaMemcpyHostToDevice),
                   "copy unified amounts to device");
    }
    const auto h2d_end = Clock::now();
    if (timings != nullptr) {
        timings->h2d_seconds =
            std::chrono::duration<double>(h2d_end - h2d_start).count();
    }

    // 4. 启动已选档位的 kernel，并等待这一批完成。
    // kernel 启动是异步的：cudaGetLastError 检查启动配置，
    // cudaDeviceSynchronize 才真正等待执行结束并暴露运行期错误。
    const auto kernel_start = Clock::now();
    launch_snapshot_metrics_kernel(
        workspace, input.snapshots.size(), resolved_mode);
    check_cuda(cudaGetLastError(), "launch SnapshotMetrics kernel");
    check_cuda(cudaDeviceSynchronize(), "run SnapshotMetrics kernel");
    const auto kernel_end = Clock::now();
    if (timings != nullptr) {
        timings->kernel_seconds =
            std::chrono::duration<double>(kernel_end - kernel_start).count();
    }

    // 5. 创建最终 Host 输出容器。
    // 普通路径最终要返回 vector，因此在 D2H 前先分配足够空间。
    const auto host_allocation_start = Clock::now();
    std::vector<SnapshotMetrics> output(input.snapshots.size());
    const auto host_allocation_end = Clock::now();
    if (timings != nullptr) {
        timings->host_output_allocation_seconds =
            std::chrono::duration<double>(host_allocation_end - host_allocation_start)
                .count();
    }

    // 6. 同步将结果从显存复制回 Host。
    const auto d2h_start = Clock::now();
    check_cuda(cudaMemcpy(output.data(),
                          workspace.output.get(),
                          output.size() * sizeof(SnapshotMetrics),
                          cudaMemcpyDeviceToHost),
               "copy metrics to host");
    const auto d2h_end = Clock::now();
    if (timings != nullptr) {
        timings->d2h_seconds =
            std::chrono::duration<double>(d2h_end - d2h_start).count();
    }

    // 7. 正式路径故意不在每批结束后 cudaFree。
    // 显存由 thread_local workspace 持有到线程退出，后续批次可直接复用；
    // 因此这里保留 device_free 计时字段，但当前策略下它应接近零。
    const auto free_start = Clock::now();
    const auto free_end = Clock::now();
    if (timings != nullptr) {
        timings->device_free_seconds =
            std::chrono::duration<double>(free_end - free_start).count();
    }

    return output;
}

// ---------- overlap 实验路径 ----------
// 作用：把同一批输入重复执行 repetitions 次，测试双 stream 流水线。
// 两个 slot 交替工作，尝试让相邻批次的 H2D、kernel、D2H 在设备允许时重叠。
// 这是实验接口；返回最后一次结果、各阶段累计活动时间和整个循环的墙钟时间。
CudaPipelineBatchResult compute_snapshot_metrics_cuda_flat_pipeline(
    const CudaMetricsInputBatch& input,
    std::size_t repetitions,
    CudaSnapshotKernelMode mode) {
    using Clock = std::chrono::steady_clock;
    // 空输入没有可调度工作；repetitions 必须为正，避免“最后一次结果”无定义。
    if (input.snapshots.empty()) {
        return {};
    }
    if (repetitions == 0) {
        throw std::invalid_argument("pipeline repetitions must be positive");
    }
    // 在创建 stream 和 event 前确认 CUDA 环境可用，并解析 automatic 档位。
    std::string unavailable_reason;
    if (!cuda_device_available(&unavailable_reason)) {
        throw std::runtime_error(unavailable_reason);
    }
    const CudaSnapshotKernelMode resolved_mode =
        mode == CudaSnapshotKernelMode::automatic
            ? select_cuda_snapshot_kernel_mode(input)
            : mode;

    CudaPipelineBatchResult result;
    result.repetitions = repetitions;
    CudaPipelineWorkspace& pipeline = current_pipeline_workspace();

    // 第一次使用时，为两个 slot 分别创建非阻塞 stream 和四个阶段边界 event；
    // 每次调用都检查缓冲区容量，已经足够时不会重新分配。
    const auto allocation_start = Clock::now();
    for (CudaPipelineSlot& slot : pipeline.slots) {
        if (slot.stream == nullptr) {
            check_cuda(cudaStreamCreateWithFlags(
                           &slot.stream, cudaStreamNonBlocking),
                       "create pipeline stream");
            check_cuda(cudaEventCreate(&slot.h2d_start),
                       "create pipeline H2D start event");
            check_cuda(cudaEventCreate(&slot.h2d_end),
                       "create pipeline H2D end event");
            check_cuda(cudaEventCreate(&slot.kernel_end),
                       "create pipeline kernel end event");
            check_cuda(cudaEventCreate(&slot.d2h_end),
                       "create pipeline D2H end event");
        }
        slot.workspace.snapshots.ensure_capacity(input.snapshots.size());
        slot.workspace.amounts.ensure_capacity(input.amounts.size());
        slot.workspace.output.ensure_capacity(input.snapshots.size());
        slot.host_output.ensure_capacity(input.snapshots.size());
    }
    result.activity.device_allocation_seconds =
        std::chrono::duration<double>(Clock::now() - allocation_start).count();

    // finish 等待一个已经提交的 slot 完成，读取 CUDA event 的设备时间，
    // 再把 slot 标记为空闲。它不复制业务结果，结果已由异步 D2H 写入 host_output。
    auto finish = [&](CudaPipelineSlot& slot) {
        if (!slot.pending) {
            return;
        }
        // 等待最后一个 event，就等价于等待该 stream 中此前的 H2D、kernel、D2H 全部完成。
        check_cuda(cudaEventSynchronize(slot.d2h_end),
                   "wait for pipeline batch");
        // event 位于同一 stream 的阶段边界，因此三个区间分别对应 H2D、kernel、D2H。
        float milliseconds = 0.0F;
        check_cuda(cudaEventElapsedTime(
                       &milliseconds, slot.h2d_start, slot.h2d_end),
                   "measure pipeline H2D");
        result.activity.h2d_seconds += milliseconds / 1000.0;
        check_cuda(cudaEventElapsedTime(
                       &milliseconds, slot.h2d_end, slot.kernel_end),
                   "measure pipeline kernel");
        result.activity.kernel_seconds += milliseconds / 1000.0;
        check_cuda(cudaEventElapsedTime(
                       &milliseconds, slot.kernel_end, slot.d2h_end),
                   "measure pipeline D2H");
        result.activity.d2h_seconds += milliseconds / 1000.0;
        slot.pending = false;
    };

    // 墙钟时间覆盖整个重复提交和最终等待过程，用来判断重叠是否真的缩短总耗时。
    const auto wall_start = Clock::now();
    for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
        // 偶数批用 slot 0，奇数批用 slot 1；再次使用某个 slot 前必须先等它完成。
        CudaPipelineSlot& slot = pipeline.slots[repetition % 2];
        finish(slot);

        // 同一 stream 内严格保持 H2D -> kernel -> D2H 的顺序；
        // 不同 stream 的相邻批次才有机会在硬件上交叠。
        check_cuda(cudaEventRecord(slot.h2d_start, slot.stream),
                   "record pipeline H2D start");
        check_cuda(cudaMemcpyAsync(
                       slot.workspace.snapshots.get(),
                       input.snapshots.data(),
                       input.snapshots.size() * sizeof(CudaSnapshotDescriptor),
                       cudaMemcpyHostToDevice,
                       slot.stream),
                   "pipeline copy snapshot metadata");
        if (!input.amounts.empty()) {
            check_cuda(cudaMemcpyAsync(
                           slot.workspace.amounts.get(),
                           input.amounts.data(),
                           input.amounts.size() * sizeof(std::int64_t),
                           cudaMemcpyHostToDevice,
                           slot.stream),
                       "pipeline copy unified amounts");
        }
        check_cuda(cudaEventRecord(slot.h2d_end, slot.stream),
                   "record pipeline H2D end");
        launch_snapshot_metrics_kernel(
            slot.workspace, input.snapshots.size(), resolved_mode, slot.stream);
        check_cuda(cudaGetLastError(), "launch pipeline SnapshotMetrics kernel");
        check_cuda(cudaEventRecord(slot.kernel_end, slot.stream),
                   "record pipeline kernel end");
        check_cuda(cudaMemcpyAsync(
                       slot.host_output.get(),
                       slot.workspace.output.get(),
                       input.snapshots.size() * sizeof(SnapshotMetrics),
                       cudaMemcpyDeviceToHost,
                       slot.stream),
                   "pipeline copy metrics to host");
        check_cuda(cudaEventRecord(slot.d2h_end, slot.stream),
                   "record pipeline D2H end");
        // API 调用只是把工作排入 stream，此时通常尚未执行完，所以标记为 pending。
        slot.pending = true;
    }

    // 循环退出时两个 slot 都可能仍在执行，必须全部收尾后才能停止墙钟计时。
    finish(pipeline.slots[0]);
    finish(pipeline.slots[1]);
    result.wall_seconds =
        std::chrono::duration<double>(Clock::now() - wall_start).count();

    // API 只返回一份业务结果，因此把最后一次 repetition 的页锁定输出复制进 vector。
    const CudaPipelineSlot& last = pipeline.slots[(repetitions - 1) % 2];
    const auto materialization_start = Clock::now();
    result.output.assign(last.host_output.get(),
                         last.host_output.get() + input.snapshots.size());
    result.activity.host_output_allocation_seconds =
        std::chrono::duration<double>(Clock::now() - materialization_start).count();
    return result;
}

// ---------- kernel 独立计时实验 ----------
// 作用：排除分配和传输，仅比较各线程档位的 kernel 执行时间。
// 数据只上传一次；先预热，再用 CUDA event 做多组重复采样。
CudaKernelBenchmarkStats benchmark_snapshot_metrics_cuda_kernel(
    const CudaMetricsInputBatch& input,
    CudaSnapshotKernelMode mode,
    std::size_t warmup_launches,
    std::size_t launches_per_sample,
    std::size_t sample_count) {
    // benchmark 没有“空输入时间”的意义，并且每组至少要启动一次、至少采一组。
    if (input.snapshots.empty()) {
        throw std::invalid_argument("kernel benchmark input must not be empty");
    }
    if (launches_per_sample == 0 || sample_count == 0) {
        throw std::invalid_argument(
            "kernel benchmark launch and sample counts must be positive");
    }

    std::string unavailable_reason;
    if (!cuda_device_available(&unavailable_reason)) {
        throw std::runtime_error(unavailable_reason);
    }

    // 解析档位并在计时前完成 workspace 分配与输入上传。
    const CudaSnapshotKernelMode resolved_mode =
        mode == CudaSnapshotKernelMode::automatic
            ? select_cuda_snapshot_kernel_mode(input)
            : mode;
    CudaMetricsWorkspace& workspace = current_workspace();
    prepare_workspace(input, workspace);
    // 预热可吸收首次 kernel 启动、缓存和频率状态变化，预热结果不进入统计。
    for (std::size_t launch = 0; launch < warmup_launches; ++launch) {
        launch_snapshot_metrics_kernel(
            workspace, input.snapshots.size(), resolved_mode);
    }
    check_cuda(cudaGetLastError(), "launch warm-up SnapshotMetrics kernel");
    check_cuda(cudaDeviceSynchronize(), "run warm-up SnapshotMetrics kernel");

    // CUDA event 在 GPU 时间线上计时，比 CPU 墙钟更适合测很短的 kernel。
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "create benchmark start event");
    // event 是手动管理的 CUDA 资源；try/catch 保证中途异常时也能销毁。
    try {
        check_cuda(cudaEventCreate(&stop), "create benchmark stop event");
        std::vector<double> milliseconds_per_launch;
        milliseconds_per_launch.reserve(sample_count);
        for (std::size_t sample = 0; sample < sample_count; ++sample) {
            check_cuda(cudaEventRecord(start), "record benchmark start event");

            // 一组中连续启动多次，再除以次数，可降低计时分辨率和单次波动的影响。
            for (std::size_t launch = 0; launch < launches_per_sample; ++launch) {
                launch_snapshot_metrics_kernel(workspace,
                                               input.snapshots.size(),
                                               resolved_mode);
            }
            check_cuda(cudaGetLastError(), "launch timed SnapshotMetrics kernel");
            check_cuda(cudaEventRecord(stop), "record benchmark stop event");
            check_cuda(cudaEventSynchronize(stop), "wait for benchmark stop event");
            float elapsed_milliseconds = 0.0F;
            check_cuda(cudaEventElapsedTime(&elapsed_milliseconds, start, stop),
                       "measure SnapshotMetrics kernel time");
            milliseconds_per_launch.push_back(
                static_cast<double>(elapsed_milliseconds) /
                static_cast<double>(launches_per_sample));
        }
        check_cuda(cudaEventDestroy(stop), "destroy benchmark stop event");
        stop = nullptr;
        check_cuda(cudaEventDestroy(start), "destroy benchmark start event");
        start = nullptr;

        // 排序后报告中位数作为主要结果，同时保留最小值和最大值观察波动范围。
        std::sort(milliseconds_per_launch.begin(), milliseconds_per_launch.end());
        CudaKernelBenchmarkStats result;
        result.median_milliseconds_per_launch =
            milliseconds_per_launch[milliseconds_per_launch.size() / 2];
        result.minimum_milliseconds_per_launch = milliseconds_per_launch.front();
        result.maximum_milliseconds_per_launch = milliseconds_per_launch.back();
        result.launches_per_sample = launches_per_sample;
        result.sample_count = sample_count;
        return result;
    } catch (...) {
        // 清理已经成功创建的 event，然后用 throw; 原样继续抛出原异常。
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        throw;
    }
}

// ---------- 普通 Snapshot 兼容入口 ----------
// 作用：兼容 CPU 阶段使用的 vector<Snapshot> 输入。
// 先将嵌套 vector 压平成 descriptor + amounts，再复用正式连续批路径。
std::vector<SnapshotMetrics> compute_snapshot_metrics_cuda(
    const std::vector<Snapshot>& snapshots,
    std::size_t k,
    CudaMetricsTimings* timings,
    CudaSnapshotKernelMode mode) {
    using Clock = std::chrono::steady_clock;
    // 先清零计时结构，并检查业务参数 K。
    if (timings != nullptr) {
        *timings = {};
    }
    if (k == 0 || k > 16384) {
        throw std::invalid_argument("K must be in [1, 16384]");
    }
    if (snapshots.empty()) {
        return {};
    }

    const auto flatten_start = Clock::now();
    CudaMetricsInputBatch input;
    input.snapshots.reserve(snapshots.size());

    // 第一遍只统计每侧实际取 min(K, level 数) 后的元素总数。
    // 这样可以一次 reserve 足够的页锁定内存，避免压平过程中反复扩容和复制。
    std::size_t total_a_count = 0;
    std::size_t total_b_count = 0;
    for (const Snapshot& snapshot : snapshots) {
        const std::size_t a_count = std::min(k, snapshot.a_levels.size());
        const std::size_t b_count = std::min(k, snapshot.b_levels.size());
        // 用减法形式检查加法溢出，避免 total + count 已经溢出后再判断。
        if (a_count > std::numeric_limits<std::size_t>::max() - total_a_count ||
            b_count > std::numeric_limits<std::size_t>::max() - total_b_count) {
            throw std::length_error("flattened SnapshotMetrics input is too large");
        }
        total_a_count += a_count;
        total_b_count += b_count;
    }
    input.amounts.reserve(total_a_count + total_b_count);

    // 第二遍真正写入连续 amounts，并为每份快照记录 offset/count 和极值 key。
    for (const Snapshot& snapshot : snapshots) {
        const std::size_t a_count = std::min(k, snapshot.a_levels.size());
        const std::size_t b_count = std::min(k, snapshot.b_levels.size());
        // offset 是统一 amounts 数组中的元素下标，不是字节偏移。
        const std::uint64_t a_offset = input.amounts.size();
        for (std::size_t i = 0; i < a_count; ++i) {
            input.amounts.push_back(snapshot.a_levels[i].amount);
        }
        const std::uint64_t b_offset = input.amounts.size();
        for (std::size_t i = 0; i < b_count; ++i) {
            input.amounts.push_back(snapshot.b_levels[i].amount);
        }
        // A/B 的 front() 已按订单簿规则分别是 A 最大 key 和 B 最小 key；
        // 某侧为空时写入 -1 哨兵，且对应 count 为 0。
        input.snapshots.push_back(CudaSnapshotDescriptor{
            snapshot.event_seq,
            a_offset,
            b_offset,
            static_cast<std::uint32_t>(a_count),
            static_cast<std::uint32_t>(b_count),
            snapshot.a_levels.empty() ? -1 : snapshot.a_levels.front().key,
            snapshot.b_levels.empty() ? -1 : snapshot.b_levels.front().key,
        });
        // 压平时顺手累计自动选档统计，避免稍后再扫描一遍 descriptors。
        input.record_snapshot_work(static_cast<std::uint32_t>(a_count),
                                   static_cast<std::uint32_t>(b_count));
    }
    const auto flatten_end = Clock::now();

    // 把连续输入交给正式 CUDA 路径；其内部计时不包含上面的 flatten。
    CudaMetricsTimings flat_timings;
    std::vector<SnapshotMetrics> output =
        compute_snapshot_metrics_cuda_flat(input, timings == nullptr ? nullptr
                                                                     : &flat_timings,
                                           mode);

    // 主动释放两块页锁定输入并单独计时。
    // 页锁定内存是稀缺系统资源，用完后不应等待函数析构顺序隐式处理。
    const auto host_input_free_start = Clock::now();
    input.snapshots.release();
    input.amounts.release();
    const auto host_input_free_end = Clock::now();

    // 合并底层 CUDA 阶段计时，再补上兼容入口独有的 flatten 和 Host 释放时间。
    if (timings != nullptr) {
        *timings = flat_timings;
        timings->flatten_seconds =
            std::chrono::duration<double>(flatten_end - flatten_start).count();
        timings->host_input_free_seconds =
            std::chrono::duration<double>(host_input_free_end - host_input_free_start)
                .count();
    }
    return output;
}

} // namespace doq
