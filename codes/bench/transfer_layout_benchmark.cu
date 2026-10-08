#include "types.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + " failed: " + cudaGetErrorString(status));
    }
}

template <typename T>
class PinnedBuffer {
public:
    explicit PinnedBuffer(std::size_t count) : count_(count) {
        if (count != 0) {
            check_cuda(cudaMallocHost(reinterpret_cast<void**>(&data_),
                                      count * sizeof(T)),
                       "cudaMallocHost transfer-layout buffer");
        }
    }
    ~PinnedBuffer() { cudaFreeHost(data_); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    T* get() { return data_; }
    const T* get() const { return data_; }
    std::size_t size() const { return count_; }
private:
    T* data_{};
    std::size_t count_{};
};

class DeviceBytes {
public:
    explicit DeviceBytes(std::size_t bytes) {
        if (bytes != 0) {
            check_cuda(cudaMalloc(&data_, bytes),
                       "cudaMalloc transfer-layout buffer");
        }
    }
    ~DeviceBytes() { cudaFree(data_); }
    DeviceBytes(const DeviceBytes&) = delete;
    DeviceBytes& operator=(const DeviceBytes&) = delete;
    void* get() { return data_; }
private:
    void* data_{};
};

struct Descriptor {
    std::uint64_t event_seq{};
    std::uint64_t a_offset{};
    std::uint64_t b_offset{};
    std::uint32_t a_count{};
    std::uint32_t b_count{};
    std::int64_t a_extreme_key{-1};
    std::int64_t b_extreme_key{-1};
};

struct SourceSnapshot {
    std::int64_t a_extreme_key{-1};
    std::int64_t b_extreme_key{-1};
    std::vector<std::int64_t> a_amounts;
    std::vector<std::int64_t> b_amounts;
};

std::vector<SourceSnapshot> read_source(const std::filesystem::path& path,
                                        std::size_t k) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open source: " + path.string());
    std::size_t count = 0;
    if (!(input >> count) || count == 0) {
        throw std::runtime_error("invalid public Snapshot count");
    }
    std::vector<SourceSnapshot> result;
    result.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        std::uint64_t event_seq = 0;
        std::size_t a_depth = 0;
        std::size_t b_depth = 0;
        if (!(input >> event_seq >> a_depth >> b_depth)) {
            throw std::runtime_error("invalid public Snapshot header");
        }
        SourceSnapshot snapshot;
        snapshot.a_amounts.reserve(std::min(k, a_depth));
        snapshot.b_amounts.reserve(std::min(k, b_depth));
        for (std::size_t level = 0; level < a_depth; ++level) {
            std::int64_t key = 0;
            std::int64_t amount = 0;
            if (!(input >> key >> amount)) {
                throw std::runtime_error("invalid public A level");
            }
            if (level == 0) snapshot.a_extreme_key = key;
            if (level < k) snapshot.a_amounts.push_back(amount);
        }
        for (std::size_t level = 0; level < b_depth; ++level) {
            std::int64_t key = 0;
            std::int64_t amount = 0;
            if (!(input >> key >> amount)) {
                throw std::runtime_error("invalid public B level");
            }
            if (level == 0) snapshot.b_extreme_key = key;
            if (level < k) snapshot.b_amounts.push_back(amount);
        }
        result.push_back(std::move(snapshot));
    }
    return result;
}

__device__ __forceinline__ void write_metrics(
    const Descriptor& descriptor,
    std::int64_t a_sum,
    std::int64_t b_sum,
    doq::SnapshotMetrics* output) {
    doq::SnapshotMetrics metrics{};
    metrics.event_seq = descriptor.event_seq;
    metrics.a_extreme_key = descriptor.a_extreme_key;
    metrics.b_extreme_key = descriptor.b_extreme_key;
    metrics.a_amount_top_k = a_sum;
    metrics.b_amount_top_k = b_sum;
    metrics.balance_num = a_sum - b_sum;
    metrics.balance_den = a_sum + b_sum;
    if (descriptor.a_extreme_key == -1 || descriptor.b_extreme_key == -1) {
        metrics.key_gap = -1;
        metrics.weighted_key_num = -1;
        metrics.weighted_key_den = 0;
    } else {
        metrics.key_gap = descriptor.b_extreme_key - descriptor.a_extreme_key;
        metrics.weighted_key_num =
            descriptor.b_extreme_key * a_sum +
            descriptor.a_extreme_key * b_sum;
        metrics.weighted_key_den = metrics.balance_den;
    }
    *output = metrics;
}

template <unsigned int GroupSize>
__global__ void metrics_kernel(const Descriptor* descriptors,
                               const std::int64_t* a_amounts,
                               const std::int64_t* b_amounts,
                               std::uint64_t snapshot_count,
                               doq::SnapshotMetrics* output) {
    const unsigned int warp_lane = threadIdx.x & 31U;
    const unsigned int lane = warp_lane & (GroupSize - 1U);
    const unsigned int group_base = warp_lane - lane;
    constexpr unsigned int bits =
        GroupSize == 32 ? 0xffffffffU : (1U << GroupSize) - 1U;
    const unsigned int mask = bits << group_base;
    const std::uint64_t global_thread =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t first_group = global_thread / GroupSize;
    const std::uint64_t stride =
        static_cast<std::uint64_t>(blockDim.x) * gridDim.x / GroupSize;
    for (std::uint64_t index = first_group; index < snapshot_count; index += stride) {
        const Descriptor descriptor = descriptors[index];
        std::int64_t a_sum = 0;
        std::int64_t b_sum = 0;
        for (std::uint32_t i = lane; i < descriptor.a_count; i += GroupSize) {
            a_sum += a_amounts[descriptor.a_offset + i];
        }
        for (std::uint32_t i = lane; i < descriptor.b_count; i += GroupSize) {
            b_sum += b_amounts[descriptor.b_offset + i];
        }
        for (int offset = GroupSize / 2; offset > 0; offset /= 2) {
            a_sum += __shfl_down_sync(mask, a_sum, offset, GroupSize);
            b_sum += __shfl_down_sync(mask, b_sum, offset, GroupSize);
        }
        if (lane == 0) write_metrics(descriptor, a_sum, b_sum, &output[index]);
    }
}

void launch_kernel(unsigned int group_size,
                   const Descriptor* descriptors,
                   const std::int64_t* a_amounts,
                   const std::int64_t* b_amounts,
                   std::size_t snapshot_count,
                   doq::SnapshotMetrics* output,
                   cudaStream_t stream) {
    constexpr unsigned int threads = 256;
    const std::uint64_t groups_per_block = threads / group_size;
    const unsigned int blocks = static_cast<unsigned int>(
        std::min<std::uint64_t>(
            (snapshot_count + groups_per_block - 1) / groups_per_block, 65535));
    switch (group_size) {
    case 4:
        metrics_kernel<4><<<blocks, threads, 0, stream>>>(
            descriptors, a_amounts, b_amounts, snapshot_count, output);
        break;
    case 8:
        metrics_kernel<8><<<blocks, threads, 0, stream>>>(
            descriptors, a_amounts, b_amounts, snapshot_count, output);
        break;
    case 16:
        metrics_kernel<16><<<blocks, threads, 0, stream>>>(
            descriptors, a_amounts, b_amounts, snapshot_count, output);
        break;
    default:
        metrics_kernel<32><<<blocks, threads, 0, stream>>>(
            descriptors, a_amounts, b_amounts, snapshot_count, output);
        break;
    }
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

bool equal_metrics(const doq::SnapshotMetrics& a,
                   const doq::SnapshotMetrics& b) {
    return std::memcmp(&a, &b, sizeof(doq::SnapshotMetrics)) == 0;
}

std::uint64_t checksum(const doq::SnapshotMetrics* rows, std::size_t count) {
    std::uint64_t value = 1469598103934665603ULL;
    const auto* bytes = reinterpret_cast<const unsigned char*>(rows);
    for (std::size_t i = 0; i < count * sizeof(doq::SnapshotMetrics); ++i) {
        value ^= bytes[i];
        value *= 1099511628211ULL;
    }
    return value;
}

struct LayoutResult {
    std::string name;
    double host_prepare_seconds{};
    double h2d_seconds{};
    double kernel_seconds{};
    double d2h_seconds{};
    double wall_seconds{};
    double h2d_gib_per_second{};
    std::uint64_t checksum{};
    std::vector<doq::SnapshotMetrics> output;
};

LayoutResult run_layout(const std::string& name,
                        const std::vector<SourceSnapshot>& source,
                        std::size_t snapshot_count,
                        std::size_t repetitions,
                        unsigned int group_size) {
    const auto prepare_start = Clock::now();
    std::size_t total_a = 0;
    std::size_t total_b = 0;
    for (std::size_t i = 0; i < snapshot_count; ++i) {
        total_a += source[i % source.size()].a_amounts.size();
        total_b += source[i % source.size()].b_amounts.size();
    }
    const std::size_t descriptor_bytes = snapshot_count * sizeof(Descriptor);
    const std::size_t a_bytes = total_a * sizeof(std::int64_t);
    const std::size_t b_bytes = total_b * sizeof(std::int64_t);
    const std::size_t unified_count = total_a + total_b;
    const std::size_t unified_bytes = unified_count * sizeof(std::int64_t);
    const std::size_t arena_amount_offset = align_up(descriptor_bytes, alignof(std::int64_t));
    const std::size_t arena_bytes = arena_amount_offset + unified_bytes;

    std::unique_ptr<PinnedBuffer<Descriptor>> host_descriptors;
    std::unique_ptr<PinnedBuffer<std::int64_t>> host_a;
    std::unique_ptr<PinnedBuffer<std::int64_t>> host_b;
    std::unique_ptr<PinnedBuffer<std::int64_t>> host_unified;
    std::unique_ptr<PinnedBuffer<unsigned char>> host_arena;
    Descriptor* descriptors = nullptr;
    std::int64_t* a_amounts = nullptr;
    std::int64_t* b_amounts = nullptr;
    if (name == "one_copy") {
        host_arena = std::make_unique<PinnedBuffer<unsigned char>>(arena_bytes);
        descriptors = reinterpret_cast<Descriptor*>(host_arena->get());
        a_amounts = reinterpret_cast<std::int64_t*>(
            host_arena->get() + arena_amount_offset);
        b_amounts = a_amounts;
    } else {
        host_descriptors = std::make_unique<PinnedBuffer<Descriptor>>(snapshot_count);
        descriptors = host_descriptors->get();
        if (name == "two_copy") {
            host_unified =
                std::make_unique<PinnedBuffer<std::int64_t>>(unified_count);
            a_amounts = host_unified->get();
            b_amounts = a_amounts;
        } else {
            host_a = std::make_unique<PinnedBuffer<std::int64_t>>(total_a);
            host_b = std::make_unique<PinnedBuffer<std::int64_t>>(total_b);
            a_amounts = host_a->get();
            b_amounts = host_b->get();
        }
    }

    std::size_t a_cursor = 0;
    std::size_t b_cursor = 0;
    std::size_t unified_cursor = 0;
    for (std::size_t i = 0; i < snapshot_count; ++i) {
        const SourceSnapshot& item = source[i % source.size()];
        const bool unified = name != "three_copy";
        const std::size_t a_offset = unified ? unified_cursor : a_cursor;
        std::int64_t* a_destination = unified
            ? a_amounts + unified_cursor
            : a_amounts + a_cursor;
        std::copy(item.a_amounts.begin(), item.a_amounts.end(), a_destination);
        if (unified) unified_cursor += item.a_amounts.size();
        else a_cursor += item.a_amounts.size();
        const std::size_t b_offset = unified ? unified_cursor : b_cursor;
        std::int64_t* b_destination = unified
            ? b_amounts + unified_cursor
            : b_amounts + b_cursor;
        std::copy(item.b_amounts.begin(), item.b_amounts.end(), b_destination);
        if (unified) unified_cursor += item.b_amounts.size();
        else b_cursor += item.b_amounts.size();
        descriptors[i] = Descriptor{
            static_cast<std::uint64_t>(i + 1),
            static_cast<std::uint64_t>(a_offset),
            static_cast<std::uint64_t>(b_offset),
            static_cast<std::uint32_t>(item.a_amounts.size()),
            static_cast<std::uint32_t>(item.b_amounts.size()),
            item.a_extreme_key,
            item.b_extreme_key,
        };
    }
    const double host_prepare_seconds =
        std::chrono::duration<double>(Clock::now() - prepare_start).count();

    const std::size_t transferred_bytes = descriptor_bytes + a_bytes + b_bytes;
    DeviceBytes device_primary(name == "one_copy" ? arena_bytes : descriptor_bytes);
    std::unique_ptr<DeviceBytes> device_a;
    std::unique_ptr<DeviceBytes> device_b;
    if (name == "three_copy") {
        device_a = std::make_unique<DeviceBytes>(a_bytes);
        device_b = std::make_unique<DeviceBytes>(b_bytes);
    } else if (name == "two_copy") {
        device_a = std::make_unique<DeviceBytes>(unified_bytes);
    }
    DeviceBytes device_output(snapshot_count * sizeof(doq::SnapshotMetrics));
    PinnedBuffer<doq::SnapshotMetrics> host_output(snapshot_count);

    auto* device_descriptors =
        reinterpret_cast<Descriptor*>(device_primary.get());
    std::int64_t* device_a_amounts = nullptr;
    std::int64_t* device_b_amounts = nullptr;
    if (name == "one_copy") {
        device_a_amounts = reinterpret_cast<std::int64_t*>(
            static_cast<unsigned char*>(device_primary.get()) + arena_amount_offset);
        device_b_amounts = device_a_amounts;
    } else if (name == "two_copy") {
        device_a_amounts = static_cast<std::int64_t*>(device_a->get());
        device_b_amounts = device_a_amounts;
    } else {
        device_a_amounts = static_cast<std::int64_t*>(device_a->get());
        device_b_amounts = static_cast<std::int64_t*>(device_b->get());
    }

    cudaStream_t stream{};
    cudaEvent_t start{};
    cudaEvent_t h2d_end{};
    cudaEvent_t kernel_end{};
    cudaEvent_t d2h_end{};
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "create transfer-layout stream");
    check_cuda(cudaEventCreate(&start), "create transfer-layout start event");
    check_cuda(cudaEventCreate(&h2d_end), "create transfer-layout H2D event");
    check_cuda(cudaEventCreate(&kernel_end), "create transfer-layout kernel event");
    check_cuda(cudaEventCreate(&d2h_end), "create transfer-layout D2H event");

    LayoutResult result;
    result.name = name;
    result.host_prepare_seconds = host_prepare_seconds;
    constexpr std::size_t warmup_repetitions = 3;
    for (std::size_t repetition = 0;
         repetition < warmup_repetitions + repetitions;
         ++repetition) {
        const auto wall_start = Clock::now();
        check_cuda(cudaEventRecord(start, stream), "record transfer-layout start");
        if (name == "one_copy") {
            check_cuda(cudaMemcpyAsync(device_primary.get(), host_arena->get(),
                                       arena_bytes, cudaMemcpyHostToDevice, stream),
                       "copy one transfer-layout arena");
        } else {
            check_cuda(cudaMemcpyAsync(device_primary.get(), descriptors,
                                       descriptor_bytes, cudaMemcpyHostToDevice, stream),
                       "copy transfer-layout descriptors");
            if (name == "two_copy") {
                check_cuda(cudaMemcpyAsync(device_a->get(), host_unified->get(),
                                           unified_bytes, cudaMemcpyHostToDevice, stream),
                           "copy unified transfer-layout amounts");
            } else {
                check_cuda(cudaMemcpyAsync(device_a->get(), host_a->get(), a_bytes,
                                           cudaMemcpyHostToDevice, stream),
                           "copy transfer-layout A amounts");
                check_cuda(cudaMemcpyAsync(device_b->get(), host_b->get(), b_bytes,
                                           cudaMemcpyHostToDevice, stream),
                           "copy transfer-layout B amounts");
            }
        }
        check_cuda(cudaEventRecord(h2d_end, stream),
                   "record transfer-layout H2D end");
        launch_kernel(group_size, device_descriptors, device_a_amounts,
                      device_b_amounts, snapshot_count,
                      static_cast<doq::SnapshotMetrics*>(device_output.get()), stream);
        check_cuda(cudaGetLastError(), "launch transfer-layout kernel");
        check_cuda(cudaEventRecord(kernel_end, stream),
                   "record transfer-layout kernel end");
        check_cuda(cudaMemcpyAsync(host_output.get(), device_output.get(),
                                   snapshot_count * sizeof(doq::SnapshotMetrics),
                                   cudaMemcpyDeviceToHost, stream),
                   "copy transfer-layout output");
        check_cuda(cudaEventRecord(d2h_end, stream),
                   "record transfer-layout D2H end");
        check_cuda(cudaEventSynchronize(d2h_end),
                   "wait for transfer-layout iteration");
        if (repetition >= warmup_repetitions) {
            result.wall_seconds +=
                std::chrono::duration<double>(Clock::now() - wall_start).count();
            float milliseconds = 0.0F;
            check_cuda(cudaEventElapsedTime(&milliseconds, start, h2d_end),
                       "measure transfer-layout H2D");
            result.h2d_seconds += milliseconds / 1000.0;
            check_cuda(cudaEventElapsedTime(&milliseconds, h2d_end, kernel_end),
                       "measure transfer-layout kernel");
            result.kernel_seconds += milliseconds / 1000.0;
            check_cuda(cudaEventElapsedTime(&milliseconds, kernel_end, d2h_end),
                       "measure transfer-layout D2H");
            result.d2h_seconds += milliseconds / 1000.0;
        }
    }
    result.h2d_gib_per_second =
        static_cast<double>(transferred_bytes) * repetitions /
        result.h2d_seconds / (1024.0 * 1024.0 * 1024.0);
    result.checksum = checksum(host_output.get(), snapshot_count);
    result.output.assign(host_output.get(), host_output.get() + snapshot_count);

    cudaEventDestroy(d2h_end);
    cudaEventDestroy(kernel_end);
    cudaEventDestroy(h2d_end);
    cudaEventDestroy(start);
    cudaStreamDestroy(stream);
    return result;
}

unsigned int select_group_size(const std::vector<SourceSnapshot>& source) {
    long double sum = 0.0L;
    long double squared = 0.0L;
    for (const SourceSnapshot& snapshot : source) {
        const long double work = static_cast<long double>(
            snapshot.a_amounts.size() + snapshot.b_amounts.size());
        sum += work;
        squared += work * work;
    }
    const long double weighted = sum == 0.0L ? 0.0L : squared / sum;
    if (weighted <= 96.0L) return 4;
    if (weighted <= 448.0L) return 8;
    if (weighted < 1024.0L) return 16;
    return 32;
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        const std::filesystem::path source_path = argc > 1
            ? argv[1]
            : "tests/data/public/performance/snapshots/snapshots_shallow.txt";
        const std::size_t snapshot_count = argc > 2
            ? static_cast<std::size_t>(std::stoull(argv[2]))
            : 65536;
        const std::size_t k = argc > 3
            ? static_cast<std::size_t>(std::stoull(argv[3]))
            : 50;
        const std::size_t repetitions = argc > 4
            ? static_cast<std::size_t>(std::stoull(argv[4]))
            : 3;
        const std::string order = argc > 5 ? argv[5] : "forward";
        if (order != "forward" && order != "reverse") {
            throw std::invalid_argument("order must be forward or reverse");
        }
        if (snapshot_count == 0 || k == 0 || k > 16384 || repetitions == 0) {
            throw std::invalid_argument("invalid benchmark arguments");
        }
        check_cuda(cudaFree(nullptr), "initialize CUDA context");
        void* pinned_warmup = nullptr;
        check_cuda(cudaMallocHost(&pinned_warmup, 1),
                   "initialize CUDA pinned allocator");
        check_cuda(cudaFreeHost(pinned_warmup),
                   "release CUDA pinned allocator warm-up");
        const std::vector<SourceSnapshot> source = read_source(source_path, k);
        const unsigned int group_size = select_group_size(source);
        std::vector<LayoutResult> results;
        std::array<const char*, 3> layout_names{
            "three_copy", "two_copy", "one_copy"};
        if (order == "reverse") {
            std::reverse(layout_names.begin(), layout_names.end());
        }
        for (const char* name : layout_names) {
            results.push_back(run_layout(
                name, source, snapshot_count, repetitions, group_size));
        }
        for (std::size_t layout = 1; layout < results.size(); ++layout) {
            if (results[layout].output.size() != results[0].output.size()) {
                throw std::runtime_error("transfer-layout output sizes differ");
            }
            for (std::size_t i = 0; i < results[0].output.size(); ++i) {
                if (!equal_metrics(results[0].output[i], results[layout].output[i])) {
                    throw std::runtime_error(
                        "transfer-layout output mismatch at snapshot " +
                        std::to_string(i));
                }
            }
        }
        std::cout << std::fixed << std::setprecision(6)
                  << "source=" << source_path.string() << '\n'
                  << "source_snapshots=" << source.size() << '\n'
                  << "replicated_snapshots=" << snapshot_count << '\n'
                  << "K=" << k << '\n'
                  << "repetitions=" << repetitions << '\n'
                  << "order=" << order << '\n'
                  << "group_size=" << group_size << '\n';
        for (const LayoutResult& result : results) {
            std::cout << result.name << "_host_prepare_seconds="
                      << result.host_prepare_seconds << '\n'
                      << result.name << "_h2d_seconds_per_pass="
                      << result.h2d_seconds / repetitions << '\n'
                      << result.name << "_kernel_seconds_per_pass="
                      << result.kernel_seconds / repetitions << '\n'
                      << result.name << "_d2h_seconds_per_pass="
                      << result.d2h_seconds / repetitions << '\n'
                      << result.name << "_cuda_seconds_per_pass="
                      << result.wall_seconds / repetitions << '\n'
                      << result.name << "_h2d_gib_per_second="
                      << result.h2d_gib_per_second << '\n'
                      << result.name << "_checksum=" << result.checksum << '\n';
        }
        std::cout << "outputs_match=yes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
