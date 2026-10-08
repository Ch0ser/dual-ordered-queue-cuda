#include "metrics_cuda.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
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
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
                   "cudaMalloc text benchmark buffer");
    }
    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() { return data_; }
private:
    T* data_{};
};

template <typename T>
class PinnedBuffer {
public:
    explicit PinnedBuffer(std::size_t count) : count_(count) {
        check_cuda(cudaMallocHost(reinterpret_cast<void**>(&data_),
                                  count * sizeof(T)),
                   "cudaMallocHost text benchmark buffer");
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

__device__ bool parse_u64(const char* text,
                          std::uint64_t& position,
                          std::uint64_t end,
                          std::uint64_t& value) {
    while (position < end &&
           (text[position] == ' ' || text[position] == '\r' ||
            text[position] == '\n' || text[position] == '\t')) {
        ++position;
    }
    if (position == end || text[position] < '0' || text[position] > '9') {
        return false;
    }
    value = 0;
    while (position < end && text[position] >= '0' && text[position] <= '9') {
        value = value * 10U + static_cast<unsigned int>(text[position] - '0');
        ++position;
    }
    return true;
}

__global__ void parse_text_metrics_kernel(
    const char* text,
    const std::uint64_t* snapshot_offsets,
    std::uint64_t text_size,
    std::uint64_t snapshot_count,
    std::uint32_t k,
    doq::SnapshotMetrics* output,
    unsigned int* parse_error) {
    const std::uint64_t first =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t stride =
        static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
    for (std::uint64_t index = first; index < snapshot_count; index += stride) {
        std::uint64_t position = snapshot_offsets[index];
        const std::uint64_t end = index + 1 < snapshot_count
            ? snapshot_offsets[index + 1]
            : text_size;
        std::uint64_t event_seq = 0;
        std::uint64_t a_depth = 0;
        std::uint64_t b_depth = 0;
        if (!parse_u64(text, position, end, event_seq) ||
            !parse_u64(text, position, end, a_depth) ||
            !parse_u64(text, position, end, b_depth)) {
            atomicExch(parse_error, 1U);
            continue;
        }

        std::int64_t a_sum = 0;
        std::int64_t b_sum = 0;
        std::int64_t a_extreme = -1;
        std::int64_t b_extreme = -1;
        for (std::uint64_t level = 0; level < a_depth; ++level) {
            std::uint64_t key = 0;
            std::uint64_t amount = 0;
            if (!parse_u64(text, position, end, key) ||
                !parse_u64(text, position, end, amount)) {
                atomicExch(parse_error, 1U);
                break;
            }
            if (level == 0) a_extreme = static_cast<std::int64_t>(key);
            if (level < k) a_sum += static_cast<std::int64_t>(amount);
        }
        for (std::uint64_t level = 0; level < b_depth; ++level) {
            std::uint64_t key = 0;
            std::uint64_t amount = 0;
            if (!parse_u64(text, position, end, key) ||
                !parse_u64(text, position, end, amount)) {
                atomicExch(parse_error, 1U);
                break;
            }
            if (level == 0) b_extreme = static_cast<std::int64_t>(key);
            if (level < k) b_sum += static_cast<std::int64_t>(amount);
        }

        doq::SnapshotMetrics metrics{};
        metrics.event_seq = event_seq;
        metrics.a_extreme_key = a_extreme;
        metrics.b_extreme_key = b_extreme;
        metrics.a_amount_top_k = a_sum;
        metrics.b_amount_top_k = b_sum;
        metrics.balance_num = a_sum - b_sum;
        metrics.balance_den = a_sum + b_sum;
        if (a_extreme == -1 || b_extreme == -1) {
            metrics.key_gap = -1;
            metrics.weighted_key_num = -1;
            metrics.weighted_key_den = 0;
        } else {
            metrics.key_gap = b_extreme - a_extreme;
            metrics.weighted_key_num = b_extreme * a_sum + a_extreme * b_sum;
            metrics.weighted_key_den = metrics.balance_den;
        }
        output[index] = metrics;
    }
}

bool equal_metrics(const doq::SnapshotMetrics& left,
                   const doq::SnapshotMetrics& right) {
    return left.event_seq == right.event_seq &&
           left.a_extreme_key == right.a_extreme_key &&
           left.b_extreme_key == right.b_extreme_key &&
           left.a_amount_top_k == right.a_amount_top_k &&
           left.b_amount_top_k == right.b_amount_top_k &&
           left.key_gap == right.key_gap &&
           left.balance_num == right.balance_num &&
           left.balance_den == right.balance_den &&
           left.weighted_key_num == right.weighted_key_num &&
           left.weighted_key_den == right.weighted_key_den;
}

void append_level(std::string& text, std::int64_t key, std::int64_t amount) {
    text += std::to_string(key);
    text.push_back(' ');
    text += std::to_string(amount);
    text.push_back('\n');
}

void make_dataset(const std::filesystem::path& path,
                  std::size_t snapshot_count,
                  std::uint32_t depth,
                  doq::CudaMetricsInputBatch& flat,
                  std::vector<doq::SnapshotMetrics>& expected) {
    std::string text;
    text.reserve(snapshot_count * (32U + static_cast<std::size_t>(depth) * 42U));
    text += std::to_string(snapshot_count);
    text.push_back('\n');
    flat.snapshots.reserve(snapshot_count);
    flat.amounts.reserve(snapshot_count * depth * 2);
    expected.reserve(snapshot_count);
    for (std::size_t index = 0; index < snapshot_count; ++index) {
        const std::uint64_t event_seq = index + 1;
        text += std::to_string(event_seq);
        text.push_back(' ');
        text += std::to_string(depth);
        text.push_back(' ');
        text += std::to_string(depth);
        text.push_back('\n');
        const std::uint64_t a_offset = flat.amounts.size();
        const std::int64_t a_extreme = 100000 + static_cast<std::int64_t>(index % 97);
        const std::int64_t b_extreme = 200000 + static_cast<std::int64_t>(index % 89);
        std::int64_t a_sum = 0;
        std::int64_t b_sum = 0;
        for (std::uint32_t level = 0; level < depth; ++level) {
            const std::int64_t amount =
                1 + static_cast<std::int64_t>((index * 17 + level * 13) % 10000);
            append_level(text, a_extreme - level, amount);
            flat.amounts.push_back(amount);
            a_sum += amount;
        }
        const std::uint64_t b_offset = flat.amounts.size();
        for (std::uint32_t level = 0; level < depth; ++level) {
            const std::int64_t amount =
                1 + static_cast<std::int64_t>((index * 19 + level * 11) % 10000);
            append_level(text, b_extreme + level, amount);
            flat.amounts.push_back(amount);
            b_sum += amount;
        }
        flat.snapshots.push_back(doq::CudaSnapshotDescriptor{
            event_seq, a_offset, b_offset, depth, depth, a_extreme, b_extreme});
        flat.record_snapshot_work(depth, depth);
        doq::SnapshotMetrics metrics{};
        metrics.event_seq = event_seq;
        metrics.a_extreme_key = a_extreme;
        metrics.b_extreme_key = b_extreme;
        metrics.a_amount_top_k = a_sum;
        metrics.b_amount_top_k = b_sum;
        metrics.key_gap = b_extreme - a_extreme;
        metrics.balance_num = a_sum - b_sum;
        metrics.balance_den = a_sum + b_sum;
        metrics.weighted_key_num = b_extreme * a_sum + a_extreme * b_sum;
        metrics.weighted_key_den = metrics.balance_den;
        expected.push_back(metrics);
    }
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!output) throw std::runtime_error("failed to write text benchmark dataset");
}

std::size_t count_tokens(const char* text, std::size_t begin, std::size_t end) {
    std::size_t tokens = 0;
    bool inside = false;
    for (std::size_t i = begin; i < end; ++i) {
        const bool whitespace = text[i] == ' ' || text[i] == '\t' ||
                                text[i] == '\r' || text[i] == '\n';
        if (!whitespace && !inside) ++tokens;
        inside = !whitespace;
    }
    return tokens;
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        const std::size_t snapshot_count = argc > 1
            ? static_cast<std::size_t>(std::stoull(argv[1]))
            : 65536;
        const std::uint32_t depth = argc > 2
            ? static_cast<std::uint32_t>(std::stoul(argv[2]))
            : 50;
        const std::size_t repetitions = argc > 3
            ? static_cast<std::size_t>(std::stoull(argv[3]))
            : 5;
        if (snapshot_count == 0 || depth == 0 || repetitions == 0) {
            throw std::invalid_argument("all arguments must be positive");
        }
        cudaDeviceProp device_properties{};
        check_cuda(cudaGetDeviceProperties(&device_properties, 0),
                   "query CUDA device properties");
        const std::filesystem::path path =
            "bench_data/datasets/gpu_text_" + std::to_string(snapshot_count) +
            "_d" + std::to_string(depth) + ".txt";

        doq::CudaMetricsInputBatch flat;
        std::vector<doq::SnapshotMetrics> expected;
        const auto generation_start = Clock::now();
        make_dataset(path, snapshot_count, depth, flat, expected);
        const double generation_seconds =
            std::chrono::duration<double>(Clock::now() - generation_start).count();

        const std::size_t text_bytes =
            static_cast<std::size_t>(std::filesystem::file_size(path));
        PinnedBuffer<char> host_text(text_bytes);
        const auto read_start = Clock::now();
        std::ifstream input(path, std::ios::binary);
        input.read(host_text.get(), static_cast<std::streamsize>(text_bytes));
        if (!input) throw std::runtime_error("failed to read raw text dataset");
        const double read_seconds =
            std::chrono::duration<double>(Clock::now() - read_start).count();

        const auto index_start = Clock::now();
        std::vector<std::uint64_t> offsets;
        offsets.reserve(snapshot_count);
        std::size_t line_start = 0;
        bool first_line = true;
        for (std::size_t i = 0; i <= text_bytes; ++i) {
            if (i == text_bytes || host_text.get()[i] == '\n') {
                if (!first_line &&
                    count_tokens(host_text.get(), line_start, i) == 3) {
                    offsets.push_back(line_start);
                }
                first_line = false;
                line_start = i + 1;
            }
        }
        if (offsets.size() != snapshot_count) {
            throw std::runtime_error("CPU byte index found the wrong snapshot count");
        }
        PinnedBuffer<std::uint64_t> host_offsets(snapshot_count);
        std::copy(offsets.begin(), offsets.end(), host_offsets.get());
        const double index_seconds =
            std::chrono::duration<double>(Clock::now() - index_start).count();

        doq::CudaMetricsTimings flat_activity;
        double flat_wall = 0.0;
        std::vector<doq::SnapshotMetrics> flat_output;
        (void)doq::compute_snapshot_metrics_cuda_flat(
            flat, nullptr, doq::CudaSnapshotKernelMode::automatic);
        for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
            doq::CudaMetricsTimings timing;
            const auto start = Clock::now();
            flat_output = doq::compute_snapshot_metrics_cuda_flat(
                flat, &timing, doq::CudaSnapshotKernelMode::automatic);
            flat_wall += std::chrono::duration<double>(Clock::now() - start).count();
            flat_activity.h2d_seconds += timing.h2d_seconds;
            flat_activity.kernel_seconds += timing.kernel_seconds;
            flat_activity.d2h_seconds += timing.d2h_seconds;
        }

        DeviceBuffer<char> device_text(text_bytes);
        DeviceBuffer<std::uint64_t> device_offsets(snapshot_count);
        DeviceBuffer<doq::SnapshotMetrics> device_output(snapshot_count);
        DeviceBuffer<unsigned int> device_error(1);
        PinnedBuffer<doq::SnapshotMetrics> host_output(snapshot_count);
        cudaEvent_t start{};
        cudaEvent_t h2d_end{};
        cudaEvent_t kernel_end{};
        cudaEvent_t d2h_end{};
        check_cuda(cudaEventCreate(&start), "create text start event");
        check_cuda(cudaEventCreate(&h2d_end), "create text H2D event");
        check_cuda(cudaEventCreate(&kernel_end), "create text kernel event");
        check_cuda(cudaEventCreate(&d2h_end), "create text D2H event");

        double text_wall = 0.0;
        double text_h2d = 0.0;
        double text_kernel = 0.0;
        double text_d2h = 0.0;
        const unsigned int blocks = static_cast<unsigned int>(
            std::min<std::size_t>((snapshot_count + 255) / 256, 65535));
        for (std::size_t repetition = 0; repetition <= repetitions; ++repetition) {
            check_cuda(cudaMemset(device_error.get(), 0, sizeof(unsigned int)),
                       "clear text parse error");
            const auto wall_start = Clock::now();
            check_cuda(cudaEventRecord(start), "record text start");
            check_cuda(cudaMemcpyAsync(device_text.get(), host_text.get(), text_bytes,
                                       cudaMemcpyHostToDevice),
                       "copy raw text to device");
            check_cuda(cudaMemcpyAsync(device_offsets.get(), host_offsets.get(),
                                       snapshot_count * sizeof(std::uint64_t),
                                       cudaMemcpyHostToDevice),
                       "copy text offsets to device");
            check_cuda(cudaEventRecord(h2d_end), "record text H2D end");
            parse_text_metrics_kernel<<<blocks, 256>>>(
                device_text.get(), device_offsets.get(), text_bytes,
                snapshot_count, depth, device_output.get(), device_error.get());
            check_cuda(cudaGetLastError(), "launch text parser kernel");
            check_cuda(cudaEventRecord(kernel_end), "record text kernel end");
            check_cuda(cudaMemcpyAsync(host_output.get(), device_output.get(),
                                       snapshot_count * sizeof(doq::SnapshotMetrics),
                                       cudaMemcpyDeviceToHost),
                       "copy text metrics to host");
            check_cuda(cudaEventRecord(d2h_end), "record text D2H end");
            check_cuda(cudaEventSynchronize(d2h_end), "wait for text benchmark");
            if (repetition != 0) {
                text_wall +=
                    std::chrono::duration<double>(Clock::now() - wall_start).count();
                float milliseconds = 0.0F;
                check_cuda(cudaEventElapsedTime(&milliseconds, start, h2d_end),
                           "measure text H2D");
                text_h2d += milliseconds / 1000.0;
                check_cuda(cudaEventElapsedTime(&milliseconds, h2d_end, kernel_end),
                           "measure text kernel");
                text_kernel += milliseconds / 1000.0;
                check_cuda(cudaEventElapsedTime(&milliseconds, kernel_end, d2h_end),
                           "measure text D2H");
                text_d2h += milliseconds / 1000.0;
            }
        }
        unsigned int parse_error = 0;
        check_cuda(cudaMemcpy(&parse_error, device_error.get(), sizeof(parse_error),
                              cudaMemcpyDeviceToHost),
                   "copy text parse error");
        if (parse_error != 0) throw std::runtime_error("GPU text parse failed");
        for (std::size_t i = 0; i < snapshot_count; ++i) {
            if (!equal_metrics(expected[i], host_output.get()[i]) ||
                !equal_metrics(expected[i], flat_output[i])) {
                throw std::runtime_error("text/flat mismatch at snapshot " +
                                         std::to_string(i));
            }
        }

        cudaEventDestroy(d2h_end);
        cudaEventDestroy(kernel_end);
        cudaEventDestroy(h2d_end);
        cudaEventDestroy(start);

        const double flat_bytes =
            flat.snapshots.size() * sizeof(doq::CudaSnapshotDescriptor) +
            flat.amounts.size() * sizeof(std::int64_t);
        std::cout << std::fixed << std::setprecision(6)
                  << "gpu_name=" << device_properties.name << '\n'
                  << "gpu_async_engine_count="
                  << device_properties.asyncEngineCount << '\n'
                  << "gpu_device_overlap=" << device_properties.deviceOverlap << '\n'
                  << "gpu_concurrent_kernels="
                  << device_properties.concurrentKernels << '\n'
                  << "snapshots=" << snapshot_count << '\n'
                  << "depth_per_side=" << depth << '\n'
                  << "repetitions=" << repetitions << '\n'
                  << "text_mib=" << text_bytes / (1024.0 * 1024.0) << '\n'
                  << "flat_mib=" << flat_bytes / (1024.0 * 1024.0) << '\n'
                  << "text_to_flat_byte_ratio=" << text_bytes / flat_bytes << '\n'
                  << "generation_seconds=" << generation_seconds << '\n'
                  << "text_file_read_seconds=" << read_seconds << '\n'
                  << "cpu_byte_index_seconds=" << index_seconds << '\n'
                  << "flat_cuda_seconds_per_pass=" << flat_wall / repetitions << '\n'
                  << "flat_h2d_seconds_per_pass="
                  << flat_activity.h2d_seconds / repetitions << '\n'
                  << "flat_kernel_seconds_per_pass="
                  << flat_activity.kernel_seconds / repetitions << '\n'
                  << "flat_d2h_seconds_per_pass="
                  << flat_activity.d2h_seconds / repetitions << '\n'
                  << "text_cuda_seconds_per_pass=" << text_wall / repetitions << '\n'
                  << "text_h2d_seconds_per_pass=" << text_h2d / repetitions << '\n'
                  << "text_parse_kernel_seconds_per_pass="
                  << text_kernel / repetitions << '\n'
                  << "text_d2h_seconds_per_pass=" << text_d2h / repetitions << '\n'
                  << "text_speedup_vs_flat=" << flat_wall / text_wall << '\n'
                  << "outputs_match=yes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
