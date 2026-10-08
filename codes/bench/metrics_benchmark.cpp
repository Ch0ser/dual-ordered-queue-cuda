#include "metrics.hpp"
#include "metrics_cuda.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::array<char, 8> kMagic{{'D', 'O', 'Q', 'B', 'E', 'N', 'C', 'H'}};
constexpr std::uint32_t kFormatVersion = 1;
constexpr std::uint64_t kDefaultSeed = 20260927;

struct DatasetHeader {
    std::uint32_t version{};
    std::uint64_t snapshot_count{};
    std::uint64_t seed{};
    std::uint32_t nominal_depth{};
    std::uint64_t a_level_count{};
    std::uint64_t b_level_count{};
};

class SplitMix64 {
public:
    explicit SplitMix64(std::uint64_t seed) : state_(seed) {}

    std::uint64_t next() {
        std::uint64_t value = (state_ += 0x9E3779B97F4A7C15ULL);
        value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
        value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
        return value ^ (value >> 31U);
    }

    std::uint64_t below(std::uint64_t limit) {
        if (limit == 0) {
            return 0;
        }
        return next() % limit;
    }

private:
    std::uint64_t state_;
};

template <typename T>
void write_value(std::ostream& output, const T& value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!output) {
        throw std::runtime_error("failed while writing benchmark dataset");
    }
}

template <typename T>
T read_value(std::istream& input) {
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) {
        throw std::runtime_error("benchmark dataset is truncated");
    }
    return value;
}

void write_header(std::ostream& output, const DatasetHeader& header) {
    output.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
    write_value(output, header.version);
    write_value(output, header.snapshot_count);
    write_value(output, header.seed);
    write_value(output, header.nominal_depth);
    write_value(output, header.a_level_count);
    write_value(output, header.b_level_count);
}

DatasetHeader read_header(std::istream& input) {
    std::array<char, 8> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!input || magic != kMagic) {
        throw std::runtime_error("not a DOQ benchmark dataset");
    }

    DatasetHeader header;
    header.version = read_value<std::uint32_t>(input);
    header.snapshot_count = read_value<std::uint64_t>(input);
    header.seed = read_value<std::uint64_t>(input);
    header.nominal_depth = read_value<std::uint32_t>(input);
    header.a_level_count = read_value<std::uint64_t>(input);
    header.b_level_count = read_value<std::uint64_t>(input);
    if (header.version != kFormatVersion) {
        throw std::runtime_error("unsupported benchmark dataset version");
    }
    return header;
}

std::uint64_t parse_positive(const char* text,
                             const char* name,
                             std::uint64_t maximum) {
    try {
        std::size_t consumed = 0;
        const std::string input(text);
        const std::uint64_t value = std::stoull(input, &consumed);
        if (consumed != input.size() || value == 0 || value > maximum) {
            throw std::invalid_argument("invalid");
        }
        return value;
    } catch (const std::exception&) {
        throw std::invalid_argument(
            std::string(name) + " must be in [1, " + std::to_string(maximum) + "]");
    }
}

std::uint32_t generate_depth(SplitMix64& random, std::uint32_t nominal_depth) {
    const std::uint64_t bucket = random.below(10'000);
    if (bucket < 50) {
        return 0; // 0.5% empty-side edge cases.
    }
    if (bucket < 1'500) {
        return 1U + static_cast<std::uint32_t>(
                        random.below(std::max<std::uint32_t>(1U, nominal_depth / 2U)));
    }
    if (bucket < 9'000) {
        const std::uint32_t low = std::max<std::uint32_t>(1U, nominal_depth * 3U / 4U);
        const std::uint32_t high = std::max(low, nominal_depth * 5U / 4U);
        return low + static_cast<std::uint32_t>(random.below(high - low + 1U));
    }
    const std::uint32_t low = std::max<std::uint32_t>(1U, nominal_depth * 5U / 4U + 1U);
    const std::uint32_t high = std::max(low, nominal_depth * 2U);
    return low + static_cast<std::uint32_t>(random.below(high - low + 1U));
}

std::int64_t generate_amount(SplitMix64& random) {
    const std::uint64_t bucket = random.below(10'000);
    if (bucket < 7'000) {
        return static_cast<std::int64_t>(1U + random.below(20U)) * 100;
    }
    if (bucket < 9'500) {
        return static_cast<std::int64_t>(21U + random.below(180U)) * 100;
    }
    return static_cast<std::int64_t>(201U + random.below(800U)) * 100;
}

std::int64_t next_key_gap(SplitMix64& random) {
    const std::uint64_t bucket = random.below(100U);
    return bucket < 80U ? 1 : (bucket < 95U ? 2 : 3);
}

void generate_dataset(const std::filesystem::path& path,
                      std::uint64_t snapshot_count,
                      std::uint32_t nominal_depth,
                      std::uint64_t seed) {
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot create dataset: " + path.string());
    }

    DatasetHeader header{kFormatVersion, snapshot_count, seed, nominal_depth, 0, 0};
    write_header(output, header);

    SplitMix64 random(seed);
    std::int64_t midpoint = 1'000'000;
    for (std::uint64_t index = 0; index < snapshot_count; ++index) {
        midpoint += static_cast<std::int64_t>(random.below(5U)) - 2;
        midpoint = std::max<std::int64_t>(midpoint, 100'000);
        const std::int64_t spread = 1 + static_cast<std::int64_t>(random.below(5U));
        const std::int64_t best_a = midpoint;
        const std::int64_t best_b = midpoint + spread;
        const std::uint32_t a_depth = generate_depth(random, nominal_depth);
        const std::uint32_t b_depth = generate_depth(random, nominal_depth);

        write_value(output, index + 1U);
        write_value(output, a_depth);
        write_value(output, b_depth);

        std::int64_t key = best_a;
        for (std::uint32_t level = 0; level < a_depth; ++level) {
            if (level != 0) {
                key -= next_key_gap(random);
            }
            write_value(output, key);
            write_value(output, generate_amount(random));
        }

        key = best_b;
        for (std::uint32_t level = 0; level < b_depth; ++level) {
            if (level != 0) {
                key += next_key_gap(random);
            }
            write_value(output, key);
            write_value(output, generate_amount(random));
        }

        header.a_level_count += a_depth;
        header.b_level_count += b_depth;
    }

    output.seekp(0);
    write_header(output, header);
}

struct LoadedBatch {
    std::vector<doq::Snapshot> cpu_snapshots;
    doq::CudaMetricsInputBatch cuda_input;
};

LoadedBatch read_batch(std::istream& input,
                       std::uint64_t count,
                       std::uint32_t maximum_depth,
                       std::size_t k,
                       bool preflattened) {
    LoadedBatch batch;
    batch.cpu_snapshots.reserve(static_cast<std::size_t>(count));
    if (preflattened) {
        batch.cuda_input.snapshots.reserve(static_cast<std::size_t>(count));
        if (k > std::numeric_limits<std::size_t>::max() /
                    (2 * static_cast<std::size_t>(count))) {
            throw std::length_error("preflattened benchmark batch is too large");
        }
        const std::size_t maximum_amounts =
            static_cast<std::size_t>(count) * 2 * k;
        batch.cuda_input.amounts.reserve(maximum_amounts);
    }
    for (std::uint64_t index = 0; index < count; ++index) {
        doq::Snapshot snapshot;
        snapshot.event_seq = read_value<std::uint64_t>(input);
        const std::uint32_t a_depth = read_value<std::uint32_t>(input);
        const std::uint32_t b_depth = read_value<std::uint32_t>(input);
        if (a_depth > maximum_depth || b_depth > maximum_depth) {
            throw std::runtime_error("dataset contains an invalid depth");
        }
        snapshot.a_levels.resize(a_depth);
        snapshot.b_levels.resize(b_depth);
        const std::uint64_t a_offset = batch.cuda_input.amounts.size();
        for (std::size_t level_index = 0;
             level_index < snapshot.a_levels.size();
             ++level_index) {
            doq::LevelSnapshot& level = snapshot.a_levels[level_index];
            level.key = read_value<std::int64_t>(input);
            level.amount = read_value<std::int64_t>(input);
            if (preflattened && level_index < k) {
                batch.cuda_input.amounts.push_back(level.amount);
            }
        }
        const std::uint64_t b_offset = batch.cuda_input.amounts.size();
        for (std::size_t level_index = 0;
             level_index < snapshot.b_levels.size();
             ++level_index) {
            doq::LevelSnapshot& level = snapshot.b_levels[level_index];
            level.key = read_value<std::int64_t>(input);
            level.amount = read_value<std::int64_t>(input);
            if (preflattened && level_index < k) {
                batch.cuda_input.amounts.push_back(level.amount);
            }
        }
        if (preflattened) {
            const std::uint32_t a_count = static_cast<std::uint32_t>(
                std::min(k, snapshot.a_levels.size()));
            const std::uint32_t b_count = static_cast<std::uint32_t>(
                std::min(k, snapshot.b_levels.size()));
            batch.cuda_input.snapshots.push_back(doq::CudaSnapshotDescriptor{
                snapshot.event_seq,
                a_offset,
                b_offset,
                a_count,
                b_count,
                snapshot.a_levels.empty() ? -1 : snapshot.a_levels.front().key,
                snapshot.b_levels.empty() ? -1 : snapshot.b_levels.front().key,
            });
            batch.cuda_input.record_snapshot_work(a_count, b_count);
        }
        batch.cpu_snapshots.push_back(std::move(snapshot));
    }
    return batch;
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

void mix_checksum(std::uint64_t& checksum, const doq::SnapshotMetrics& metrics) {
    const std::array<std::uint64_t, 10> values{{
        metrics.event_seq,
        static_cast<std::uint64_t>(metrics.a_extreme_key),
        static_cast<std::uint64_t>(metrics.b_extreme_key),
        static_cast<std::uint64_t>(metrics.a_amount_top_k),
        static_cast<std::uint64_t>(metrics.b_amount_top_k),
        static_cast<std::uint64_t>(metrics.key_gap),
        static_cast<std::uint64_t>(metrics.balance_num),
        static_cast<std::uint64_t>(metrics.balance_den),
        static_cast<std::uint64_t>(metrics.weighted_key_num),
        static_cast<std::uint64_t>(metrics.weighted_key_den),
    }};
    for (const std::uint64_t value : values) {
        checksum ^= value;
        checksum *= 1099511628211ULL;
    }
}

struct BenchmarkResult {
    double load_seconds{};
    double cpu_seconds{};
    double cuda_seconds{};
    doq::CudaMetricsTimings cuda_timings;
    double cuda_pipeline_wall_seconds{};
    std::uint64_t top_k_level_visits{};
    std::uint64_t checksum{1469598103934665603ULL};
    std::array<std::uint64_t, 5> cuda_kernel_mode_calls{};
};

doq::CudaSnapshotKernelMode parse_cuda_kernel_mode(const std::string& text) {
    if (text == "auto") {
        return doq::CudaSnapshotKernelMode::automatic;
    }
    if (text == "thread") {
        return doq::CudaSnapshotKernelMode::thread_per_snapshot;
    }
    if (text == "group4") {
        return doq::CudaSnapshotKernelMode::four_threads_per_snapshot;
    }
    if (text == "group8") {
        return doq::CudaSnapshotKernelMode::eight_threads_per_snapshot;
    }
    if (text == "group16") {
        return doq::CudaSnapshotKernelMode::sixteen_threads_per_snapshot;
    }
    if (text == "warp" || text == "group32") {
        return doq::CudaSnapshotKernelMode::warp_per_snapshot;
    }
    throw std::invalid_argument(
        "cuda_kernel_mode must be auto, thread, group4, group8, group16, or warp");
}

std::size_t measured_mode_index(doq::CudaSnapshotKernelMode mode) {
    switch (mode) {
    case doq::CudaSnapshotKernelMode::thread_per_snapshot:
        return 0;
    case doq::CudaSnapshotKernelMode::four_threads_per_snapshot:
        return 1;
    case doq::CudaSnapshotKernelMode::eight_threads_per_snapshot:
        return 2;
    case doq::CudaSnapshotKernelMode::sixteen_threads_per_snapshot:
        return 3;
    case doq::CudaSnapshotKernelMode::warp_per_snapshot:
        return 4;
    case doq::CudaSnapshotKernelMode::automatic:
        break;
    }
    throw std::invalid_argument("automatic mode must be resolved before counting");
}

BenchmarkResult run_benchmark(const std::filesystem::path& path,
                              const DatasetHeader& header,
                              std::size_t k,
                              std::uint64_t batch_size,
                              std::uint64_t repetitions,
                              bool preflattened,
                              doq::CudaSnapshotKernelMode kernel_mode,
                              bool overlap) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open dataset: " + path.string());
    }
    (void)read_header(input);

    BenchmarkResult result;
    std::uint64_t processed = 0;
    bool warmed_up = false;
    const std::uint32_t maximum_depth = header.nominal_depth * 2U;
    while (processed < header.snapshot_count) {
        const std::uint64_t current_size =
            std::min(batch_size, header.snapshot_count - processed);
        const auto load_start = std::chrono::steady_clock::now();
        LoadedBatch batch =
            read_batch(input, current_size, maximum_depth, k, preflattened);
        const auto load_end = std::chrono::steady_clock::now();
        result.load_seconds +=
            std::chrono::duration<double>(load_end - load_start).count();
        for (const doq::Snapshot& snapshot : batch.cpu_snapshots) {
            result.top_k_level_visits +=
                std::min(k, snapshot.a_levels.size()) +
                std::min(k, snapshot.b_levels.size());
        }
        doq::CudaSnapshotKernelMode resolved_mode = kernel_mode;
        if (kernel_mode == doq::CudaSnapshotKernelMode::automatic && preflattened) {
            resolved_mode =
                doq::select_cuda_snapshot_kernel_mode(batch.cuda_input);
        }

        if (!warmed_up) {
            // A full normal-size call brings the CUDA context and GPU clocks to a
            // representative state. It is deliberately excluded from timing.
            if (preflattened) {
                (void)doq::compute_snapshot_metrics_cuda_flat(
                    batch.cuda_input, nullptr, kernel_mode);
            } else {
                (void)doq::compute_snapshot_metrics_cuda(
                    batch.cpu_snapshots, k, nullptr, kernel_mode);
            }
            warmed_up = true;
        }

        if (overlap) {
            std::vector<doq::SnapshotMetrics> cpu_output;
            for (std::uint64_t repetition = 0;
                 repetition < repetitions;
                 ++repetition) {
                const auto cpu_start = std::chrono::steady_clock::now();
                cpu_output.clear();
                cpu_output.reserve(batch.cpu_snapshots.size());
                for (const doq::Snapshot& snapshot : batch.cpu_snapshots) {
                    cpu_output.push_back(doq::compute_snapshot_metrics(snapshot, k));
                }
                result.cpu_seconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - cpu_start).count();
                if (repetition == 0) {
                    for (const doq::SnapshotMetrics& metrics : cpu_output) {
                        mix_checksum(result.checksum, metrics);
                    }
                }
            }
            const auto pipeline_call_start = std::chrono::steady_clock::now();
            const doq::CudaPipelineBatchResult pipeline =
                doq::compute_snapshot_metrics_cuda_flat_pipeline(
                    batch.cuda_input,
                    static_cast<std::size_t>(repetitions),
                    kernel_mode);
            result.cuda_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - pipeline_call_start).count();
            result.cuda_pipeline_wall_seconds += pipeline.wall_seconds;
            result.cuda_timings.device_allocation_seconds +=
                pipeline.activity.device_allocation_seconds;
            result.cuda_timings.h2d_seconds += pipeline.activity.h2d_seconds;
            result.cuda_timings.kernel_seconds += pipeline.activity.kernel_seconds;
            result.cuda_timings.d2h_seconds += pipeline.activity.d2h_seconds;
            result.cuda_timings.host_output_allocation_seconds +=
                pipeline.activity.host_output_allocation_seconds;
            result.cuda_kernel_mode_calls[measured_mode_index(resolved_mode)] +=
                repetitions;
            if (cpu_output.size() != pipeline.output.size()) {
                throw std::runtime_error("CPU/CUDA pipeline output sizes differ");
            }
            for (std::size_t index = 0; index < cpu_output.size(); ++index) {
                if (!equal_metrics(cpu_output[index], pipeline.output[index])) {
                    throw std::runtime_error(
                        "CPU/CUDA pipeline mismatch at event_seq=" +
                        std::to_string(cpu_output[index].event_seq));
                }
            }
        } else for (std::uint64_t repetition = 0;
                    repetition < repetitions;
                    ++repetition) {
            const auto cpu_start = std::chrono::steady_clock::now();
            std::vector<doq::SnapshotMetrics> cpu_output;
            cpu_output.reserve(batch.cpu_snapshots.size());
            for (const doq::Snapshot& snapshot : batch.cpu_snapshots) {
                cpu_output.push_back(doq::compute_snapshot_metrics(snapshot, k));
            }
            const auto cpu_end = std::chrono::steady_clock::now();
            result.cpu_seconds +=
                std::chrono::duration<double>(cpu_end - cpu_start).count();

            const auto cuda_start = std::chrono::steady_clock::now();
            doq::CudaMetricsTimings cuda_timings;
            const std::vector<doq::SnapshotMetrics> cuda_output = preflattened
                ? doq::compute_snapshot_metrics_cuda_flat(batch.cuda_input,
                                                          &cuda_timings,
                                                          kernel_mode)
                : doq::compute_snapshot_metrics_cuda(batch.cpu_snapshots,
                                                     k,
                                                     &cuda_timings,
                                                     kernel_mode);
            const auto cuda_end = std::chrono::steady_clock::now();
            result.cuda_seconds +=
                std::chrono::duration<double>(cuda_end - cuda_start).count();
            result.cuda_timings.device_query_seconds +=
                cuda_timings.device_query_seconds;
            result.cuda_timings.flatten_seconds += cuda_timings.flatten_seconds;
            result.cuda_timings.device_allocation_seconds +=
                cuda_timings.device_allocation_seconds;
            result.cuda_timings.h2d_seconds += cuda_timings.h2d_seconds;
            result.cuda_timings.kernel_seconds += cuda_timings.kernel_seconds;
            result.cuda_timings.host_output_allocation_seconds +=
                cuda_timings.host_output_allocation_seconds;
            result.cuda_timings.d2h_seconds += cuda_timings.d2h_seconds;
            result.cuda_timings.device_free_seconds +=
                cuda_timings.device_free_seconds;
            result.cuda_timings.host_input_free_seconds +=
                cuda_timings.host_input_free_seconds;
            if (preflattened) {
                ++result.cuda_kernel_mode_calls[measured_mode_index(resolved_mode)];
            }

            if (cpu_output.size() != cuda_output.size()) {
                throw std::runtime_error("CPU/CUDA output sizes differ");
            }
            for (std::size_t index = 0; index < cpu_output.size(); ++index) {
                if (!equal_metrics(cpu_output[index], cuda_output[index])) {
                    throw std::runtime_error(
                        "CPU/CUDA mismatch at event_seq=" +
                        std::to_string(cpu_output[index].event_seq));
                }
                if (repetition == 0) {
                    mix_checksum(result.checksum, cpu_output[index]);
                }
            }
        }
        processed += current_size;
    }
    return result;
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        const std::filesystem::path dataset_path =
            argc > 1 ? argv[1] : "bench_data/datasets/snapshots_500k_d50.bin";
        const std::uint64_t snapshot_count =
            argc > 2 ? parse_positive(argv[2], "snapshot_count", 10'000'000)
                     : 500'000;
        const std::uint32_t nominal_depth = static_cast<std::uint32_t>(
            argc > 3 ? parse_positive(argv[3], "nominal_depth", 8'192) : 50);
        const std::size_t k = static_cast<std::size_t>(
            argc > 4 ? parse_positive(argv[4], "K", 16'384) : 50);
        const std::uint64_t batch_size =
            argc > 5 ? parse_positive(argv[5], "batch_size", 1'000'000) : 65'536;
        const std::uint64_t repetitions =
            argc > 6 ? parse_positive(argv[6], "repetitions", 20) : 3;
        const std::string cuda_input_mode = argc > 7 ? argv[7] : "nested";
        if (cuda_input_mode != "nested" && cuda_input_mode != "preflattened") {
            throw std::invalid_argument(
                "cuda_input_mode must be nested or preflattened");
        }
        const bool preflattened = cuda_input_mode == "preflattened";
        const std::string cuda_kernel_mode_text = argc > 8 ? argv[8] : "thread";
        const doq::CudaSnapshotKernelMode cuda_kernel_mode =
            parse_cuda_kernel_mode(cuda_kernel_mode_text);
        const std::string cuda_execution_mode = argc > 9 ? argv[9] : "serial";
        if (cuda_execution_mode != "serial" && cuda_execution_mode != "overlap") {
            throw std::invalid_argument(
                "cuda_execution_mode must be serial or overlap");
        }
        const bool overlap = cuda_execution_mode == "overlap";
        if (overlap && !preflattened) {
            throw std::invalid_argument(
                "overlap execution requires preflattened input");
        }

        std::string unavailable_reason;
        if (!doq::cuda_device_available(&unavailable_reason)) {
            throw std::runtime_error("CUDA unavailable: " + unavailable_reason);
        }

        bool reused = std::filesystem::exists(dataset_path);
        double generation_seconds = 0.0;
        if (!reused) {
            std::cout << "generating_dataset=" << dataset_path.string() << '\n';
            const auto start = std::chrono::steady_clock::now();
            generate_dataset(dataset_path, snapshot_count, nominal_depth, kDefaultSeed);
            generation_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
                    .count();
        }

        std::ifstream header_input(dataset_path, std::ios::binary);
        if (!header_input) {
            throw std::runtime_error("cannot open dataset: " + dataset_path.string());
        }
        const DatasetHeader header = read_header(header_input);
        if (header.snapshot_count != snapshot_count ||
            header.nominal_depth != nominal_depth || header.seed != kDefaultSeed) {
            throw std::runtime_error(
                "existing dataset configuration differs; use another path or remove it");
        }

        const BenchmarkResult result =
            run_benchmark(dataset_path,
                          header,
                          k,
                          batch_size,
                          repetitions,
                          preflattened,
                          cuda_kernel_mode,
                          overlap);
        const double work_items =
            static_cast<double>(header.snapshot_count) * repetitions;
        const double dataset_mib =
            static_cast<double>(std::filesystem::file_size(dataset_path)) /
            (1024.0 * 1024.0);
        const double profiled_cuda_seconds =
            result.cuda_timings.device_query_seconds +
            result.cuda_timings.flatten_seconds +
            result.cuda_timings.device_allocation_seconds +
            result.cuda_timings.h2d_seconds +
            result.cuda_timings.kernel_seconds +
            result.cuda_timings.host_output_allocation_seconds +
            result.cuda_timings.d2h_seconds +
            result.cuda_timings.device_free_seconds +
            result.cuda_timings.host_input_free_seconds;

        std::cout << std::fixed << std::setprecision(6)
                  << "dataset_path=" << dataset_path.string() << '\n'
                  << "dataset_reused=" << (reused ? "yes" : "no") << '\n'
                  << "dataset_mib=" << dataset_mib << '\n'
                  << "generation_seconds=" << generation_seconds << '\n'
                  << "seed=" << header.seed << '\n'
                  << "snapshots=" << header.snapshot_count << '\n'
                  << "a_levels=" << header.a_level_count << '\n'
                  << "b_levels=" << header.b_level_count << '\n'
                  << "top_k_level_visits_per_pass="
                  << result.top_k_level_visits << '\n'
                  << "nominal_depth=" << header.nominal_depth << '\n'
                  << "K=" << k << '\n'
                  << "batch_size=" << batch_size << '\n'
                  << "repetitions=" << repetitions << '\n'
                  << "cuda_input_mode=" << cuda_input_mode << '\n'
                  << "cuda_kernel_mode=" << cuda_kernel_mode_text << '\n'
                  << "cuda_execution_mode=" << cuda_execution_mode << '\n'
                  << "load_seconds=" << result.load_seconds << '\n'
                  << "cpu_seconds_total=" << result.cpu_seconds << '\n'
                  << "cpu_seconds_per_pass="
                  << result.cpu_seconds / repetitions << '\n'
                  << "cpu_snapshots_per_second="
                  << work_items / result.cpu_seconds << '\n'
                  << "cpu_level_visits_per_second="
                  << static_cast<double>(result.top_k_level_visits) * repetitions /
                         result.cpu_seconds
                  << '\n'
                  << "cuda_seconds_total=" << result.cuda_seconds << '\n'
                  << "cuda_seconds_per_pass="
                  << result.cuda_seconds / repetitions << '\n'
                  << "cuda_snapshots_per_second="
                  << work_items / result.cuda_seconds << '\n'
                  << "cuda_level_visits_per_second="
                  << static_cast<double>(result.top_k_level_visits) * repetitions /
                         result.cuda_seconds
                  << '\n'
                  << "cuda_device_query_seconds_per_pass="
                  << result.cuda_timings.device_query_seconds / repetitions << '\n'
                  << "cuda_flatten_seconds_per_pass="
                  << result.cuda_timings.flatten_seconds / repetitions << '\n'
                  << "cuda_device_allocation_seconds_per_pass="
                  << result.cuda_timings.device_allocation_seconds / repetitions << '\n'
                  << "cuda_h2d_seconds_per_pass="
                  << result.cuda_timings.h2d_seconds / repetitions << '\n'
                  << "cuda_kernel_seconds_per_pass="
                  << result.cuda_timings.kernel_seconds / repetitions << '\n'
                  << "cuda_host_output_allocation_seconds_per_pass="
                  << result.cuda_timings.host_output_allocation_seconds /
                         repetitions
                  << '\n'
                  << "cuda_d2h_seconds_per_pass="
                  << result.cuda_timings.d2h_seconds / repetitions << '\n'
                  << "cuda_device_free_seconds_per_pass="
                  << result.cuda_timings.device_free_seconds / repetitions << '\n'
                  << "cuda_host_input_free_seconds_per_pass="
                  << result.cuda_timings.host_input_free_seconds / repetitions
                  << '\n'
                  << "cuda_unattributed_seconds_per_pass="
                  << (overlap ? 0.0
                              : (result.cuda_seconds - profiled_cuda_seconds) /
                                    repetitions)
                  << '\n'
                  << "cuda_serialized_activity_seconds_per_pass="
                  << (result.cuda_timings.h2d_seconds +
                      result.cuda_timings.kernel_seconds +
                      result.cuda_timings.d2h_seconds) / repetitions
                  << '\n'
                  << "cuda_pipeline_wall_seconds_per_pass="
                  << result.cuda_pipeline_wall_seconds / repetitions << '\n'
                  << "cuda_overlap_saved_seconds_per_pass="
                  << (overlap
                          ? (result.cuda_timings.h2d_seconds +
                             result.cuda_timings.kernel_seconds +
                             result.cuda_timings.d2h_seconds -
                             result.cuda_pipeline_wall_seconds) /
                                repetitions
                          : 0.0)
                  << '\n'
                  << "cuda_speedup_vs_cpu="
                  << result.cpu_seconds / result.cuda_seconds << '\n'
                  << "cuda_thread_calls=" << result.cuda_kernel_mode_calls[0] << '\n'
                  << "cuda_group4_calls=" << result.cuda_kernel_mode_calls[1] << '\n'
                  << "cuda_group8_calls=" << result.cuda_kernel_mode_calls[2] << '\n'
                  << "cuda_group16_calls=" << result.cuda_kernel_mode_calls[3] << '\n'
                  << "cuda_group32_calls=" << result.cuda_kernel_mode_calls[4] << '\n'
                  << "cpu_cuda_match=yes\n"
                  << "checksum=" << result.checksum << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
