#include "metrics_cuda.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint64_t kMaximumTotalLevels = 8ULL * 1024ULL * 1024ULL;
constexpr std::uint64_t kTargetTimedLevelVisits = 100ULL * 1024ULL * 1024ULL;
constexpr std::size_t kMinimumLaunches = 20;
constexpr std::size_t kMaximumLaunches = 5000;
constexpr std::size_t kWarmupLaunches = 10;
constexpr std::size_t kSampleCount = 5;

constexpr std::array<doq::CudaSnapshotKernelMode, 5> kModes{
    doq::CudaSnapshotKernelMode::thread_per_snapshot,
    doq::CudaSnapshotKernelMode::four_threads_per_snapshot,
    doq::CudaSnapshotKernelMode::eight_threads_per_snapshot,
    doq::CudaSnapshotKernelMode::sixteen_threads_per_snapshot,
    doq::CudaSnapshotKernelMode::warp_per_snapshot,
};

// A non-monotonic order avoids giving wider groups a systematic clock or
// thermal advantage.
constexpr std::array<std::size_t, 5> kTimingOrder{4, 0, 3, 1, 2};

struct DepthPair {
    std::size_t a{};
    std::size_t b{};
};

struct MixedProfile {
    const char* name{};
    std::vector<std::size_t> work_pattern;
    bool clustered{};
};

std::int64_t deterministic_amount(std::size_t level, std::int64_t bias) {
    return bias + static_cast<std::int64_t>(level % 13U);
}

std::int64_t deterministic_sum(std::size_t depth, std::int64_t bias) {
    const std::size_t complete_cycles = depth / 13U;
    const std::size_t remainder = depth % 13U;
    const std::size_t remainder_sequence_sum =
        remainder == 0 ? 0 : remainder * (remainder - 1U) / 2U;
    return static_cast<std::int64_t>(complete_cycles) * (13 * bias + 78) +
           static_cast<std::int64_t>(remainder) * bias +
           static_cast<std::int64_t>(remainder_sequence_sum);
}

DepthPair balanced_work(std::size_t work) {
    return DepthPair{work / 2U, work - work / 2U};
}

std::uint64_t total_levels(const std::vector<DepthPair>& depths) {
    std::uint64_t total = 0;
    for (const DepthPair depth : depths) {
        total += depth.a + depth.b;
    }
    return total;
}

doq::CudaMetricsInputBatch make_input(const std::vector<DepthPair>& depths) {
    doq::CudaMetricsInputBatch input;
    input.snapshots.reserve(depths.size());

    std::size_t total_a = 0;
    std::size_t total_b = 0;
    for (const DepthPair depth : depths) {
        if (depth.a > std::numeric_limits<std::uint32_t>::max() ||
            depth.b > std::numeric_limits<std::uint32_t>::max()) {
            throw std::length_error("benchmark depth exceeds descriptor range");
        }
        total_a += depth.a;
        total_b += depth.b;
    }
    input.amounts.reserve(total_a + total_b);

    for (std::size_t snapshot = 0; snapshot < depths.size(); ++snapshot) {
        const DepthPair depth = depths[snapshot];
        const std::uint64_t a_offset = input.amounts.size();
        for (std::size_t level = 0; level < depth.a; ++level) {
            input.amounts.push_back(deterministic_amount(level, 1));
        }
        const std::uint64_t b_offset = input.amounts.size();
        for (std::size_t level = 0; level < depth.b; ++level) {
            input.amounts.push_back(deterministic_amount(level, 2));
        }
        input.snapshots.push_back(doq::CudaSnapshotDescriptor{
            static_cast<std::uint64_t>(snapshot + 1U),
            a_offset,
            b_offset,
            static_cast<std::uint32_t>(depth.a),
            static_cast<std::uint32_t>(depth.b),
            depth.a == 0 ? -1 : 1'000'000,
            depth.b == 0 ? -1 : 1'000'100,
        });
        input.record_snapshot_work(static_cast<std::uint32_t>(depth.a),
                                   static_cast<std::uint32_t>(depth.b));
    }
    return input;
}

void verify_output(const doq::CudaMetricsInputBatch& input,
                   const std::vector<doq::SnapshotMetrics>& output,
                   const char* mode) {
    if (output.size() != input.snapshots.size()) {
        throw std::runtime_error(std::string(mode) + " output size mismatch");
    }
    for (std::size_t index = 0; index < output.size(); ++index) {
        const doq::CudaSnapshotDescriptor& descriptor =
            input.snapshots.data()[index];
        const std::int64_t a_sum = deterministic_sum(descriptor.a_count, 1);
        const std::int64_t b_sum = deterministic_sum(descriptor.b_count, 2);
        const bool empty_side = descriptor.a_count == 0 || descriptor.b_count == 0;
        const std::int64_t expected_gap = empty_side ? -1 : 100;
        const std::int64_t expected_weighted =
            empty_side ? -1 : 1'000'100 * a_sum + 1'000'000 * b_sum;
        const std::int64_t expected_weighted_den =
            empty_side ? 0 : a_sum + b_sum;
        const doq::SnapshotMetrics& metrics = output[index];
        if (metrics.event_seq != index + 1U ||
            metrics.a_extreme_key != descriptor.a_extreme_key ||
            metrics.b_extreme_key != descriptor.b_extreme_key ||
            metrics.a_amount_top_k != a_sum ||
            metrics.b_amount_top_k != b_sum ||
            metrics.key_gap != expected_gap ||
            metrics.balance_num != a_sum - b_sum ||
            metrics.balance_den != a_sum + b_sum ||
            metrics.weighted_key_num != expected_weighted ||
            metrics.weighted_key_den != expected_weighted_den) {
            throw std::runtime_error(
                std::string(mode) + " result mismatch at snapshot " +
                std::to_string(index));
        }
    }
}

std::size_t choose_launch_count(std::uint64_t levels) {
    const std::uint64_t required =
        (kTargetTimedLevelVisits + levels - 1U) / levels;
    return static_cast<std::size_t>(std::clamp<std::uint64_t>(
        required, kMinimumLaunches, kMaximumLaunches));
}

std::array<double, kModes.size()> measure_modes(
    const doq::CudaMetricsInputBatch& input,
    std::size_t launches) {
    std::array<double, kModes.size()> microseconds{};
    for (const std::size_t mode_index : kTimingOrder) {
        const doq::CudaSnapshotKernelMode mode = kModes[mode_index];
        const auto output =
            doq::compute_snapshot_metrics_cuda_flat(input, nullptr, mode);
        verify_output(input, output, doq::cuda_snapshot_kernel_mode_name(mode));
        const doq::CudaKernelBenchmarkStats stats =
            doq::benchmark_snapshot_metrics_cuda_kernel(
                input, mode, kWarmupLaunches, launches, kSampleCount);
        microseconds[mode_index] =
            stats.median_milliseconds_per_launch * 1000.0;
    }
    return microseconds;
}

double measure_one_mode(const doq::CudaMetricsInputBatch& input,
                        doq::CudaSnapshotKernelMode mode,
                        std::uint64_t levels) {
    const auto output =
        doq::compute_snapshot_metrics_cuda_flat(input, nullptr, mode);
    verify_output(input, output, doq::cuda_snapshot_kernel_mode_name(mode));
    const doq::CudaKernelBenchmarkStats stats =
        doq::benchmark_snapshot_metrics_cuda_kernel(
            input,
            mode,
            kWarmupLaunches,
            choose_launch_count(levels),
            kSampleCount);
    return stats.median_milliseconds_per_launch * 1000.0;
}

std::size_t selected_mode_index(double average_work) {
    if (average_work <= 96.0) {
        return 1;
    }
    if (average_work <= 448.0) {
        return 2;
    }
    if (average_work < 1024.0) {
        return 3;
    }
    return 4;
}

std::size_t winner_index(const std::array<double, kModes.size()>& values) {
    return static_cast<std::size_t>(
        std::distance(values.begin(),
                      std::min_element(values.begin(), values.end())));
}

void print_timings(const std::array<double, kModes.size()>& values) {
    for (const double value : values) {
        std::cout << ',' << value;
    }
}

void run_balanced_scan(bool boundary_scan) {
    const std::vector<std::size_t> snapshot_counts = boundary_scan
        ? std::vector<std::size_t>{1, 4, 32, 256, 4096, 65536}
        : std::vector<std::size_t>{1, 32, 256, 4096, 65536};
    const std::vector<std::size_t> depths = boundary_scan
        ? std::vector<std::size_t>{
              32, 40, 48, 56, 64, 72, 80, 96,
              112, 128, 160, 192, 224, 256, 384, 512}
        : std::vector<std::size_t>{
              1, 4, 8, 16, 24, 32, 48, 64, 80, 96, 128, 256,
              1024, 4096, 16384};

    std::cout << "snapshots,depth_per_side,effective_levels_per_snapshot,"
                 "total_levels,launches_per_sample,thread_us,group4_us,"
                 "group8_us,group16_us,group32_us,winner,winner_us,"
                 "thread_over_winner\n";
    for (const std::size_t snapshot_count : snapshot_counts) {
        for (const std::size_t depth : depths) {
            const std::uint64_t levels =
                static_cast<std::uint64_t>(snapshot_count) * depth * 2ULL;
            if (levels > kMaximumTotalLevels) {
                continue;
            }
            const std::vector<DepthPair> shapes(
                snapshot_count, DepthPair{depth, depth});
            doq::CudaMetricsInputBatch input = make_input(shapes);
            const std::size_t launches = choose_launch_count(levels);
            const auto values = measure_modes(input, launches);
            const std::size_t winner = winner_index(values);

            std::cout << snapshot_count << ',' << depth << ',' << depth * 2U
                      << ',' << levels << ',' << launches;
            print_timings(values);
            std::cout << ',' << doq::cuda_snapshot_kernel_mode_name(kModes[winner])
                      << ',' << values[winner] << ',' << values.front() / values[winner]
                      << '\n';
        }
    }
}

void run_asymmetric_scan() {
    constexpr std::array<std::size_t, 3> snapshot_counts{32, 256, 4096};
    constexpr std::array<std::size_t, 9> work_values{
        64, 96, 112, 256, 448, 512, 768, 1024, 2048};

    std::cout << "snapshots,shape,a_depth,b_depth,work,total_levels,"
                 "launches_per_sample,thread_us,group4_us,group8_us,"
                 "group16_us,group32_us,winner,selected,selected_over_winner,"
                 "shape_work,shape_selected,shape_over_winner\n";
    for (const std::size_t snapshot_count : snapshot_counts) {
        for (const std::size_t work : work_values) {
            const std::array<std::pair<const char*, DepthPair>, 3> shapes{{
                {"balanced", balanced_work(work)},
                {"three_to_one", DepthPair{work * 3U / 4U,
                                             work - work * 3U / 4U}},
                {"one_sided", DepthPair{work, 0}},
            }};
            for (const auto& named_shape : shapes) {
                const std::uint64_t levels =
                    static_cast<std::uint64_t>(snapshot_count) * work;
                if (levels > kMaximumTotalLevels) {
                    continue;
                }
                const std::vector<DepthPair> repeated(
                    snapshot_count, named_shape.second);
                doq::CudaMetricsInputBatch input = make_input(repeated);
                const std::size_t launches = choose_launch_count(levels);
                const auto values = measure_modes(input, launches);
                const std::size_t winner = winner_index(values);
                const std::size_t selected = selected_mode_index(
                    static_cast<double>(work));
                const double shape_work = 2.0 * static_cast<double>(
                    std::max(named_shape.second.a, named_shape.second.b));
                const std::size_t shape_selected =
                    selected_mode_index(shape_work);

                std::cout << snapshot_count << ',' << named_shape.first << ','
                          << named_shape.second.a << ',' << named_shape.second.b
                          << ',' << work << ',' << levels << ',' << launches;
                print_timings(values);
                std::cout << ','
                          << doq::cuda_snapshot_kernel_mode_name(kModes[winner])
                          << ','
                          << doq::cuda_snapshot_kernel_mode_name(kModes[selected])
                          << ',' << values[selected] / values[winner]
                          << ',' << shape_work << ','
                          << doq::cuda_snapshot_kernel_mode_name(
                                 kModes[shape_selected])
                          << ',' << values[shape_selected] / values[winner]
                          << '\n';
            }
        }
    }
}

std::vector<DepthPair> make_mixed_shapes(const MixedProfile& profile,
                                         std::size_t snapshot_count) {
    std::vector<DepthPair> shapes;
    shapes.reserve(snapshot_count);
    for (std::size_t index = 0; index < snapshot_count; ++index) {
        std::size_t pattern_index = index % profile.work_pattern.size();
        if (profile.clustered) {
            pattern_index = std::min(
                profile.work_pattern.size() - 1U,
                index * profile.work_pattern.size() / snapshot_count);
        }
        shapes.push_back(balanced_work(profile.work_pattern[pattern_index]));
    }
    return shapes;
}

void run_mixed_scan() {
    std::vector<std::size_t> one_deep_per_32(32, 64);
    one_deep_per_32.back() = 8192;
    std::vector<std::size_t> ten_percent_deep(10, 64);
    ten_percent_deep.back() = 2048;
    const std::vector<MixedProfile> profiles{
        {"alternating_64_1024", {64, 1024}, false},
        {"clustered_64_1024", {64, 1024}, true},
        {"one_deep_per_32", one_deep_per_32, false},
        {"ten_percent_deep", ten_percent_deep, false},
        {"three_bands", {64, 256, 1024}, false},
        {"threshold_96_112", {96, 112}, false},
    };
    constexpr std::array<std::size_t, 3> snapshot_counts{256, 4096, 32768};

    std::cout << "snapshots,profile,average_work,max_work,total_levels,"
                 "launches_per_sample,thread_us,group4_us,group8_us,"
                 "group16_us,group32_us,winner,selected,selected_over_winner,"
                 "weighted_work,weighted_selected,weighted_over_winner,"
                 "bucket_count,bucketed_us,bucketed_over_winner\n";
    for (const std::size_t snapshot_count : snapshot_counts) {
        for (const MixedProfile& profile : profiles) {
            const std::vector<DepthPair> shapes =
                make_mixed_shapes(profile, snapshot_count);
            const std::uint64_t levels = total_levels(shapes);
            if (levels > kMaximumTotalLevels) {
                continue;
            }
            const double average_work =
                static_cast<double>(levels) / static_cast<double>(snapshot_count);
            long double squared_work_sum = 0.0L;
            for (const DepthPair shape : shapes) {
                const long double work =
                    static_cast<long double>(shape.a + shape.b);
                squared_work_sum += work * work;
            }
            const double weighted_work = static_cast<double>(
                squared_work_sum / static_cast<long double>(levels));
            const std::size_t max_work = *std::max_element(
                profile.work_pattern.begin(), profile.work_pattern.end());
            doq::CudaMetricsInputBatch input = make_input(shapes);
            const std::size_t launches = choose_launch_count(levels);
            const auto values = measure_modes(input, launches);
            const std::size_t winner = winner_index(values);
            const std::size_t selected = selected_mode_index(average_work);
            const std::size_t weighted_selected =
                selected_mode_index(weighted_work);

            std::array<std::vector<DepthPair>, kModes.size()> buckets;
            for (const DepthPair shape : shapes) {
                const std::size_t bucket = selected_mode_index(
                    static_cast<double>(shape.a + shape.b));
                buckets[bucket].push_back(shape);
            }
            std::size_t bucket_count = 0;
            double bucketed_microseconds = 0.0;
            for (std::size_t bucket = 1; bucket < buckets.size(); ++bucket) {
                if (buckets[bucket].empty()) {
                    continue;
                }
                ++bucket_count;
                const std::uint64_t bucket_levels = total_levels(buckets[bucket]);
                doq::CudaMetricsInputBatch bucket_input =
                    make_input(buckets[bucket]);
                bucketed_microseconds += measure_one_mode(
                    bucket_input, kModes[bucket], bucket_levels);
            }

            std::cout << snapshot_count << ',' << profile.name << ','
                      << average_work << ',' << max_work << ',' << levels << ','
                      << launches;
            print_timings(values);
            std::cout << ','
                      << doq::cuda_snapshot_kernel_mode_name(kModes[winner])
                      << ','
                      << doq::cuda_snapshot_kernel_mode_name(kModes[selected])
                      << ',' << values[selected] / values[winner]
                      << ',' << weighted_work << ','
                      << doq::cuda_snapshot_kernel_mode_name(
                             kModes[weighted_selected])
                      << ',' << values[weighted_selected] / values[winner]
                      << ',' << bucket_count << ',' << bucketed_microseconds
                      << ',' << bucketed_microseconds / values[winner] << '\n';
        }
    }
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        std::string unavailable_reason;
        if (!doq::cuda_device_available(&unavailable_reason)) {
            throw std::runtime_error("CUDA unavailable: " + unavailable_reason);
        }

        std::cout << std::fixed << std::setprecision(6);
        const std::string command = argc > 1 ? argv[1] : "balanced";
        if (command == "asymmetric") {
            run_asymmetric_scan();
        } else if (command == "mixed") {
            run_mixed_scan();
        } else if (command == "balanced" || command == "boundary") {
            run_balanced_scan(command == "boundary");
        } else {
            throw std::invalid_argument(
                "benchmark mode must be balanced, boundary, asymmetric, or mixed");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
