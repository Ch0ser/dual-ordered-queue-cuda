#include "io.hpp"
#include "metrics_cuda.hpp"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

// 本文件只负责命令行。匿名命名空间中的辅助函数仅供 main.cpp 使用，
// 不属于订单簿业务接口，也不会暴露给其他源文件。
namespace {

// ---------- 通用参数解析 ----------
// 命令行中的 M、K、batch_size 都要求是没有符号和多余字符的正整数。
std::uint64_t parse_positive_integer(std::string_view text, const char* name) {
    std::uint64_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || text.front() == '-' || text.front() == '+' ||
        result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
        value == 0) {
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    }
    return value;
}

// ---------- CUDA 默认参数 ----------
// 根据 K 估算单份输入大小，让一批数据接近 256 MiB，同时使用实验得到的
// 262144 上限，避免批次继续增大却不再提升吞吐。
std::size_t default_cuda_batch_size(std::size_t k) {
    // Keep the flattened CUDA input near 256 MiB.  The measured throughput
    // plateaus by 262144 snapshots, so larger batches only increase allocation
    // cost and peak memory without improving transfer efficiency.
    constexpr std::size_t target_input_bytes = 256ULL * 1024ULL * 1024ULL;
    constexpr std::size_t maximum_batch_size = 262144;
    const std::size_t bytes_per_snapshot =
        sizeof(doq::CudaSnapshotDescriptor) + 2 * k * sizeof(std::int64_t);
    const std::size_t budget_batch_size = target_input_bytes / bytes_per_snapshot;
    if (budget_batch_size == 0) {
        return 1;
    }
    return budget_batch_size < maximum_batch_size
               ? budget_batch_size
               : maximum_batch_size;
}

// 将用户输入的档位名称转换为程序内部的强类型枚举。
doq::CudaSnapshotKernelMode parse_cuda_kernel_mode(std::string_view text) {
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
        "kernel mode must be auto, thread, group4, group8, group16, or warp");
}

// ---------- 命令帮助 ----------
// 参数数量或命令名称错误时，统一打印三种合法用法。
void print_usage(const char* program) {
    std::cerr
        << "Usage:\n"
        << "  " << program
        << " replay <events.txt> <snapshot_interval_M> <output_directory>\n"
        << "  " << program
        << " metrics <snapshots.txt> <K> <metrics.csv>\n"
        << "  " << program
        << " metrics-cuda <snapshots.txt> <K> <metrics.csv> "
           "[batch_size] [kernel_mode]\n";
}

} // namespace

// ---------- 程序入口与命令分发 ----------
// main 不处理业务数据，只把命令行参数交给 io.cpp 的三个高层入口。
int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            print_usage(argv[0]);
            return 2;
        }

        const std::string_view command = argv[1];

        // replay：事件文件 -> Resolution、StateSummary 和 Snapshot。
        if (command == "replay") {
            if (argc != 5) {
                print_usage(argv[0]);
                return 2;
            }
            const std::uint64_t interval =
                parse_positive_integer(argv[3], "snapshot interval M");
            doq::replay_file(argv[2], interval, argv[4]);
            return 0;
        }

        // metrics：读取 Snapshot，用 CPU reference 计算指标。
        if (command == "metrics") {
            if (argc != 5) {
                print_usage(argv[0]);
                return 2;
            }
            const std::uint64_t k = parse_positive_integer(argv[3], "K");
            if (k > 16384) {
                throw std::invalid_argument("K must be in [1, 16384]");
            }
            doq::compute_metrics_file(argv[2], static_cast<std::size_t>(k), argv[4]);
            return 0;
        }

        // metrics-cuda：读取 Snapshot，按批送入 GPU 计算相同指标。
        if (command == "metrics-cuda") {
            if (argc < 5 || argc > 7) {
                print_usage(argv[0]);
                return 2;
            }
            const std::uint64_t k = parse_positive_integer(argv[3], "K");
            if (k > 16384) {
                throw std::invalid_argument("K must be in [1, 16384]");
            }
            const std::uint64_t batch_size =
                argc >= 6
                    ? parse_positive_integer(argv[5], "batch size")
                    : default_cuda_batch_size(static_cast<std::size_t>(k));
            const doq::CudaSnapshotKernelMode kernel_mode =
                argc == 7
                    ? parse_cuda_kernel_mode(argv[6])
                    : doq::CudaSnapshotKernelMode::automatic;
            doq::compute_metrics_cuda_file(argv[2],
                                           static_cast<std::size_t>(k),
                                           static_cast<std::size_t>(batch_size),
                                           argv[4],
                                           kernel_mode);
            return 0;
        }

        print_usage(argv[0]);
        return 2;
    } catch (const std::exception& error) {
        // 文件格式、参数范围和运行期错误统一转成非零退出码，便于脚本判断。
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
