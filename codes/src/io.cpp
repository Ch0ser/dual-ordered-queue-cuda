#include "io.hpp"

#include "metrics.hpp"
#include "metrics_cuda.hpp"
#include "order_book.hpp"
#include "types.hpp"

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace doq {
namespace {

// ---------- 基础文本解析 ----------
// 将文本字段转换为定长整数，并拒绝空字段、溢出和尾随字符。
constexpr std::uint64_t kMaximumEventCount = 10'000'000;
constexpr std::int64_t kMinimumInputValue = -1'000'000'000;
constexpr std::int64_t kMaximumInputValue = 1'000'000'000;

// 分词函数，按空格、制表符和换行符，把一行文本拆成多个字段。
std::vector<std::string_view> split_words(const std::string& line) {
    std::vector<std::string_view> words;
    std::size_t position = 0;
    while (position < line.size()) {
        while (position < line.size() &&
               (line[position] == ' ' || line[position] == '\t' ||
                line[position] == '\r' || line[position] == '\n')) {
            ++position;
        }
        if (position == line.size()) {
            break;
        }
        const std::size_t start = position;
        while (position < line.size() && line[position] != ' ' &&
               line[position] != '\t' && line[position] != '\r' &&
               line[position] != '\n') {
            ++position;
        }
        words.emplace_back(line.data() + start, position - start);
    }
    return words;
}

// 把文本转换成64位无符号整数。
std::uint64_t parse_u64(std::string_view text, const std::string& description) {
    if (text.empty() || text.front() == '-' || text.front() == '+') {
        throw std::runtime_error("invalid " + description + ": " + std::string(text));
    }
    std::uint64_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        throw std::runtime_error("invalid " + description + ": " + std::string(text));
    }
    return value;
}

// 把文本转换成64位有符号整数。
std::int64_t parse_i64(std::string_view text, const std::string& description) {
    if (text.empty() || text.front() == '+') {
        throw std::runtime_error("invalid " + description + ": " + std::string(text));
    }
    std::int64_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        throw std::runtime_error("invalid " + description + ": " + std::string(text));
    }
    return value;
}

// 解析文件第一行，并要求这一行只能有一个整数。
std::uint64_t parse_single_count_line(const std::string& line,
                                      const std::string& description) {
    const auto words = split_words(line);
    if (words.size() != 1) {
        throw std::runtime_error(description + " line must contain exactly one integer");
    }
    return parse_u64(words[0], description);
}

// 把事件文件的一行文本转换成Event对象。
Event parse_event_line(const std::string& line, std::uint64_t line_number) {
    // 这里只检查“文件格式是否合法”。例如 key=0 虽然会被 OrderBook 业务拒绝，
    // 但它仍是格式正确的 INSERT，不能在解析层直接丢掉。
    const auto words = split_words(line);
    const std::string context = "events.txt line " + std::to_string(line_number);
    if (words.empty()) {
        throw std::runtime_error(context + " is empty");
    }

    Event event;
    if (words[0] == "I") {
        if (words.size() != 6) {
            throw std::runtime_error(context + " must be: I seq item_id side key amount");
        }
        event.op = Op::Insert;
        event.seq = parse_u64(words[1], context + " seq");
        event.item_id = parse_u64(words[2], context + " item_id");
        if (words[3] == "A") {
            event.side = Side::A;
        } else if (words[3] == "B") {
            event.side = Side::B;
        } else {
            throw std::runtime_error(context + " side must be A or B");
        }
        event.key = parse_i64(words[4], context + " key");
        event.amount = parse_i64(words[5], context + " amount");
        if (event.key < kMinimumInputValue || event.key > kMaximumInputValue ||
            event.amount < kMinimumInputValue || event.amount > kMaximumInputValue) {
            throw std::runtime_error(context + " key/amount is outside [-10^9, 10^9]");
        }
    } else if (words[0] == "R") {
        if (words.size() != 3) {
            throw std::runtime_error(context + " must be: R seq item_id");
        }
        event.op = Op::Remove;
        event.seq = parse_u64(words[1], context + " seq");
        event.item_id = parse_u64(words[2], context + " item_id");
    } else {
        throw std::runtime_error(context + " operation must be I or R");
    }

    if (event.item_id == 0) {
        throw std::runtime_error(context + " item_id must be at least 1");
    }
    return event;
}

// ---------- 输出序列化 ----------
// 检查输出文件是否成功打开。
void ensure_output_stream(const std::ofstream& stream,
                          const std::filesystem::path& path) {
    if (!stream) {
        throw std::runtime_error("cannot open output file: " + path.string());
    }
}

// 把Resolution对象写入输出流。
void write_resolution(std::ostream& output, const Resolution& resolution) {
    output << resolution.event_seq << ',' << resolution.current_item_id << ','
           << resolution.resident_item_id << ',' << resolution.key << ','
           << resolution.amount << '\n';
}

// 把StateSummary对象写入输出流。
void write_state(std::ostream& output,
                 const StateSummary& state,
                 ApplyStatus status) {
    output << state.event_seq << ',' << state.a_extreme_key << ','
           << state.b_extreme_key << ',' << state.a_amount << ','
           << state.b_amount << ',' << state.active_count << ','
           << to_string(status) << '\n';
}

// 把Snapshot对象写入输出流。
void write_snapshot(std::ostream& output, const Snapshot& snapshot) {
    output << snapshot.event_seq << ' ' << snapshot.a_levels.size() << ' '
           << snapshot.b_levels.size() << '\n';
    for (const LevelSnapshot& level : snapshot.a_levels) {
        output << level.key << ' ' << level.amount << '\n';
    }
    for (const LevelSnapshot& level : snapshot.b_levels) {
        output << level.key << ' ' << level.amount << '\n';
    }
}

// SnapshotMetrics 对象写入 CSV 输出流 
void write_metrics(std::ostream& output, const SnapshotMetrics& metrics) {
    output << metrics.event_seq << ',' << metrics.a_extreme_key << ','
           << metrics.b_extreme_key << ',' << metrics.a_amount_top_k << ','
           << metrics.b_amount_top_k << ',' << metrics.key_gap << ','
           << metrics.balance_num << ',' << metrics.balance_den << ','
           << metrics.weighted_key_num << ',' << metrics.weighted_key_den << '\n';
}

// Snapshot level 解析和顺序校验函数
// 把Snapshot中的一行key amount转换成LevelSnapshot，并检查顺序
// A 的 key 必须递减、B 的 key 必须递增，每个 level 的 key/amount 都为正。
LevelSnapshot parse_level_line(const std::string& line,
                               std::uint64_t line_number,
                               Side side,
                               bool has_previous,
                               std::int64_t previous_key) {
    const auto words = split_words(line);
    const std::string context = "snapshots.txt line " + std::to_string(line_number);
    if (words.size() != 2) {
        throw std::runtime_error(context + " must be: key amount");
    }
    LevelSnapshot level{
        parse_i64(words[0], context + " key"),
        parse_i64(words[1], context + " amount"),
    };
    if (level.key <= 0 || level.amount <= 0) {
        throw std::runtime_error(context + " key and amount must be positive");
    }
    if (has_previous) {
        // Snapshot 文件本身必须已经按 A 递减、B 递增排列。
        const bool ordered = side == Side::A ? previous_key > level.key
                                              : previous_key < level.key;
        if (!ordered) {
            throw std::runtime_error(
                context + (side == Side::A
                               ? " A keys must be strictly decreasing"
                               : " B keys must be strictly increasing"));
        }
    }
    return level;
}

// 判断一行除了空格外是否还有其他内容。
bool contains_non_whitespace(const std::string& line) {
    return !split_words(line).empty();
}

// ---------- Snapshot 批量读取核心 ----------
// 同一套解析逻辑可输出普通 Snapshot，也可直接输出 CUDA 连续布局。
// 读取整个snapshots.txt，完成格式校验，每积累一个batch就交给调用者处理。
//检查batch_size
// → 打开文件
// → 读取快照数量S
// → 循环读取每份Snapshot
//     → 解析event_seq、NA、NB
//     → 读取A侧level
//     → 读取B侧level
//     → 检查排序和A.max < B.min
//     → 放入当前batch
// → batch满时调用consume_batch
// → 处理最后一个不足batch_size的批次
// → 检查文件后面没有额外数据
template <bool FlatCudaInput, typename BatchConsumer>
void read_snapshot_batches_impl(const std::filesystem::path& snapshots_path,
                                std::size_t batch_size,
                                std::size_t k,
                                BatchConsumer consume_batch) {
    // 同一套解析/校验支持两种输出：
    // FlatCudaInput=false 构造完整 Snapshot，供 CPU reference 使用；
    // FlatCudaInput=true 直接构造 CUDA 连续布局。
    if (batch_size == 0 || batch_size > kMaximumEventCount) {
        throw std::invalid_argument("batch size must be in [1, 10000000]");
    }

    std::ifstream input(snapshots_path);
    if (!input) {
        throw std::runtime_error("cannot open snapshots file: " + snapshots_path.string());
    }

    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("snapshots file is empty");
    }
    const std::uint64_t snapshot_count =
        parse_single_count_line(line, "snapshot count S");
    if (snapshot_count == 0 || snapshot_count > kMaximumEventCount) {
        throw std::runtime_error("snapshot count S must be in [1, 10000000]");
    }

    using Batch = std::conditional_t<FlatCudaInput,
                                     CudaMetricsInputBatch,
                                     std::vector<Snapshot>>;
    Batch batch;
    const std::size_t reserved_snapshots = static_cast<std::size_t>(
        std::min<std::uint64_t>(snapshot_count, batch_size));
    if constexpr (FlatCudaInput) {
        constexpr std::size_t kMaximumInitialPinnedSnapshots = 1'000'000;
        batch.snapshots.reserve(
            std::min(reserved_snapshots, kMaximumInitialPinnedSnapshots));
        constexpr std::size_t kMaximumInitialPinnedAmounts = 33'554'432;
        const std::size_t upper_amount_count =
            k != 0 && reserved_snapshots <=
                          std::numeric_limits<std::size_t>::max() / (2 * k)
                ? reserved_snapshots * 2 * k
                : kMaximumInitialPinnedAmounts;
        const std::size_t initial_amount_capacity =
            std::min(upper_amount_count, kMaximumInitialPinnedAmounts);
        batch.amounts.reserve(initial_amount_capacity);
    } else {
        batch.reserve(reserved_snapshots);
    }
    std::uint64_t line_number = 1;
    std::uint64_t previous_event_seq = 0;
    bool has_previous_event_seq = false;
    for (std::uint64_t snapshot_index = 0;
         snapshot_index < snapshot_count;
         ++snapshot_index) {
        if (!std::getline(input, line)) {
            throw std::runtime_error("snapshots file ended before snapshot " +
                                     std::to_string(snapshot_index + 1));
        }
        ++line_number;
        const auto header = split_words(line);
        if (header.size() != 3) {
            throw std::runtime_error("snapshots.txt line " +
                                     std::to_string(line_number) +
                                     " must be: event_seq NA NB");
        }

        const std::uint64_t event_seq =
            parse_u64(header[0], "snapshot event_seq");
        const std::uint64_t a_depth = parse_u64(header[1], "snapshot NA");
        const std::uint64_t b_depth = parse_u64(header[2], "snapshot NB");
        if (a_depth > kMaximumEventCount || b_depth > kMaximumEventCount) {
            throw std::runtime_error("snapshot depth exceeds the maximum event count");
        }
        if (has_previous_event_seq && event_seq <= previous_event_seq) {
            throw std::runtime_error(
                "snapshot event_seq values must be strictly increasing");
        }
        previous_event_seq = event_seq;
        has_previous_event_seq = true;

        Snapshot snapshot;
        if constexpr (!FlatCudaInput) {
            snapshot.event_seq = event_seq;
            snapshot.a_levels.reserve(static_cast<std::size_t>(a_depth));
            snapshot.b_levels.reserve(static_cast<std::size_t>(b_depth));
        }
        std::uint64_t a_offset = 0;
        std::uint64_t b_offset = 0;
        if constexpr (FlatCudaInput) {
            a_offset = static_cast<std::uint64_t>(batch.amounts.size());
        }
        std::int64_t a_extreme_key = -1;
        std::int64_t b_extreme_key = -1;
        std::int64_t previous_key = 0;
        for (std::uint64_t i = 0; i < a_depth; ++i) {
            if (!std::getline(input, line)) {
                throw std::runtime_error("snapshots file ended inside A levels");
            }
            ++line_number;
            const LevelSnapshot level = parse_level_line(
                line, line_number, Side::A, i != 0, previous_key);
            if (i == 0) {
                a_extreme_key = level.key;
            }
            if constexpr (FlatCudaInput) {
                if (i < k) {
                    batch.amounts.push_back(level.amount);
                }
            } else {
                snapshot.a_levels.push_back(level);
            }
            previous_key = level.key;
        }

        if constexpr (FlatCudaInput) {
            b_offset = static_cast<std::uint64_t>(batch.amounts.size());
        }
        previous_key = 0;
        for (std::uint64_t i = 0; i < b_depth; ++i) {
            if (!std::getline(input, line)) {
                throw std::runtime_error("snapshots file ended inside B levels");
            }
            ++line_number;
            const LevelSnapshot level = parse_level_line(
                line, line_number, Side::B, i != 0, previous_key);
            if (i == 0) {
                b_extreme_key = level.key;
            }
            if constexpr (FlatCudaInput) {
                if (i < k) {
                    batch.amounts.push_back(level.amount);
                }
            } else {
                snapshot.b_levels.push_back(level);
            }
            previous_key = level.key;
        }

        if (a_extreme_key != -1 && b_extreme_key != -1 &&
            a_extreme_key >= b_extreme_key) {
            throw std::runtime_error("snapshot violates A.max_key < B.min_key");
        }

        if constexpr (FlatCudaInput) {
            batch.snapshots.push_back(CudaSnapshotDescriptor{
                event_seq,
                a_offset,
                b_offset,
                static_cast<std::uint32_t>(std::min<std::uint64_t>(k, a_depth)),
                static_cast<std::uint32_t>(std::min<std::uint64_t>(k, b_depth)),
                a_extreme_key,
                b_extreme_key,
            });
            batch.record_snapshot_work(
                static_cast<std::uint32_t>(std::min<std::uint64_t>(k, a_depth)),
                static_cast<std::uint32_t>(std::min<std::uint64_t>(k, b_depth)));
            if (batch.snapshots.size() == batch_size) {
                consume_batch(batch);
                batch.snapshots.clear();
                batch.amounts.clear();
                batch.clear_work_summary();
            }
        } else {
            batch.push_back(std::move(snapshot));
            if (batch.size() == batch_size) {
                consume_batch(batch);
                batch.clear();
            }
        }
    }

    if constexpr (FlatCudaInput) {
        if (!batch.snapshots.empty()) {
            consume_batch(batch);
        }
    } else {
        if (!batch.empty()) {
            consume_batch(batch);
        }
    }
    while (std::getline(input, line)) {
        if (contains_non_whitespace(line)) {
            throw std::runtime_error(
                "snapshots file has extra data after the declared S snapshots");
        }
    }
    if (input.bad()) {
        throw std::runtime_error("failed while reading snapshots file");
    }
}

// CPU 与 CUDA 两种批数据类型的轻量封装。
// 读取文本文件→ 解析并构造普通vector<Snapshot>
template <typename BatchConsumer>
void read_snapshot_batches(const std::filesystem::path& snapshots_path,
                           std::size_t batch_size,
                           BatchConsumer consume_batch) {
    read_snapshot_batches_impl<false>(
        snapshots_path, batch_size, 0, std::move(consume_batch));
}

//读取Snapshot文本 → 直接构造CUDA连续批数据
template <typename BatchConsumer>
void read_cuda_snapshot_batches(const std::filesystem::path& snapshots_path,
                                std::size_t batch_size,
                                std::size_t k,
                                BatchConsumer consume_batch) {
    read_snapshot_batches_impl<true>(
        snapshots_path, batch_size, k, std::move(consume_batch));
}

} // namespace



// 三个核心 I/O 流程：replay_file、compute_metrics_file、compute_metrics_cuda_file
// Event 文本 -> OrderBook -> Resolution、StateSummary 和 Snapshot 文件。
void replay_file(const std::filesystem::path& events_path,
                 std::uint64_t snapshot_interval,
                 const std::filesystem::path& output_directory) {
    // replay_file 是阶段一的总控流程：解析一个事件 -> 修改 OrderBook ->
    // 写 Resolution/State -> 按 M 判断是否写 Snapshot。

    // 1. 检查运行时参数 M，并打开事件输入文件。
    // M 表示每处理多少个输入事件生成一份 Snapshot，不能为 0。
    if (snapshot_interval == 0) {
        throw std::invalid_argument("snapshot interval M must be greater than 0");
    }

    std::ifstream input(events_path);
    if (!input) {
        throw std::runtime_error("cannot open events file: " + events_path.string());
    }

    // 2. 读取第一行声明的事件数量 N。
    // 后续必须恰好读到 N 条事件；提前结束或出现额外非空数据都会报错。
    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("events file is empty");
    }
    const std::uint64_t event_count = parse_single_count_line(line, "event count N");
    if (event_count == 0 || event_count > kMaximumEventCount) {
        throw std::runtime_error("event count N must be in [1, 10000000]");
    }

    // 3. 创建输出目录，并同时打开阶段一要求的三个输出文件。
    // Resolution 和 State 是 CSV；Snapshot 使用任务书规定的变长文本格式。
    std::error_code filesystem_error;
    std::filesystem::create_directories(output_directory, filesystem_error);
    if (filesystem_error) {
        throw std::runtime_error("cannot create output directory: " +
                                 filesystem_error.message());
    }

    const auto resolutions_path = output_directory / "resolutions.csv";
    const auto states_path = output_directory / "states.csv";
    const auto snapshots_path = output_directory / "snapshots.txt";
    std::ofstream resolutions_output(resolutions_path);
    std::ofstream states_output(states_path);
    std::ofstream snapshots_output(snapshots_path);
    ensure_output_stream(resolutions_output, resolutions_path);
    ensure_output_stream(states_output, states_path);
    ensure_output_stream(snapshots_output, snapshots_path);

    // 4. snapshots.txt 的第一行必须先写快照总数 S。
    // 除了第 M、2M、3M... 个事件外，最后不足 M 个事件时仍保存最终状态，
    // 因此 S 等于 ceil(N/M)。
    const std::uint64_t snapshot_count =
        event_count / snapshot_interval +
        (event_count % snapshot_interval == 0 ? 0 : 1);
    snapshots_output << snapshot_count << '\n';

    // 5. 创建一份从空状态开始的 OrderBook，并记录上一条 seq。
    // 输入文件按行排列，但仍需显式检查 seq 全局唯一且严格递增。
    OrderBook book;
    std::uint64_t previous_seq = 0;
    bool has_previous_seq = false;
    for (std::uint64_t index = 1; index <= event_count; ++index) {
        // 6.1 读取并解析第 index 个事件。
        // parse_event_line 只检查文本格式和输入范围，业务上的非法值、重复ID
        // 和未知REMOVE由 OrderBook::apply 判断。
        if (!std::getline(input, line)) {
            throw std::runtime_error("events file ended before event " +
                                     std::to_string(index));
        }
        const Event event = parse_event_line(line, index + 1);
        // seq 使用输入值，不要求连续，但必须全局严格递增。
        if (has_previous_seq && event.seq <= previous_seq) {
            throw std::runtime_error("event seq must be globally unique and strictly increasing");
        }
        previous_seq = event.seq;
        has_previous_seq = true;

        // 6.2 执行一个完整事件。
        // apply 返回本事件产生的 0..N 条 Resolution、最终状态和处理状态码。
        const ApplyResult result = book.apply(event);

        // 6.3 按实际处理顺序写全部 Resolution，再写唯一一条 StateSummary。
        // 被拒绝事件没有 Resolution，但仍会写带当前 seq 的 StateSummary。
        // 一个 INSERT 可能跨多个 resident item，因此先按产生顺序写 0..N 条
        // Resolution；StateSummary 则每个事件恰好写一条。
        for (const Resolution& resolution : result.resolutions) {
            write_resolution(resolutions_output, resolution);
        }
        write_state(states_output, result.state, result.status);

        // 6.4 到达 M 的倍数或最后一个事件时，在事件完全处理后保存快照。
        // Snapshot 来自 OrderBook 当前状态，只保留每个 level 的聚合量。
        if (index % snapshot_interval == 0 || index == event_count) {
            // 快照一定在本事件完全处理且 StateSummary 生成之后取得。
            write_snapshot(snapshots_output, book.make_snapshot(event.seq));
        }
    }

    // 7. 声明的 N 条事件之后只允许出现空白行，避免悄悄忽略额外事件；
    // input.bad() 则表示底层读取过程本身失败。
    while (std::getline(input, line)) {
        if (contains_non_whitespace(line)) {
            throw std::runtime_error("events file has extra data after the declared N events");
        }
    }
    if (input.bad()) {
        throw std::runtime_error("failed while reading events file");
    }

    // 8. 最后统一检查三个输出流，捕获磁盘写入过程中的失败。
    if (!resolutions_output || !states_output || !snapshots_output) {
        throw std::runtime_error("failed while writing replay output files");
    }
}

// ---------- CPU metrics 文件流程 ----------
// 流式读取 Snapshot，调用 CPU reference，并保持输入顺序写出指标。
void compute_metrics_file(const std::filesystem::path& snapshots_path,
                          std::size_t k,
                          const std::filesystem::path& metrics_path) {
    // 这是 CUDA 的正确性 reference。按一份一个 batch 流式读取，可避免把
    // 最多一千万份 Snapshot 同时放入内存。

    // 1. 检查计算深度 K。每侧最终只统计前 min(K, depth) 个 level。
    if (k == 0 || k > 16384) {
        throw std::invalid_argument("K must be in [1, 16384]");
    }

    // 2. 如果输出文件位于子目录，先创建其父目录。
    if (!metrics_path.parent_path().empty()) {
        std::error_code filesystem_error;
        std::filesystem::create_directories(metrics_path.parent_path(), filesystem_error);
        if (filesystem_error) {
            throw std::runtime_error("cannot create metrics output directory: " +
                                     filesystem_error.message());
        }
    }

    // 3. 打开最终 metrics CSV，并立即检查是否成功。
    std::ofstream output(metrics_path);
    ensure_output_stream(output, metrics_path);

    // 4. 以 batch_size=1 流式读取 Snapshot。
    // 每得到一份快照就调用 CPU compute_snapshot_metrics，再立刻写出，
    // 因而内存占用不随快照总数 S 增长。
    read_snapshot_batches(snapshots_path, 1, [&](const std::vector<Snapshot>& batch) {
        // batch_size=1，所以 front() 就是当前刚解析完成的 Snapshot。
        write_metrics(output, compute_snapshot_metrics(batch.front(), k));
    });

    // 5. 检查计算完成后的最终写入状态。
    if (!output) {
        throw std::runtime_error("failed while writing metrics file");
    }
}

// ---------- CUDA metrics 文件流程 ----------
// 按 batch_size 构造连续输入，每批送入 GPU，再保持输入顺序写出指标。
void compute_metrics_cuda_file(const std::filesystem::path& snapshots_path,
                               std::size_t k,
                               std::size_t batch_size,
                               const std::filesystem::path& metrics_path,
                               CudaSnapshotKernelMode kernel_mode) {
    // 1. 检查 K 和 batch_size，避免非法深度、空批次或过大的批次计数。
    if (k == 0 || k > 16384) {
        throw std::invalid_argument("K must be in [1, 16384]");
    }
    if (batch_size == 0 || batch_size > kMaximumEventCount) {
        throw std::invalid_argument("batch size must be in [1, 10000000]");
    }

    // 2. 在读取大文件前确认程序已编译CUDA且当前存在可用设备。
    // CPU-only构建会通过 stub 返回清晰原因，而不会静默退回CPU。
    std::string unavailable_reason;
    if (!cuda_device_available(&unavailable_reason)) {
        throw std::runtime_error(unavailable_reason);
    }

    // 3. 创建输出文件的父目录，然后打开最终 metrics CSV。
    if (!metrics_path.parent_path().empty()) {
        std::error_code filesystem_error;
        std::filesystem::create_directories(metrics_path.parent_path(), filesystem_error);
        if (filesystem_error) {
            throw std::runtime_error("cannot create metrics output directory: " +
                                     filesystem_error.message());
        }
    }
    std::ofstream output(metrics_path);
    ensure_output_stream(output, metrics_path);

    // 4. 按 batch_size 流式读取 Snapshot。
    // 解析时直接形成 descriptor + amounts 连续布局；每批准备好后，lambda
    // 调用CUDA完成H2D、kernel、D2H，再按输入顺序逐行写出结果。
    read_cuda_snapshot_batches(
        snapshots_path,
        batch_size,
        k,
        [&](const CudaMetricsInputBatch& batch) {
            const std::vector<SnapshotMetrics> metrics =
                compute_snapshot_metrics_cuda_flat(
                    batch, nullptr, kernel_mode);
            for (const SnapshotMetrics& row : metrics) {
                write_metrics(output, row);
            }
        });

    // 5. 检查最后一次批量写出后，输出流是否仍处于正常状态。
    if (!output) {
        throw std::runtime_error("failed while writing metrics file");
    }
}

} // namespace doq
