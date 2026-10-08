#include "io.hpp"
#include "metrics.hpp"
#include "metrics_cuda.hpp"
#include "order_book.hpp"
#include "types.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        const auto stamp = std::chrono::high_resolution_clock::now()
                               .time_since_epoch()
                               .count();
        path_ = std::filesystem::temp_directory_path() /
                ("doq_phase_one_tests_" + std::to_string(stamp));
        std::filesystem::create_directories(path_);
    }

    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    const std::filesystem::path& path() const {
        return path_;
    }

private:
    std::filesystem::path path_;
};

void check(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Actual, typename Expected>
void check_equal(const Actual& actual,
                 const Expected& expected,
                 const std::string& message) {
    if (!(actual == expected)) {
        throw std::runtime_error(message);
    }
}

void write_text(const std::filesystem::path& path, const std::string& contents) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot create test file: " + path.string());
    }
    output << contents;
}

std::string read_text(const std::filesystem::path& path) {
    // Text mode normalizes the platform newline sequence so the same expected
    // strings work on Windows and Unix-like systems.
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot read test file: " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

doq::Event insert_event(std::uint64_t seq,
                        std::uint64_t item_id,
                        doq::Side side,
                        std::int64_t key,
                        std::int64_t amount) {
    return doq::Event{seq, item_id, side, doq::Op::Insert, key, amount};
}

doq::Event remove_event(std::uint64_t seq, std::uint64_t item_id) {
    return doq::Event{seq, item_id, doq::Side::A, doq::Op::Remove, 0, 0};
}

void check_invariants(const doq::OrderBook& book) {
    std::string error;
    check(book.validate_invariants(&error), "invariant failure: " + error);
}

void test_official_example_end_to_end() {
    TemporaryDirectory temporary;
    const auto events = temporary.path() / "events.txt";
    const auto output = temporary.path() / "out";
    write_text(events,
               "6\n"
               "I 1 101 A 100 10\n"
               "I 2 102 A 100 5\n"
               "I 3 201 B 99 12\n"
               "R 4 102\n"
               "I 5 301 B 105 4\n"
               "I 6 401 A 106 2\n");

    doq::replay_file(events, 3, output);

    check_equal(read_text(output / "resolutions.csv"),
                std::string("3,201,101,100,10\n"
                            "3,201,102,100,2\n"
                            "6,401,301,105,2\n"),
                "official resolutions.csv mismatch");
    check_equal(read_text(output / "states.csv"),
                std::string("1,100,-1,10,0,1,OK\n"
                            "2,100,-1,15,0,2,OK\n"
                            "3,100,-1,3,0,1,OK\n"
                            "4,-1,-1,0,0,0,OK\n"
                            "5,-1,105,0,4,1,OK\n"
                            "6,-1,105,0,2,1,OK\n"),
                "official states.csv mismatch");
    check_equal(read_text(output / "snapshots.txt"),
                std::string("2\n"
                            "3 1 0\n"
                            "100 3\n"
                            "6 0 1\n"
                            "105 2\n"),
                "official snapshots.txt mismatch");

    const auto metrics = output / "metrics.csv";
    doq::compute_metrics_file(output / "snapshots.txt", 1, metrics);
    check_equal(read_text(metrics),
                std::string("3,100,-1,3,0,-1,3,3,-1,0\n"
                            "6,-1,105,0,2,-1,-2,2,-1,0\n"),
                "official example metrics mismatch");
}

void test_fifo_and_middle_remove() {
    doq::OrderBook book;
    book.apply(insert_event(1, 101, doq::Side::A, 100, 2));
    book.apply(insert_event(2, 102, doq::Side::A, 100, 3));
    book.apply(insert_event(3, 103, doq::Side::A, 100, 4));
    check_equal(book.apply(remove_event(4, 102)).status,
                doq::ApplyStatus::Ok,
                "middle removal must succeed");

    const doq::ApplyResult result =
        book.apply(insert_event(5, 201, doq::Side::B, 100, 5));
    check_equal(result.resolutions.size(), std::size_t{2}, "FIFO resolution count");
    check_equal(result.resolutions[0].resident_item_id,
                std::uint64_t{101},
                "FIFO must consume the head first");
    check_equal(result.resolutions[0].amount, std::int64_t{2}, "first FIFO amount");
    check_equal(result.resolutions[1].resident_item_id,
                std::uint64_t{103},
                "removed middle item must not reappear");
    check_equal(result.resolutions[1].amount, std::int64_t{3}, "second FIFO amount");
    check_equal(result.state.a_amount, std::int64_t{1}, "resident tail remainder");
    check_equal(result.state.active_count, std::uint64_t{1}, "active count after FIFO");
    check_invariants(book);
}

void test_remove_head_and_tail() {
    {
        doq::OrderBook book;
        book.apply(insert_event(1, 1, doq::Side::A, 100, 2));
        book.apply(insert_event(2, 2, doq::Side::A, 100, 3));
        book.apply(insert_event(3, 3, doq::Side::A, 100, 4));
        check_equal(book.apply(remove_event(4, 1)).status,
                    doq::ApplyStatus::Ok,
                    "head removal must succeed");
        const doq::ApplyResult result =
            book.apply(insert_event(5, 10, doq::Side::B, 100, 1));
        check_equal(result.resolutions[0].resident_item_id,
                    std::uint64_t{2},
                    "the second item must become head after head removal");
        check_invariants(book);
    }

    {
        doq::OrderBook book;
        book.apply(insert_event(1, 1, doq::Side::A, 100, 2));
        book.apply(insert_event(2, 2, doq::Side::A, 100, 3));
        book.apply(insert_event(3, 3, doq::Side::A, 100, 4));
        check_equal(book.apply(remove_event(4, 3)).status,
                    doq::ApplyStatus::Ok,
                    "tail removal must succeed");
        const doq::ApplyResult result =
            book.apply(insert_event(5, 10, doq::Side::B, 100, 5));
        check_equal(result.resolutions.size(), std::size_t{2}, "tail-removal count");
        check_equal(result.resolutions[0].resident_item_id,
                    std::uint64_t{1},
                    "tail removal must retain the original head");
        check_equal(result.resolutions[1].resident_item_id,
                    std::uint64_t{2},
                    "removed tail must not be resolved");
        check_equal(result.state.active_count,
                    std::uint64_t{0},
                    "remaining residents must resolve completely");
        check_invariants(book);
    }
}

void test_cross_level_resolution() {
    doq::OrderBook book;
    book.apply(insert_event(1, 101, doq::Side::A, 103, 4));
    book.apply(insert_event(2, 102, doq::Side::A, 101, 6));
    book.apply(insert_event(3, 103, doq::Side::A, 101, 5));

    const doq::ApplyResult result =
        book.apply(insert_event(4, 201, doq::Side::B, 100, 12));
    check_equal(result.resolutions.size(), std::size_t{3}, "cross-level count");
    check_equal(result.resolutions[0].resident_item_id, std::uint64_t{101}, "level 103");
    check_equal(result.resolutions[1].resident_item_id, std::uint64_t{102}, "level 101 head");
    check_equal(result.resolutions[2].resident_item_id, std::uint64_t{103}, "level 101 tail");
    check_equal(result.resolutions[2].amount, std::int64_t{2}, "partial tail amount");
    check_equal(result.state.a_extreme_key, std::int64_t{101}, "remaining A key");
    check_equal(result.state.a_amount, std::int64_t{3}, "remaining A amount");
    check_equal(result.state.b_extreme_key, std::int64_t{-1}, "B must be empty");
    check_invariants(book);
}

void test_invalid_duplicate_and_not_found() {
    doq::OrderBook book;
    check_equal(book.apply(insert_event(1, 101, doq::Side::A, 0, 5)).status,
                doq::ApplyStatus::RejectedInvalid,
                "invalid insert status");
    check_equal(book.apply(insert_event(2, 101, doq::Side::A, 100, 5)).status,
                doq::ApplyStatus::Ok,
                "invalid insert must not consume item_id");
    check_equal(book.apply(remove_event(3, 101)).status,
                doq::ApplyStatus::Ok,
                "remove active item");
    check_equal(book.apply(insert_event(4, 101, doq::Side::B, 0, 2)).status,
                doq::ApplyStatus::RejectedInvalid,
                "invalid status must precede duplicate status");
    check_equal(book.apply(insert_event(5, 101, doq::Side::B, 110, 2)).status,
                doq::ApplyStatus::RejectedDuplicate,
                "successful id must never be reused");
    check_equal(book.apply(remove_event(6, 101)).status,
                doq::ApplyStatus::RejectedNotFound,
                "removed item is no longer active");
    check_invariants(book);
}

void test_snapshot_interval_counts_invalid_events() {
    TemporaryDirectory temporary;
    const auto events = temporary.path() / "events.txt";
    const auto output = temporary.path() / "out";
    write_text(events,
               "4\n"
               "I 10 1 A 100 4\n"
               "I 20 2 B 105 6\n"
               "I 30 3 A 0 1\n"
               "R 40 1\n");

    doq::replay_file(events, 3, output);
    check_equal(read_text(output / "snapshots.txt"),
                std::string("2\n"
                            "30 1 1\n"
                            "100 4\n"
                            "105 6\n"
                            "40 0 1\n"
                            "105 6\n"),
                "invalid events count toward M and a final snapshot is required");
}

void test_metrics_with_both_and_empty_sides() {
    TemporaryDirectory temporary;
    const auto snapshots = temporary.path() / "snapshots.txt";
    const auto metrics = temporary.path() / "metrics.csv";
    write_text(snapshots,
               "2\n"
               "10 3 2\n"
               "100 7\n"
               "98 5\n"
               "95 10\n"
               "105 4\n"
               "108 6\n"
               "20 0 1\n"
               "110 9\n");

    doq::compute_metrics_file(snapshots, 2, metrics);
    check_equal(read_text(metrics),
                std::string("10,100,105,12,10,5,2,22,2260,22\n"
                            "20,-1,110,0,9,-1,-9,9,-1,0\n"),
                "SnapshotMetrics formulas or empty-side rule mismatch");
}

void test_partial_match_puts_current_item_on_own_side() {
    doq::OrderBook book;
    book.apply(insert_event(1, 201, doq::Side::B, 99, 3));
    book.apply(insert_event(2, 202, doq::Side::B, 102, 5));
    const doq::ApplyResult result =
        book.apply(insert_event(3, 101, doq::Side::A, 100, 6));

    check_equal(result.resolutions.size(), std::size_t{1}, "partial-match count");
    check_equal(result.resolutions[0].resident_item_id,
                std::uint64_t{201},
                "A must consume the minimum B key");
    check_equal(result.state.a_extreme_key, std::int64_t{100}, "A remainder key");
    check_equal(result.state.a_amount, std::int64_t{3}, "A remainder amount");
    check_equal(result.state.b_extreme_key, std::int64_t{102}, "next B key");
    check_equal(result.state.b_amount, std::int64_t{5}, "next B amount");
    check_invariants(book);
}

void test_input_rejects_non_increasing_seq() {
    TemporaryDirectory temporary;
    const auto events = temporary.path() / "events.txt";
    write_text(events,
               "2\n"
               "I 2 1 A 100 1\n"
               "R 2 1\n");

    bool threw = false;
    try {
        doq::replay_file(events, 1, temporary.path() / "out");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "non-increasing seq must be rejected as malformed input");
}

void test_cuda_metrics_match_cpu() {
    std::string unavailable_reason;
    if (!doq::cuda_device_available(&unavailable_reason)) {
        std::cout << "[SKIP] CUDA consistency: " << unavailable_reason << '\n';
        return;
    }

    TemporaryDirectory temporary;
    const auto snapshots = temporary.path() / "snapshots.txt";
    write_text(snapshots,
               "6\n"
               "10 0 0\n"
               "20 3 0\n"
               "100 2\n"
               "90 3\n"
               "80 5\n"
               "30 0 2\n"
               "110 7\n"
               "120 11\n"
               "40 1 1\n"
               "100 13\n"
               "105 17\n"
               "50 4 3\n"
               "100 2\n"
               "98 3\n"
               "96 5\n"
               "94 7\n"
               "105 11\n"
               "108 13\n"
               "112 17\n"
               "60 2 5\n"
               "200 19\n"
               "190 23\n"
               "210 29\n"
               "220 31\n"
               "230 37\n"
               "240 41\n"
               "250 43\n");

    for (const std::size_t k : {std::size_t{1}, std::size_t{3}, std::size_t{16384}}) {
        const auto cpu_path = temporary.path() /
                              ("metrics_cpu_k" + std::to_string(k) + ".csv");
        doq::compute_metrics_file(snapshots, k, cpu_path);
        const std::string expected = read_text(cpu_path);

        for (const doq::CudaSnapshotKernelMode mode :
             {doq::CudaSnapshotKernelMode::automatic,
              doq::CudaSnapshotKernelMode::thread_per_snapshot}) {
            for (const std::size_t batch_size :
                 {std::size_t{1}, std::size_t{2}, std::size_t{64}}) {
                const auto cuda_path = temporary.path() /
                    ("metrics_cuda_k" + std::to_string(k) + "_b" +
                     std::to_string(batch_size) + "_" +
                     doq::cuda_snapshot_kernel_mode_name(mode) + ".csv");
                doq::compute_metrics_cuda_file(
                    snapshots, k, batch_size, cuda_path, mode);
                check_equal(read_text(cuda_path),
                            expected,
                            "CUDA metrics differ from CPU reference for K=" +
                                std::to_string(k) + ", batch_size=" +
                                std::to_string(batch_size) + ", mode=" +
                                doq::cuda_snapshot_kernel_mode_name(mode));
            }
        }
    }

    doq::CudaMetricsInputBatch pipeline_input;
    pipeline_input.amounts.push_back(2);
    pipeline_input.amounts.push_back(3);
    pipeline_input.amounts.push_back(5);
    pipeline_input.amounts.push_back(7);
    pipeline_input.snapshots.push_back(
        doq::CudaSnapshotDescriptor{70, 0, 2, 2, 2, 100, 110});
    pipeline_input.record_snapshot_work(2, 2);
    const doq::CudaPipelineBatchResult pipeline =
        doq::compute_snapshot_metrics_cuda_flat_pipeline(
            pipeline_input, 3, doq::CudaSnapshotKernelMode::automatic);
    check(pipeline.output.size() == 1,
          "CUDA pipeline must return one metric row");
    const doq::SnapshotMetrics& row = pipeline.output.front();
    check(row.event_seq == 70 && row.a_amount_top_k == 5 &&
              row.b_amount_top_k == 12 && row.key_gap == 10 &&
              row.balance_num == -7 && row.balance_den == 17 &&
              row.weighted_key_num == 1750 && row.weighted_key_den == 17,
          "CUDA overlapped pipeline differs from the expected metrics");
}

} // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests{
        {"official example end-to-end", test_official_example_end_to_end},
        {"FIFO and middle remove", test_fifo_and_middle_remove},
        {"head and tail remove", test_remove_head_and_tail},
        {"cross-level resolution", test_cross_level_resolution},
        {"invalid, duplicate, not found", test_invalid_duplicate_and_not_found},
        {"snapshot interval", test_snapshot_interval_counts_invalid_events},
        {"snapshot metrics", test_metrics_with_both_and_empty_sides},
        {"partial match remainder", test_partial_match_puts_current_item_on_own_side},
        {"strict seq parsing", test_input_rejects_non_increasing_seq},
        {"CPU/CUDA SnapshotMetrics consistency", test_cuda_metrics_match_cpu},
    };

    std::size_t passed = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            ++passed;
            std::cout << "[PASS] " << name << '\n';
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
        }
    }

    std::cout << passed << '/' << tests.size() << " tests passed\n";
    return passed == tests.size() ? 0 : 1;
}
