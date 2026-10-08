#include "metrics.hpp"
#include "order_book.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::uint64_t parse_argument(const char* text, const char* name) {
    try {
        std::size_t consumed = 0;
        const std::string input(text);
        const std::uint64_t value = std::stoull(input, &consumed);
        if (consumed != input.size() || value == 0) {
            throw std::invalid_argument("invalid");
        }
        return value;
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    }
}

doq::Event event_for(std::uint64_t index) {
    const std::uint64_t seq = index + 1;
    const std::uint64_t item_id = index + 1;
    switch (index % 4) {
    case 0:
        return {seq, item_id, doq::Side::A, doq::Op::Insert, 100, 10};
    case 1:
        return {seq, item_id, doq::Side::A, doq::Op::Insert, 99, 5};
    case 2:
        return {seq, item_id, doq::Side::B, doq::Op::Insert, 100, 8};
    default:
        return {seq, item_id, doq::Side::B, doq::Op::Insert, 99, 7};
    }
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        const std::uint64_t event_count =
            argc > 1 ? parse_argument(argv[1], "event_count") : 1'000'000;
        const std::uint64_t snapshot_interval =
            argc > 2 ? parse_argument(argv[2], "snapshot_interval") : 997;
        const std::uint64_t k = argc > 3 ? parse_argument(argv[3], "K") : 64;
        if (event_count > 10'000'000) {
            throw std::invalid_argument("event_count must not exceed 10000000");
        }
        if (k > 16384) {
            throw std::invalid_argument("K must not exceed 16384");
        }

        doq::OrderBook book;
        std::vector<doq::Snapshot> snapshots;
        snapshots.reserve(static_cast<std::size_t>(
            event_count / snapshot_interval +
            (event_count % snapshot_interval == 0 ? 0 : 1)));

        std::uint64_t resolution_count = 0;
        const auto replay_start = std::chrono::steady_clock::now();
        for (std::uint64_t index = 0; index < event_count; ++index) {
            const doq::Event event = event_for(index);
            const doq::ApplyResult result = book.apply(event);
            resolution_count += result.resolutions.size();
            if ((index + 1) % snapshot_interval == 0 || index + 1 == event_count) {
                snapshots.push_back(book.make_snapshot(event.seq));
            }
        }
        const auto replay_end = std::chrono::steady_clock::now();

        std::int64_t checksum = 0;
        const auto metrics_start = std::chrono::steady_clock::now();
        for (const doq::Snapshot& snapshot : snapshots) {
            const doq::SnapshotMetrics metrics =
                doq::compute_snapshot_metrics(snapshot, static_cast<std::size_t>(k));
            checksum += metrics.balance_num;
        }
        const auto metrics_end = std::chrono::steady_clock::now();

        std::string invariant_error;
        if (!book.validate_invariants(&invariant_error)) {
            throw std::runtime_error("invariant failure: " + invariant_error);
        }

        const double replay_seconds =
            std::chrono::duration<double>(replay_end - replay_start).count();
        const double metrics_seconds =
            std::chrono::duration<double>(metrics_end - metrics_start).count();

        std::cout << std::fixed << std::setprecision(6)
                  << "events=" << event_count << '\n'
                  << "snapshot_interval=" << snapshot_interval << '\n'
                  << "snapshots=" << snapshots.size() << '\n'
                  << "K=" << k << '\n'
                  << "resolutions=" << resolution_count << '\n'
                  << "replay_seconds=" << replay_seconds << '\n'
                  << "replay_events_per_second="
                  << static_cast<double>(event_count) / replay_seconds << '\n'
                  << "metrics_seconds=" << metrics_seconds << '\n'
                  << "metrics_snapshots_per_second="
                  << static_cast<double>(snapshots.size()) / metrics_seconds << '\n'
                  << "checksum=" << checksum << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
