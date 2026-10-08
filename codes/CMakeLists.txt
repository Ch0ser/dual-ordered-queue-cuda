cmake_minimum_required(VERSION 3.18)

project(dual_ordered_queue LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(DOQ_BUILD_TESTS "Build the phase-one test suite" ON)
option(DOQ_BUILD_BENCHMARK "Build the repeatable CPU benchmark" ON)
option(DOQ_ENABLE_CUDA "Build the CUDA SnapshotMetrics implementation when available" ON)

# Source files contain Chinese explanatory comments. Windows compilers otherwise
# inherit the system code page (for example CP936), which can make nvcc/MSVC
# misread UTF-8 bytes before compilation.
if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 86 89 CACHE STRING
        "CUDA architectures to compile (override for the target GPU when needed)")
endif()

set(DOQ_HAS_CUDA OFF)
if(DOQ_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
        set(DOQ_HAS_CUDA ON)
        message(STATUS "CUDA SnapshotMetrics enabled: ${CMAKE_CUDA_COMPILER}")
    else()
        message(STATUS "CUDA compiler not found; building CPU paths and a clear CUDA error stub")
    endif()
endif()

add_library(doq_core
    src/io.cpp
    src/metrics.cpp
    src/order_book.cpp
)

if(DOQ_HAS_CUDA)
    target_sources(doq_core PRIVATE src/metrics_cuda.cu)
    target_compile_definitions(doq_core PUBLIC DOQ_HAS_CUDA=1)
else()
    target_sources(doq_core PRIVATE src/metrics_cuda_stub.cpp)
endif()

target_include_directories(doq_core PUBLIC src)

if(MSVC)
    target_compile_options(doq_core PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4;/permissive->
    )
else()
    target_compile_options(doq_core PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
endif()

add_executable(doq src/main.cpp)
target_link_libraries(doq PRIVATE doq_core)

if(DOQ_BUILD_TESTS)
    enable_testing()
    add_executable(doq_tests tests/test_main.cpp)
    target_link_libraries(doq_tests PRIVATE doq_core)
    add_test(NAME correctness_tests COMMAND doq_tests)
endif()

if(DOQ_BUILD_BENCHMARK)
    add_executable(doq_benchmark bench/benchmark.cpp)
    target_link_libraries(doq_benchmark PRIVATE doq_core)

    add_executable(doq_metrics_benchmark bench/metrics_benchmark.cpp)
    target_link_libraries(doq_metrics_benchmark PRIVATE doq_core)

    add_executable(doq_kernel_shape_benchmark bench/kernel_shape_benchmark.cpp)
    target_link_libraries(doq_kernel_shape_benchmark PRIVATE doq_core)

    if(DOQ_HAS_CUDA)
        add_executable(doq_gpu_text_benchmark bench/gpu_text_benchmark.cu)
        target_link_libraries(doq_gpu_text_benchmark PRIVATE doq_core)

        add_executable(doq_transfer_layout_benchmark
            bench/transfer_layout_benchmark.cu)
        target_link_libraries(doq_transfer_layout_benchmark PRIVATE doq_core)
    endif()
endif()
