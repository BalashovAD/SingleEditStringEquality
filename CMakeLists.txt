cmake_minimum_required(VERSION 3.24)
project(cmpString)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "-Werror -Wall -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
# Tests

enable_testing()
set(GTEST_DIR ${PROJECT_SOURCE_DIR}/thirdparty/googletest)
add_subdirectory(${GTEST_DIR} ${CMAKE_BINARY_DIR}/googletest)

add_executable(unit-tests test.cpp fn.cpp fn.h)
target_link_libraries(unit-tests PRIVATE gtest gtest_main)
target_include_directories(unit-tests PRIVATE
        ${GTEST_DIR}/googletest/include)

add_test(NAME unit-tests
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# BENCHMARK
set(BENCHMARK_ENABLE_TESTING off)

set(BENCHMARK_DIR ${PROJECT_SOURCE_DIR}/thirdparty/benchmark)
add_subdirectory(${BENCHMARK_DIR} ${CMAKE_BINARY_DIR}/benchmark)
set(BENCHMARK_LIBRARIES benchmark::benchmark)
add_executable(bench benchmark.cpp fn.cpp fn.h)
target_include_directories(bench PRIVATE
        ${BENCHMARK_DIR}/include)
target_link_libraries(bench ${BENCHMARK_LIBRARIES})
