#include <benchmark/benchmark.h>
#include <random>
#include <array>

#include "fn.h"

using sv = std::string_view;
using fn = bool(*)(sv, sv);

std::string gen(size_t sizeOfString) {
    constexpr std::string_view chars = "abcdefghijklmnopqrstuvwxyz"
                              "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                              "0123456789";

    std::random_device rd;
    std::mt19937 engine(rd());
    std::uniform_int_distribution<> dist(0, chars.size() - 1);

    std::string result;
    for (size_t i = 0; i != sizeOfString; ++i) {
        result += chars[dist(engine)];
    }
    return result;
}

std::string diff1(std::string str) {
    str[str.size() / 2] = '!';
    return str;
}

std::string diff2(std::string str) {
    str[0] = '!';
    return str;
}

std::string diff3(std::string str) {
    str.push_back('#');
    return str;
}

std::string diff4(std::string str) {
    auto i = str.size() / 2;
    std::string result = str.substr(0, i);
    result.push_back('!');
    result.append(str.data() + i, str.size() - i);
    return str;
}

std::string diff5(std::string str) {
    return '!' + str;
}


static void BM_eq(benchmark::State& state, fn fn, std::string const& challenge) {
    for (auto _ : state) {
        (fn(challenge, challenge));
    }
}

using DiffFn = std::string(*)(std::string);
static void BM_diff(benchmark::State& state, fn fn, std::string const& challenge) {
    static constexpr size_t DIFF_COUNT = 5;
    auto diffList = std::array<DiffFn, DIFF_COUNT>{diff1, diff2, diff3, diff4, diff5};
    std::array<std::string, DIFF_COUNT> rhsList;
    for (auto i = 0; i != DIFF_COUNT; ++i) {
        rhsList[i] = diffList[i](challenge);
    }

    for (auto _ : state) {
        for (auto const& rhs : rhsList) {
            (fn(challenge, rhs));
        }
    }
}


static inline std::string SHORT_CHALLENGE = gen(15);
static inline std::string MID_CHALLENGE = gen(45);
static inline std::string LONG_CHALLENGE = gen(16 * 80 + 5);
static inline std::string INF_CHALLENGE = gen(1024 * 120);


#define DEF_BENCH(name, fn) \
BENCHMARK_CAPTURE(BM_eq, EQ_short_ ## name, fn, SHORT_CHALLENGE); \
BENCHMARK_CAPTURE(BM_diff, DIFF_short_ ## name, fn, SHORT_CHALLENGE);\
BENCHMARK_CAPTURE(BM_eq, EQ_mid_ ## name, fn, MID_CHALLENGE);\
BENCHMARK_CAPTURE(BM_diff, DIFF_mid_ ## name, fn, MID_CHALLENGE);\
BENCHMARK_CAPTURE(BM_eq, EQ_long_ ## name, fn, LONG_CHALLENGE);\
BENCHMARK_CAPTURE(BM_diff, DIFF_long_ ## name, fn, LONG_CHALLENGE);\
BENCHMARK_CAPTURE(BM_eq, EQ_inf_ ## name, fn, INF_CHALLENGE);\
BENCHMARK_CAPTURE(BM_diff, DIFF_inf_ ## name, fn, INF_CHALLENGE);

DEF_BENCH(slow, oneChangeSlow);
DEF_BENCH(fast, oneChangeNoSIMDFast);
DEF_BENCH(sse, oneChange);
DEF_BENCH(avx, oneChangeAVX);
DEF_BENCH(sseFast, oneChangeFast);
DEF_BENCH(avxFast, oneChangeFastAVX);

BENCHMARK_MAIN();
