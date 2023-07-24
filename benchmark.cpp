#include <benchmark/benchmark.h>
#include <random>
#include <array>
#include <stdexcept>
#include <cstring>

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

char gen1() {
    return gen(1)[0];
}

void changeSymbol(char& c) {
    char newChar = gen1();
    while (newChar == c)
        newChar = gen1();
    c = newChar;
}

std::string diff1(std::string str) {
    changeSymbol(str[str.size() / 2]);
    return str;
}

std::string diff2(std::string str) {
    changeSymbol(str[0]);
    return str;
}

std::string diff3(std::string str) {
    str.push_back(gen1());
    return str;
}

std::string diff4(std::string str) {
    auto i = str.size() / 2;
    std::string result = str.substr(0, i);
    result.push_back(gen1());
    result.append(str.data() + i, str.size() - i);
    return str;
}

std::string diff5(std::string str) {
    return gen1() + str;
}

std::string diff6(std::string str) {
    str[str.size() / 2] = gen1();
    str[str.size() - 1] = gen1();
    return str;
}

static constexpr size_t DIFF_COUNT = 6;

static void BM_eq(benchmark::State& state, fn fn, std::string const& challenge) {
    for (auto _ : state) {
        (fn(challenge, challenge));
    }

    state.SetBytesProcessed(static_cast<int64_t>(2 * challenge.size() * state.iterations()));
    state.SetItemsProcessed(state.iterations());

    if (fn(challenge, challenge) != oneChangeSlow(challenge, challenge)) {
        state.SkipWithError("Check failed (EQ)");
    }
}

using DiffFn = std::string(*)(std::string);
static void BM_diff(benchmark::State& state, fn fn, std::string const& challenge) {
    auto diffList = std::array<DiffFn, DIFF_COUNT>{diff1, diff2, diff3, diff4, diff5, diff6};
    std::array<std::string, DIFF_COUNT> rhsList;
    for (auto i = 0; i != DIFF_COUNT; ++i) {
        rhsList[i] = diffList[i](challenge);
    }

    for (auto _ : state) {
        for (auto const& rhs : rhsList) {
            (fn(challenge, rhs));
        }
    }

    int64_t bytes = 0;
    for (auto const& rhs : rhsList) {
        bytes += rhs.size() + challenge.size();
    }
    state.SetBytesProcessed(bytes * state.iterations());
    state.SetItemsProcessed(state.iterations() * DIFF_COUNT);

    for (auto const& rhs : rhsList) {
        if (fn(challenge, rhs) != oneChangeSlow(challenge, rhs)) {
            state.SkipWithError("Check failed (DIFF)");
        }
    }
}

static void BM_memcmp(benchmark::State& state, std::string const& challenge) {
    void *buffer = malloc(challenge.size());
    for (auto _ : state) {
        auto *dst = memcpy(buffer, challenge.data(), challenge.size());
        benchmark::DoNotOptimize(dst);
    }

    state.SetBytesProcessed(static_cast<int64_t>(challenge.size() * state.iterations()));
    free(buffer);
}

static inline std::string SHORT_CHALLENGE = gen(15);
static inline std::string MID_CHALLENGE = gen(45);
static inline std::string LONG_CHALLENGE = gen(16 * 80 + 5); // 1285 = 1 Kb
static inline std::string LONG10_CHALLENGE = gen(1024 * 10 + 13); // 10 Kb
static inline std::string LONG30_CHALLENGE = gen(1024 * 30 + 11); // 30 Kb
static inline std::string INF_CHALLENGE = gen(1024 * 120); // 120 Kb


#define DEF_BENCH(name, fn) \
BENCHMARK_CAPTURE(BM_eq, EQ_short_ ## name, fn, SHORT_CHALLENGE); \
BENCHMARK_CAPTURE(BM_diff, DIFF_short_ ## name, fn, SHORT_CHALLENGE);\
BENCHMARK_CAPTURE(BM_eq, EQ_mid_ ## name, fn, MID_CHALLENGE);\
BENCHMARK_CAPTURE(BM_diff, DIFF_mid_ ## name, fn, MID_CHALLENGE);\
BENCHMARK_CAPTURE(BM_eq, EQ_long_ ## name, fn, LONG_CHALLENGE);\
BENCHMARK_CAPTURE(BM_diff, DIFF_long_ ## name, fn, LONG_CHALLENGE);\
BENCHMARK_CAPTURE(BM_eq, EQ_long10_ ## name, fn, LONG10_CHALLENGE);\
BENCHMARK_CAPTURE(BM_diff, DIFF_long10_ ## name, fn, LONG10_CHALLENGE);\
BENCHMARK_CAPTURE(BM_eq, EQ_long30_ ## name, fn, LONG30_CHALLENGE);\
BENCHMARK_CAPTURE(BM_diff, DIFF_long30_ ## name, fn, LONG30_CHALLENGE);\
BENCHMARK_CAPTURE(BM_eq, EQ_inf_ ## name, fn, INF_CHALLENGE);\
BENCHMARK_CAPTURE(BM_diff, DIFF_inf_ ## name, fn, INF_CHALLENGE);

DEF_BENCH(slow, oneChangeSlow);
DEF_BENCH(fast, oneChangeNoSIMDFast);
DEF_BENCH(sse, oneChange);
DEF_BENCH(avx, oneChangeAVX);
DEF_BENCH(sseFast, oneChangeFast);
DEF_BENCH(avxFast, oneChangeFastAVX);

BENCHMARK_CAPTURE(BM_memcmp, memcmp, SHORT_CHALLENGE);
BENCHMARK_CAPTURE(BM_memcmp, memcmp, INF_CHALLENGE);


BENCHMARK_MAIN();
