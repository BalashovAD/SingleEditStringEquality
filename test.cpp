#include <gtest/gtest.h>

#include <source_location>
#include <bitset>

#include "fn.h"

using namespace testing;
using sv = std::string_view;
using fn = bool(*)(sv, sv);

class OneChangeTest : public ::testing::TestWithParam<fn> {
public:
    static constexpr sv prefix15 = "aqzwsxedcrfvrfv";
    static constexpr sv prefix16 = "aqzwsxedcrfvrfva";
    static constexpr sv prefix32 = "aqzwsxedcaqzwsxedcrfvrfvarfvrfva";
    static constexpr sv prefix31 = "aqzwsedcaqzwsxedcrfvrfvarfvrfva";
    static constexpr sv prefix33 = "aqzwsedcaqzwsxedcrfvrfv2arfvrfva";
    static constexpr sv prefix14 = "zwsxedc1rfvrfv";
    static constexpr sv prefix13 = "zwsxec1rfvrfv";
    static constexpr sv prefix12 = "zwxec1rfvrfv";

    void SetUp() override {
        m_pFn = GetParam();
    }

    auto tt(std::string_view lhs, std::string_view rhs, std::source_location current = std::source_location::current()) {
        test(lhs, rhs, true, current);
    }

    auto tf(std::string_view lhs, std::string_view rhs, std::source_location current = std::source_location::current()) {
        test(lhs, rhs, false, current);
    }
private:
    static std::string location(std::source_location current) noexcept {
        return std::string(current.file_name()) + ":" + std::to_string(current.line());
    }

    void test(sv lhs, sv rhs, bool result, std::source_location current) {
        std::vector<std::tuple<sv, sv, sv>> testCases{
                {"", "", "0-0"},
                {prefix15, prefix14, "15-14"},
                {prefix14, prefix15, "14-15"},
                {prefix16, prefix15, "16-15"},
                {prefix16, prefix16, "16-16"},
                {prefix14, prefix14, "14-14"},
                {prefix13, prefix14, "13-14"},
                {prefix12, prefix14, "12-14"},
                {"", prefix14, "0-14"},
                {prefix14, "", "14-0"},
                {prefix16, "", "16-0"},
                {prefix32, prefix32, "32-32"},
                {prefix32, "", "32-0"},
                {prefix31, "", "31-0"},
                {prefix31, prefix33, "31-33"},
                {prefix33, prefix33, "33-33"},
                {"", prefix33, "0-33"},
        };

        for (auto const& [p, s, d] : testCases) {
            std::string lhsStr = std::string(p) + std::string(lhs) + std::string(s);
            std::string rhsStr = std::string(p) + std::string(rhs) + std::string(s);
            EXPECT_EQ(m_pFn(lhsStr, rhsStr), result) << location(current) << " " << d;
        }
    }

    fn m_pFn;
};


TEST_P(OneChangeTest, Tests) {
    tt("abc", "abc");
    tt("abc", "abb");
    tt("abecd", "abcd");
    tt("abcd", "abc");
    tt("abc", "abcd");
    tt("", "");
    tt("", "a");
    tt("a", "");
    tt("abc", "bc");
    tt("aaaaaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaaaaa" "a");

    tf("abc", "bbb");
    tf("aaaaaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb");
    tf("aaaaaaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb");
    tf("abc", "bb");
    tf("abc", "cc");
    tf("a", "aaa");
    tf("", "aa");
    tf("acb", "abc");
    tf("abc" "aaaaaaaaaaaaaaaaaaa" "abb", "abb" "aaaaaaaaaaaaaaaaaaa" "abc");
    tf("abc" "aaaaaaaaaaaaaaaaaaa" "ab", "abb" "aaaaaaaaaaaaaaaaaaa" "abc");
    tf("c" "aaaaaaaaaaaaaaaaaaa", "b" "aaaaaaaaaaaaaaaaaaa" "a");
}

INSTANTIATE_TEST_SUITE_P(Slow, OneChangeTest, ::testing::Values(oneChangeSlow));
INSTANTIATE_TEST_SUITE_P(NoSIMDFast, OneChangeTest, ::testing::Values(oneChangeNoSIMDFast));
INSTANTIATE_TEST_SUITE_P(Common, OneChangeTest, ::testing::Values(oneChange));
INSTANTIATE_TEST_SUITE_P(CommonAVX, OneChangeTest, ::testing::Values(oneChangeAVX));
INSTANTIATE_TEST_SUITE_P(Fast, OneChangeTest, ::testing::Values(oneChangeFast));
INSTANTIATE_TEST_SUITE_P(FastAVX, OneChangeTest, ::testing::Values(oneChangeFastAVX));

void printInBinary(unsigned int num) {
    std::bitset<32> binary(num);  // assuming 32-bit unsigned int
    std::cout << binary << std::endl;
}

TEST(Movemask, Eq128) {
    __m128i target = _mm_loadu_si128(reinterpret_cast<const __m128i*>("tttttttttttttttt"));
    __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>("tttttttttttttttt"));
    __m128i cmpResult = _mm_cmpeq_epi8(chunk, target);
    unsigned int mask = _mm_movemask_epi8(cmpResult);
    EXPECT_EQ(mask, 0x0000FFFF);
    EXPECT_EQ(std::popcount(mask), 16);
    EXPECT_EQ(popcount(cmpResult), 128);
}

TEST(Movemask, Ne128) {
    __m128i target = _mm_loadu_si128(reinterpret_cast<const __m128i*>("tftttttttttttttt"));
    __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>("tttttttttttttttt"));
    __m128i cmpResult = _mm_cmpeq_epi8(chunk, target);
    unsigned int mask = _mm_movemask_epi8(cmpResult);
    EXPECT_NE(mask, 0x0000FFFF);
    EXPECT_EQ(std::popcount(mask), 15);
    EXPECT_EQ(popcount(cmpResult), 128 - 8);
    printInBinary(mask);
    auto revMask = ~mask;
    printInBinary(revMask);
    auto firstError = _tzcnt_u32(revMask);
    std::cout << firstError << std::endl;
    EXPECT_EQ(firstError, 1);
}


TEST(Movemask, Eq256) {
    __m256i target = _mm256_loadu_si256(reinterpret_cast<const __m256i*>("ttttyyyyyyyyttttttttttttyyyyyyyy"));
    __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>("ttttyyyyyyyyttttttttttttyyyyyyyy"));
    __m256i cmpResult = _mm256_cmpeq_epi8(chunk, target);
    unsigned int mask = _mm256_movemask_epi8(cmpResult);
    EXPECT_EQ(mask, 0xFFFFFFFF);
    EXPECT_EQ(std::popcount(mask), 32);
}

TEST(Movemask, Ne256) {
    __m256i target = _mm256_loadu_si256(reinterpret_cast<const __m256i*>("ttttyyyyyyyyttttttttttttyyyyyyyy"));
    __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>("ttttyyyyyyyyttxtttttttttyyyyyyyy"));
    __m256i cmpResult = _mm256_cmpeq_epi8(chunk, target);
    unsigned int mask = _mm256_movemask_epi8(cmpResult);
    EXPECT_NE(mask, 0xFFFFFFFF);
    EXPECT_EQ(std::popcount(mask), 31);
    auto revMask = ~mask;
    auto firstError = _tzcnt_u32(revMask);
    EXPECT_EQ(firstError, 14);
}

