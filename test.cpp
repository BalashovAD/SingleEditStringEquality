#include <gtest/gtest.h>

#include <source_location>

#include "fn.h"

using namespace testing;
using sv = std::string_view;
using fn = bool(*)(sv, sv);

class OneChangeTest : public ::testing::TestWithParam<fn> {
public:
    static constexpr sv prefix15 = "aqzwsxedcrfvrfv";
    static constexpr sv prefix16 = "aqzwsxedcrfvrfva";
    static constexpr sv prefix14 = "zwsxedc1rfvrfv";

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
    std::string location(std::source_location current) const {
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
                {"", prefix14, "0-14"},
                {prefix14, "", "14-0"},
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
    // abcd abd
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
INSTANTIATE_TEST_SUITE_P(Common, OneChangeTest, ::testing::Values(oneChange));
INSTANTIATE_TEST_SUITE_P(Split, OneChangeTest, ::testing::Values(oneChangeSplit));