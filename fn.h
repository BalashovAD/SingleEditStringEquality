#pragma once

#include <string_view>
#include <immintrin.h>


bool oneChangeSlow(std::string_view lhs, std::string_view rhs) noexcept;
bool oneChangeNoSIMDFast(std::string_view lhs, std::string_view rhs) noexcept;
bool oneChange(std::string_view lhs, std::string_view rhs) noexcept;
bool oneChangeFast(std::string_view lhs, std::string_view rhs) noexcept;
bool oneChangeAVX(std::string_view lhs, std::string_view rhs) noexcept;
bool oneChangeFastAVX(std::string_view lhs, std::string_view rhs) noexcept;

unsigned popcount(__m128i v) noexcept;
unsigned popcount(__m256i v) noexcept;