#pragma once

#include <string_view>

bool oneChangeSlow(std::string_view lhs, std::string_view rhs) noexcept;
bool oneChange(std::string_view lhs, std::string_view rhs) noexcept;
bool oneChangeSplit(std::string_view lhs, std::string_view rhs) noexcept;