#include <iostream>
#include <utility>
#include <immintrin.h>
#include <cassert>

using it = std::string_view::const_iterator;

bool oneChangeSlow(std::string_view lhs, std::string_view rhs) noexcept {
    if (lhs.size() < rhs.size()) {
        return oneChangeSlow(rhs, lhs);
    }
    if (lhs.size() - rhs.size() > 1) {
        return false;
    }

    const auto minSize = rhs.size();
    const bool oneSize = lhs.size() == minSize;
    bool oneError = false;
    for (size_t i = 0; i != minSize; ++i) {
        if (lhs[i + (oneError && !oneSize)] != rhs[i]) {
            if (std::exchange(oneError, true)) {
                return false;
            }
            i -= !oneSize;
        }
    }

    return true;
}

bool oneChangeNoSIMDFast(std::string_view lhs, std::string_view rhs) noexcept {
    std::string_view::const_iterator re, rb, le, lb;
    if (lhs.size() > rhs.size()) {
        if (lhs.size() - rhs.size() > 1) {
            return false;
        }
        re = rhs.end();
        rb = rhs.begin();
        le = lhs.end();
        lb = lhs.begin();
    } else {
        if (rhs.size() - lhs.size() > 1) {
            return false;
        }
        re = lhs.end();
        rb = lhs.begin();
        le = rhs.end();
        lb = rhs.begin();
    }

    const size_t size = re - rb;
    const auto oneSizeLocal = (re - rb) == (le - lb);
    size_t i = 0;
    bool oneError = false;
    const auto fnOneSizeOneError = [&]() {
        for (; i != size; ++i) {
            if (lb[i] != rb[i]) {
                return false;
            }
        }
        return true;
    };
    const auto fnOneSizeNoError = [&]() {
        for (; i != size; ++i) {
            if (lb[i] != rb[i]) {
                oneError = true;
                ++i;
                return fnOneSizeOneError();
            }
        }
        return true;
    };
    const auto fnDiffSizeOneError = [&]() {
        for (; i != size; ++i) {
            if (lb[i + 1] != rb[i]) {
                return false;
            }
        }
        return true;
    };
    const auto fnDiffSizeNoError = [&]() {
        for (; i != size; ++i) {
            if (lb[i] != rb[i]) {
                oneError = true;
                return fnDiffSizeOneError();
            }
        }
        return true;
    };
    if (oneSizeLocal) {
        return fnOneSizeNoError();
    } else {
        return fnDiffSizeNoError();
    }
}


unsigned popcount(__m128i v) {
    const __m128i lookup = _mm_setr_epi8(
            /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
            /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
            /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
            /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4
    );
    __m128i low_mask = _mm_set1_epi8(0x0f);
    __m128i lo  = _mm_and_si128(v, low_mask);
    __m128i hi  = _mm_and_si128(_mm_srli_epi16(v, 4), low_mask);
    __m128i popcnt1 = _mm_shuffle_epi8(lookup, lo);
    __m128i popcnt2 = _mm_shuffle_epi8(lookup, hi);
    auto result = _mm_sad_epu8(_mm_add_epi8(popcnt1, popcnt2), _mm_setzero_si128());
    return _mm_extract_epi64(result, 0) + _mm_extract_epi64(result, 1);
}


bool oneChange(std::string_view lhs, std::string_view rhs) noexcept {
    if (lhs.size() < rhs.size()) {
        return oneChange(rhs, lhs);
    }

    const auto maxSize = lhs.size();
    const auto minSize = rhs.size();
    if (maxSize - minSize > 1) {
        return false;
    }

    const bool oneSize = maxSize == minSize;
    bool oneError = false;

    const auto slow = [&oneError](it lb, it le, it rb, it re) noexcept {
        const size_t size = re - rb;
        const auto oneSizeLocal = (re - rb) == (le - lb);
        for (size_t i = 0; i != size; ++i) {
            if (lb[i + (oneError && !oneSizeLocal)] != rb[i]) {
                if (std::exchange(oneError, true)) {
                    return false;
                }
                i -= !oneSizeLocal;
            }
        }
        return true;
    };


    for (size_t i = 0; i + 16 <= minSize; i += 16) {
        __m128i target = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lhs.data() + i + (oneError && !oneSize)));
        __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rhs.data() + i));
        __m128i cmpResult = _mm_cmpeq_epi8(chunk, target);
        auto count = (128 - popcount(cmpResult)) / 8;

        if (count != 0) {
            if (oneError || (oneSize && count > 1)) {
                return false;
            } else if (count == 1 && oneSize) {
                oneError = true;
            } else if (!slow(lhs.data() + i, lhs.data() + i + 16 + 1, rhs.data() + i, rhs.data() + i + 16)
                        || (--i, false)) {
                // count > 1 && different size
                return false;
            }
        }
    }

    auto pos = minSize - (minSize % 16);
    return slow(lhs.data() + pos + (oneError && !oneSize), lhs.end(), rhs.data() + pos, rhs.end());
}


bool oneChangeSameSize(std::string_view lhs, std::string_view rhs) noexcept {
    assert(lhs.size() == rhs.size());
    bool oneError = false;
    const auto size = lhs.size();
    const auto slow = [&oneError](it lb, it le, it rb, it re) noexcept {
        const size_t size = re - rb;
        const auto oneSizeLocal = (re - rb) == (le - lb);
        for (size_t i = 0; i != size; ++i) {
            if (lb[i + (oneError && !oneSizeLocal)] != rb[i]) {
                if (std::exchange(oneError, true)) {
                    return false;
                }
                i -= !oneSizeLocal;
            }
        }
        return true;
    };

    for (size_t i = 0; i + 16 <= size; i += 16) {
        __m128i target = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lhs.data() + i));
        __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rhs.data() + i));
        __m128i cmpResult = _mm_cmpeq_epi8(chunk, target);
        auto count = (128 - popcount(cmpResult)) / 8;

        if (count != 0) {
            if (count > 1 || std::exchange(oneError, true)) {
                return false;
            }
        }
    }

    auto pos = size - (size % 16);
    return slow(lhs.data() + pos, lhs.end(), rhs.data() + pos, rhs.end());
}

bool oneChangeDiffSize(std::string_view lhs, std::string_view rhs) noexcept {
    assert(lhs.size() > rhs.size());
    if (lhs.size() - rhs.size() != 1) {
        return false;
    }

    bool oneError = false;
    const auto minSize = rhs.size();
    const auto slow = [&oneError](it lb, it le, it rb, it re) noexcept {
        const size_t size = re - rb;
        const auto oneSizeLocal = (re - rb) == (le - lb);
        for (size_t i = 0; i != size; ++i) {
            if (lb[i + (oneError && !oneSizeLocal)] != rb[i]) {
                if (std::exchange(oneError, true)) {
                    return false;
                }
                i -= !oneSizeLocal;
            }
        }
        return true;
    };

    for (size_t i = 0; i + 16 <= minSize; i += 16) {
        __m128i target = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lhs.data() + i + oneError));
        __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rhs.data() + i));
        __m128i cmpResult = _mm_cmpeq_epi8(chunk, target);
        auto count = (128 - popcount(cmpResult)) / 8;

        if (count != 0) {
            if (oneError
                    || (!slow(lhs.data() + i, lhs.data() + i + 16 + 1, rhs.data() + i, rhs.data() + i + 16)
                    || (--i, false))) {
                return false;
            }
        }
    }

    auto pos = minSize - (minSize % 16);
    return slow(lhs.data() + pos + oneError, lhs.end(), rhs.data() + pos, rhs.end());
}


bool oneChangeSplit(std::string_view lhs, std::string_view rhs) noexcept {
    if (lhs.size() == rhs.size()) {
        return oneChangeSameSize(lhs, rhs);
    } else if (lhs.size() < rhs.size()) {
        return oneChangeDiffSize(rhs, lhs);
    } else {
        return oneChangeDiffSize(lhs, rhs);
    }
}

bool slowF(bool& oneError, it lb, it le, it rb, it re) noexcept {
    const size_t size = re - rb;
    const auto oneSizeLocal = (re - rb) == (le - lb);
    size_t i = 0;
    const auto fnOneSizeOneError = [&]() {
        for (; i != size; ++i) {
            if (lb[i] != rb[i]) {
                return false;
            }
        }
        return true;
    };
    const auto fnOneSizeNoError = [&]() {
        for (; i != size; ++i) {
            if (lb[i] != rb[i]) {
                oneError = true;
                ++i;
                return fnOneSizeOneError();
            }
        }
        return true;
    };
    const auto fnDiffSizeOneError = [&]() {
        for (; i != size; ++i) {
            if (lb[i + 1] != rb[i]) {
                return false;
            }
        }
        return true;
    };
    const auto fnDiffSizeNoError = [&]() {
        for (; i != size; ++i) {
            if (lb[i] != rb[i]) {
                oneError = true;
                return fnDiffSizeOneError();
            }
        }
        return true;
    };
    if (oneSizeLocal) {
        if (oneError) {
            return fnOneSizeOneError();
        } else {
            return fnOneSizeNoError();
        }
    } else {
        if (oneError) {
            return fnDiffSizeOneError();
        } else {
            return fnDiffSizeNoError();
        }
    }
};


bool oneChangeSameSizeFast(std::string_view lhs, std::string_view rhs) noexcept {
    assert(lhs.size() == rhs.size());
    bool oneError = false;
    const auto size = lhs.size();

    for (size_t i = 0; i + 16 <= size; i += 16) {
        __m128i target = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lhs.data() + i));
        __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rhs.data() + i));
        __m128i cmpResult = _mm_cmpeq_epi8(chunk, target);
        auto mask = _mm_movemask_epi8(cmpResult);

        if (mask != 0xffff) {
            auto count = (128 - popcount(cmpResult)) / 8;

            if (count > 1 || std::exchange(oneError, true)) {
                return false;
            }
        }
    }

    auto pos = size - (size % 16);
    return slowF(oneError, lhs.data() + pos, lhs.end(), rhs.data() + pos, rhs.end());
}


bool oneChangeDiffSizeFast(std::string_view lhs, std::string_view rhs) noexcept {
    assert(lhs.size() > rhs.size());
    if (lhs.size() - rhs.size() != 1) {
        return false;
    }

    bool oneError = false;
    const auto minSize = rhs.size();
    size_t i = 0;
    const auto fnOneError = [&]() {
        for (; i + 16 <= minSize; i += 16) {
            __m128i target = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lhs.data() + i + 1));
            __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rhs.data() + i));
            __m128i cmpResult = _mm_cmpeq_epi8(chunk, target);
            auto mask = _mm_movemask_epi8(cmpResult);

            if (mask != 0xffff) {
                return false;
            }
        }
        auto pos = minSize - (minSize % 16);
        return slowF(oneError, lhs.data() + pos + oneError, lhs.end(), rhs.data() + pos, rhs.end());
    };

    const auto fnNoError = [&]() {
        for (; i + 16 <= minSize; i += 16) {
            __m128i target = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lhs.data() + i + oneError));
            __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rhs.data() + i));
            __m128i cmpResult = _mm_cmpeq_epi8(chunk, target);
            auto mask = _mm_movemask_epi8(cmpResult);

            if (mask != 0xffff) {
                if (slowF(oneError, lhs.data() + i, lhs.data() + i + 16 + 1, rhs.data() + i, rhs.data() + i + 16)) {
                    i += 16;
                    return fnOneError();
                } else {
                    return false;
                }
            }

        }
        auto pos = minSize - (minSize % 16);
        return slowF(oneError, lhs.data() + pos + oneError, lhs.end(), rhs.data() + pos, rhs.end());
    };

    return fnNoError();
}

bool oneChangeFast(std::string_view lhs, std::string_view rhs) noexcept {
    if (lhs.size() == rhs.size()) {
        return oneChangeSameSizeFast(lhs, rhs);
    } else if (lhs.size() < rhs.size()) {
        return oneChangeDiffSizeFast(rhs, lhs);
    } else {
        return oneChangeDiffSizeFast(lhs, rhs);
    }
}
