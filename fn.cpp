#include <iostream>
#include <utility>
#include <immintrin.h>
#include <cassert>
#include <bit>
#include <cstring>


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

unsigned popcount(__m128i v) noexcept {
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


unsigned popcount(__m256i v) noexcept {
    const __m256i mask1 = _mm256_set1_epi64x(0x5555555555555555LL);
    const __m256i mask2 = _mm256_set1_epi64x(0x3333333333333333LL);
    const __m256i mask3 = _mm256_set1_epi64x(0x0F0F0F0F0F0F0F0FLL);
    const __m256i mask4 = _mm256_set1_epi64x(0x00FF00FF00FF00FFLL);
    const __m256i mask5 = _mm256_set1_epi64x(0x0000FFFF0000FFFFLL);
    const __m256i mask6 = _mm256_set1_epi64x(0x00000000FFFFFFFFLL);

    v = _mm256_sub_epi64(v, _mm256_and_si256(_mm256_srli_epi64(v, 1), mask1));
    v = _mm256_add_epi64(_mm256_and_si256(v, mask2), _mm256_and_si256(_mm256_srli_epi64(v, 2), mask2));
    v = _mm256_and_si256(_mm256_add_epi64(v, _mm256_srli_epi64(v, 4)), mask3);
    v = _mm256_add_epi64(_mm256_and_si256(v, mask4), _mm256_and_si256(_mm256_srli_epi64(v, 8), mask4));
    v = _mm256_add_epi64(_mm256_and_si256(v, mask5), _mm256_and_si256(_mm256_srli_epi64(v, 16), mask5));
    v = _mm256_add_epi64(_mm256_and_si256(v, mask6), _mm256_and_si256(_mm256_srli_epi64(v, 32), mask6));

    unsigned long long count =
            _mm256_extract_epi64(v, 0) +
            _mm256_extract_epi64(v, 1) +
            _mm256_extract_epi64(v, 2) +
            _mm256_extract_epi64(v, 3);

    return count;
}

unsigned countOfErrors(__m128i v) noexcept {
    unsigned mask = _mm_movemask_epi8(v);
    return 16 - std::popcount(mask);
}


unsigned countOfErrors(__m256i v) noexcept {
    unsigned mask = _mm256_movemask_epi8(v);
    return 32 - std::popcount(mask);
}

unsigned findFirstError(unsigned int mask) noexcept {
    return _tzcnt_u32(~mask);
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
        auto count = countOfErrors(cmpResult);

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


bool oneChangeAVX(std::string_view lhs, std::string_view rhs) noexcept {
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


    for (size_t i = 0; i + 32 <= minSize; i += 32) {
        __m256i target = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lhs.data() + i + (oneError && !oneSize)));
        __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rhs.data() + i));
        __m256i cmpResult = _mm256_cmpeq_epi8(chunk, target);
        auto count = countOfErrors(cmpResult);

        if (count != 0) {
            if (oneError || (oneSize && count > 1)) {
                return false;
            } else if (count == 1 && oneSize) {
                oneError = true;
            } else if (!slow(lhs.data() + i, lhs.data() + i + 32 + 1, rhs.data() + i, rhs.data() + i + 32)
                       || (--i, false)) {
                // count > 1 && different size
                return false;
            }
        }
    }

    auto pos = minSize - (minSize % 32);
    return slow(lhs.data() + pos + (oneError && !oneSize), lhs.end(), rhs.data() + pos, rhs.end());
};


bool slowF(it lb, it le, it rb, it re) noexcept {
    const auto size = re - rb;
    assert(le - lb == size + 1);
    if (size == 0) {
        return true;
    }
    int i = 0;
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
                return fnDiffSizeOneError();
            }
        }
        return true;
    };

    return fnDiffSizeNoError();
}

__m128i load16(char const*const begin, size_t size) noexcept {
    static constexpr auto STEP_SIZE = 16;
    assert(size <= STEP_SIZE);
    char buffer[STEP_SIZE];
    memset(buffer, 0, STEP_SIZE);
    memcpy(buffer, begin, size);
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(buffer));
}

__m256i load32(char const*const begin, size_t size) noexcept {
    static constexpr auto STEP_SIZE = 32;
    assert(size <= STEP_SIZE);
    alignas(STEP_SIZE) char buffer[STEP_SIZE];
    memset(buffer, 0, STEP_SIZE);
    memcpy(buffer, begin, size);
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(buffer));
}


unsigned tailEqSIMD(it lb, it rb, size_t size) noexcept {
    if (size == 0) {
        return 0;
    }

    const auto handle16 = [&]() {
        auto lhs = load16(lb, size);
        auto rhs = load16(rb, size);
        auto cmp = _mm_cmpeq_epi8(lhs, rhs);
        return countOfErrors(cmp);
    };

    const auto handle32 = [&]() {
        auto lhs = load32(lb, size);
        auto rhs = load32(rb, size);
        auto cmp = _mm256_cmpeq_epi8(lhs, rhs);
        return countOfErrors(cmp);
    };

    if (size <= 16) {
        return handle16();
    } else {
        return handle32();
    }
}


unsigned tailEqMEMCMP(it lb, it rb, size_t size) noexcept {
    unsigned diff = 0;
    for (const auto end = lb + size; lb != end; ++lb, ++rb) {
        if (*lb != *rb) {
            ++diff;
        }
    }
    return diff;
}

bool oneChangeSameSizeFast(std::string_view lhs, std::string_view rhs) noexcept {
    assert(lhs.size() == rhs.size());
    const auto size = lhs.size();
    if (size <= 16) {
        return tailEqMEMCMP(lhs.data(), rhs.data(), size) <= 1;
    }
    bool oneError = false;

    size_t i = 0;
    const auto reducedSize = size - 16;
    for (; i <= reducedSize; i += 16) {
        __m128i target = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lhs.data() + i));
        __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rhs.data() + i));
        __m128i cmpResult = _mm_cmpeq_epi8(chunk, target);
        auto count = countOfErrors(cmpResult);

        if (count != 0) [[unlikely]] {
            if (count > 1 || std::exchange(oneError, true)) {
                return false;
            }
        }
    }

    return tailEqMEMCMP(lhs.data() + i, rhs.data() + i, size - i) + oneError <= 1;
}


bool oneChangeDiffSizeFast(std::string_view lhs, std::string_view rhs) noexcept {
    assert(lhs.size() > rhs.size());
    const auto minSize = rhs.size();
    if (lhs.size() - minSize != 1) {
        return false;
    }

    if (minSize <= 16) {
        return slowF(lhs.data(), lhs.end(), rhs.data(), rhs.end());
    }

    size_t i = 0;
    const auto reducedSize = minSize - 16;
    const auto fnOneError = [&]() {
        for (; i <= reducedSize; i += 16) {
            __m128i target = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lhs.data() + i + 1));
            __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rhs.data() + i));
            __m128i cmpResult = _mm_cmpeq_epi8(chunk, target);
            int mask = _mm_movemask_epi8(cmpResult);

            if (mask != 0x0000ffff) [[unlikely]] {
                return false;
            }
        }
        return tailEqMEMCMP(lhs.data() + i + 1, rhs.data() + i, minSize - i) == 0;
    };

    const auto fnNoError = [&]() {
        for (; i <= reducedSize; i += 16) {
            __m128i target = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lhs.data() + i));
            __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rhs.data() + i));
            __m128i cmpResult = _mm_cmpeq_epi8(chunk, target);
            int mask = _mm_movemask_epi8(cmpResult);

            if (mask != 0x0000ffff) [[unlikely]] {
                auto firstError = findFirstError(mask);
                i += firstError;
                return fnOneError();
            }

        }

        return slowF(lhs.data() + i, lhs.end(), rhs.data() + i, rhs.end());
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


bool oneChangeSameSizeFastAVX(std::string_view lhs, std::string_view rhs) noexcept {
    assert(lhs.size() == rhs.size());
    bool oneError = false;
    const auto size = lhs.size();

    if (size <= 32) {
        return tailEqMEMCMP(lhs.data(), rhs.data(), size) <= 1;
    }

    size_t i = 0;
    const auto reducedSize = size - 32;
    for (; i <= reducedSize; i += 32) {
        __m256i target = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lhs.data() + i));
        __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rhs.data() + i));
        __m256i cmpResult = _mm256_cmpeq_epi8(chunk, target);
        auto count = countOfErrors(cmpResult);

        if (count != 0) [[unlikely]] {
            if (count > 1 || std::exchange(oneError, true)) {
                return false;
            }
        }
    }

    return tailEqMEMCMP(lhs.data() + i, rhs.data() + i, size - i) + oneError <= 1;
}


bool oneChangeDiffSizeFastAVX(std::string_view lhs, std::string_view rhs) noexcept {
    assert(lhs.size() > rhs.size());
    const auto minSize = rhs.size();
    if (lhs.size() - minSize != 1) {
        return false;
    }

    if (minSize <= 32) {
        return slowF(lhs.data(), lhs.end(), rhs.data(), rhs.end());
    }

    size_t i = 0;
    const auto reducedSize = minSize - 32;
    const auto fnOneError = [&]() {
        for (; i <= reducedSize; i += 32) {
            __m256i target = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lhs.data() + i + 1));
            __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rhs.data() + i));
            __m256i cmpResult = _mm256_cmpeq_epi8(chunk, target);
            unsigned int mask = _mm256_movemask_epi8(cmpResult);

            if (mask != 0xffffffff) [[unlikely]] {
                return false;
            }
        }
        return tailEqMEMCMP(lhs.data() + i + 1, rhs.data() + i, minSize - i) == 0;
    };

    const auto fnNoError = [&]() {
        for (; i <= reducedSize; i += 32) {
            __m256i target = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lhs.data() + i));
            __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rhs.data() + i));
            __m256i cmpResult = _mm256_cmpeq_epi8(chunk, target);
            unsigned int mask = _mm256_movemask_epi8(cmpResult);

            if (mask != 0xffffffff) [[unlikely]] {
                auto firstError = findFirstError(mask);
                i += firstError;
                return fnOneError();
            }
        }

        return slowF(lhs.data() + i, lhs.end(), rhs.data() + i, rhs.end());
    };

    return fnNoError();
}

bool oneChangeFastAVX(std::string_view lhs, std::string_view rhs) noexcept {
    if (lhs.size() == rhs.size()) {
        return oneChangeSameSizeFastAVX(lhs, rhs);
    } else if (lhs.size() < rhs.size()) {
        return oneChangeDiffSizeFastAVX(rhs, lhs);
    } else {
        return oneChangeDiffSizeFastAVX(lhs, rhs);
    }
}
