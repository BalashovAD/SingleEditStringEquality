### Introduction

This project is a Proof-of-Concept (PoC) implementation of SIMD optimization for a simple task: 
compare two strings and return true if you can get one from another with a single operation, 
such as delete, add, or change one symbol.   
This project explores the potential performance benefits of SIMD operations in string comparison tasks.

SIMD (Single Instruction, Multiple Data) represents a paradigm in data processing 
that allows the simultaneous execution of the same operation (instruction) on more than one data point.
Thus, such machines exploit data-level parallelism, as opposed to task-level parallelism.

### SSEn
SSE (Streaming SIMD Extensions)

The Streaming SIMD Extensions (SSE) instruction set introduces 8 dedicated 128-bit registers (XMM0 to XMM7) to the processor's architecture. 
These registers support concurrent computation on multiple data points, significantly accelerating certain types of operations.

Each 128-bit register can handle multiple data types. Specifically, it can process up to 
4 single-precision (32-bit) floating-point numbers, 
2 double-precision (64-bit) numbers, 
or integer data of varying sizes, ranging from 8 to 128 bits.

For example, an operation like vector addition could be performed on 
four 32-bit floating-point numbers simultaneously using a single SSE instruction, 
vastly increasing the throughput for such operations.

### AVXn 
Advanced Vector Extensions (AVX) advance the SIMD capabilities even further, 
primarily by increasing the register size. 
AVX introduces 256-bit wide YMM registers, thus augmenting the amount of data that can be processed simultaneously.

These larger registers allow for processing up to 8 single-precision (32-bit) or 4 double-precision (64-bit) floating-point numbers, 
or integer data of varying sizes, from 8 to 256 bits, concurrently.
This capability greatly improves the throughput of such operations and, thus, 
the overall performance of tasks that can exploit this form of parallelism.

Later versions of AVX introduced even wider registers (512 bits in AVX-512), 
further extending the capabilities of SIMD operations.

By optimizing for SSE and AVX instructions, 
applications like this project's string comparison task can achieve higher levels of performance and efficiency, 
especially when dealing with large amounts of data. 
Implementations using these instruction sets can lead to significant reductions in computation time and improved parallelism, 
making them highly advantageous for a wide range of applications.

### Basic solution
The key points:
- Size of equal string can have difference 0 or 1
- Operation of adding or deleting is similar, stay only deleting
- if size difference is 1 - op should be deleting
- suggest that lhs is always bigger than rhs

```c++
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
        if (lhs[i + (oneError && !oneSize)] != rhs[i]) { // unlikely
            if (std::exchange(oneError, true)) {
                return false;
            }
            i -= !oneSize;
        }
    }

    return true;
}
```
This code is already contains some optimization and removing branch from loop.  

### Add SIMD
In the first iteration, we will use the 16, or 32 for AVX, packed 8 bits numbers (`unsigned char`) 
to make fast comparing operation. Handle the part less than the register size with old method.
```
std::string_view f = "aaaaaaaaaaaaaaaa"; // 16 'a' == 0x61 == 97 one bite char string
std::string_view s = "abaaaaaaaaaaaaaa"; // 'b' == 0x62 == 98
__m128i target = _mm_loadu_si128(reinterpret_cast<const __m128i*>(f.data())); // read 16 bites
// target = {0x61, 0x61, 0x61, 0x61, 0x61, 0x61, 0x61, 0x61, 0x61, 0x61, 0x61, 0x61, 0x61, 0x61, 0x61, 0x61} 16 times 
__m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s.data())); // read 16 bites
// chunck = {0x61, 0x62, 0x61, 0x61, 0x61, 0x61, 0x61, 0x61, 0x61, 0x61, 0x61, 0x61, 0x61, 0x61, 0x61, 0x61}
__m128i cmpResult = _mm_cmpeq_epi8(chunk, target);
// {0xFF, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}
int mask = _mm_movemask_epi8(cmpResult);
// 0b00000000000000001111111111111101 // 16 first bits are always '0' because we moved mask from 16 bites
mask = ~mask;
// 0b1111111111111111000000000000010
auto firstError = _tzcnt_u32(mask); // firstError == 1
```
| Operation  | 1st byte | 2nd byte | 3rd byte | 4th byte | 5th byte | 6th byte | 7th byte | 8th byte | 9th byte | 10th byte | 11th byte | 12th byte | 13th byte | 14th byte | 15th byte | 16th byte |
|------------|----------|----------|----------|----------|----------|----------|----------|----------|----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| f (target) | 0x61     | 0x61     | 0x61     | 0x61     | 0x61     | 0x61     | 0x61     | 0x61     | 0x61     | 0x61      | 0x61      | 0x61      | 0x61      | 0x61      | 0x61      | 0x61      |
| s (chunk)  | 0x61     | 0x62     | 0x61     | 0x61     | 0x61     | 0x61     | 0x61     | 0x61     | 0x61     | 0x61      | 0x61      | 0x61      | 0x61      | 0x61      | 0x61      | 0x61      |
| Compare    | 0xFF     | 0x00     | 0xFF     | 0xFF     | 0xFF     | 0xFF     | 0xFF     | 0xFF     | 0xFF     | 0xFF      | 0xFF      | 0xFF      | 0xFF      | 0xFF      | 0xFF      | 0xFF      |
| Mask       | 0xFD     | 0xFF     | 0x00     | 0x00     | -        | -        | -        | -        | -        | -         | -         | -         | -         | -         | -         | -         |
| ~Mask      | 0x2      | 0x00     | 0xFF     | 0xFF     | -        | -        | -        | -        | -        | -         | -         | -         | -         | -         | -         | -         |


This is the AVX version:
```c++
for (size_t i = 0; i + 32 <= minSize; i += 32) {
    __m256i target = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lhs.data() + i + (oneError && !oneSize)));
    __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rhs.data() + i));
    __m256i cmpResult = _mm256_cmpeq_epi8(chunk, target); // set 0xFF to i bite if i bites are equal
    auto count = countOfErrors(cmpResult);
    // unsigned mask = _mm256_movemask_epi8(v); // get the major bit from every 8 bit number of v
    // for entirely equal strings cmpResult = {0xFF, 0xFF, ...} 32 elements
    // mask = 0xFFFFFFFF
    // auto count = 32 - std::popcount(mask);

    if (count != 0) { // unlikely
        if (oneError || (oneSize && count > 1)) {
            return false;
        } else if (count == 1 && oneSize) {
            oneError = true;
        } else if (!slow(lhs.data() + i, lhs.data() + i + 32 + 1, rhs.data() + i, rhs.data() + i + 32)
                   || (--i, false)) {
            // count > 1 && different size
            // rollback to slow method for this batch
            return false;
        }
    }
}

// end using slow method
auto pos = minSize - (minSize % 32);
return slow(lhs.data() + pos + (oneError && !oneSize), lhs.end(), rhs.data() + pos, rhs.end());
```
### Continue optimizations
- Split method to same size and different size cases to remove branches and computations
- Split method to without error and case after first error
- Don't use full batch rollback for error in diff size case, find position as counts the number of trailing least significant zero bits

Same size sse implementation:
```c++
bool oneChangeSameSizeFast(std::string_view lhs, std::string_view rhs) noexcept {
    assert(lhs.size() == rhs.size());
    bool oneError = false;
    const auto size = lhs.size();

    size_t i = 0;
    for (; i + 16 <= size; i += 16) {
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

    return slowF(oneError, lhs.data() + i, lhs.end(), rhs.data() + i, rhs.end());
}
```
Diff size:
```c++
bool oneChangeDiffSizeFast(std::string_view lhs, std::string_view rhs) noexcept {
    assert(lhs.size() > rhs.size());
    if (lhs.size() - rhs.size() != 1) {
        return false;
    }

    const auto minSize = rhs.size();
    size_t i = 0;
    const auto fnOneError = [&]() {
        for (; i + 16 <= minSize; i += 16) {
            __m128i target = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lhs.data() + i + 1));
            __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rhs.data() + i));
            __m128i cmpResult = _mm_cmpeq_epi8(chunk, target);
            int mask = _mm_movemask_epi8(cmpResult);

            if (mask != 0x0000ffff) [[unlikely]] {
                return false;
            }
        }
        return slowF(true, lhs.data() + i + 1, lhs.end(), rhs.data() + i, rhs.end());
    };

    const auto fnNoError = [&]() {
        for (; i + 16 <= minSize; i += 16) {
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

        return slowF(false, lhs.data() + i, lhs.end(), rhs.data() + i, rhs.end());
    };

    return fnNoError();
}
```
### Benchmark

```
Ryzen 7 3700X, windows MinGW 13.1
Run on (16 X 3593 MHz CPU s)
CPU Caches:
L1 Data 32 KiB (x8)
L1 Instruction 32 KiB (x8)
L2 Unified 512 KiB (x8)
L3 Unified 16384 KiB (x2)
```

| Method     | EQ 15        | Diff 15        | EQ 45      | DIFF 45      | EQ 1285     | DIFF 1285     | EQ 120kb   | DIFF 120kb   | Speedup(1285)           |
|------------|--------------|----------------|------------|--------------|-------------|---------------|------------|--------------|-------------------------|
| slow       | 9.91ns       | 58.8ns         | 26.2ns     | 146ns        | 671ns       | 3361ns        | 62688ns    | 314956ns     | 1x                      |
| fast       | 12.1ns       | 47.7ns         | 26.9ns     | 101ns        | 662ns       | 2073ns        | 58715ns    | 192337ns     | ~1.62x                  |
| sse        | 12.0ns       | 71.1ns         | 12.8ns     | 83.9ns       | 70.6ns      | 395ns         | 6051ns     | 30713ns      | ~8.51x                  |
| avx        | 11.5ns       | 68.8ns         | 11.8ns     | 89.2ns       | 39.5ns      | 299ns         | 3120ns     | 22495ns      | ~11.24x                 |
| sseFast    | 9.31ns       | 58.2ns         | 8.87ns     | 73.3ns       | 63.8ns      | 311ns         | 4780ns     | 22712ns      | ~10.81x                 |
| avxFast    | 9.39ns       | 60.3ns         | 9.32ns     | 73.8ns       | 64.7ns      | 282ns         | 4693ns     | 19740ns      | ~11.92x                 |


