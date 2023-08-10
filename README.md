### Introduction

This project serves as a Proof-of-Concept (PoC) for leveraging SIMD optimization in a specific scenario:
```
Compare two strings and return true if you can get one from another with a single operation, such as:
- Deleting one character;
- Adding one character;
- Replacing one character.
```
   
This project explores the potential performance benefits of SIMD operations in string comparison tasks.

SIMD (Single Instruction, Multiple Data) represents a paradigm in data processing 
that allows the simultaneous execution of the same operation (instruction) on more than one data point.
In essence, it utilizes data-level parallelism, rather than task-level parallelism.

Consider a simple operation like adding two lists of numbers. 
In a conventional scalar operation, you would iterate over each element of the list and perform the addition one by one.
With SIMD, however, entire lists can be added together in one operation, greatly boosting computational speed.

SIMD works by utilizing wide data registers in modern CPUs that can store multiple data points, such as vectors of numbers. 
A single SIMD instruction can then perform the same operation on each element in the register in parallel.

For example, if the CPU supports 128-bit wide registers, you could store four 32-bit integers in a single register.
For instance, a single SIMD operation could simultaneously add these four integers to another set of four integers stored in a different register.

SIMD offers a powerful means of enhancing computational speed by allowing for operations to be performed on multiple data points concurrently. 
It is a cornerstone of modern high-performance computing and forms the backbone of many everyday applications. 

The SSE and AVX are instruction sets that extend the SIMD capabilities of Intel and AMD processors.
The use of these extensions can greatly speed up your code, 
especially when dealing with tasks like comparing arrays or strings, or processing image and audio data. 
Here, we will focus on how to utilize SSE and AVX SIMD operations to speed up the comparison of strings.

### SSEn
SSE (Streaming SIMD Extensions)

The SSE instruction set introduces 8 dedicated 128-bit registers (XMM0 to XMM7) to the processor's architecture. 
These registers support concurrent computation on multiple data points, significantly accelerating certain types of operations.

Each 128-bit register supports multiple data types. Specifically, it can process up to 
4 single-precision (32-bit) floating-point numbers, 
2 double-precision (64-bit) numbers, 
or integer data of varying sizes, ranging from 8 to 128 bits.

For example, an operation, such as vector addition, can process on 
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
- Size of comparable strings can have difference 0 or 1
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
This code already contains optimizations and removes branching from the loop.

### Add SIMD
When comparing strings, typically, we need to compare each character in sequence until we either find a mismatch or reach the end of the shortest string.
For large strings, sequential comparison can be time-consuming.
With SSE or AVX, we can accelerate this process by loading chunks of the strings (16 bytes for SSE or 32 bytes for AVX) into SIMD registers
and comparing these chunks simultaneously.

For strings that aren't multiples of the SIMD register size, we handle the remaining characters using the traditional method. 
Also, if a string is smaller than the SIMD register size, we would use the traditional comparison method.

In theory, using SIMD operations, we can achieve up to a 16x (or 32x) speedup for infinite length strings since we're processing 16 (or 32) characters in a single operation. 
However, in reality, this optimal speedup isn't always achievable due to overheads associated with SIMD operations.
It's essential to remember that SIMD operations can sometimes be slower than regular operations depending on the CPU, 
so performance gains aren't always guaranteed. 
SIMD instructions might be more expensive due to factors such as loading and storing data to and from SIMD registers, 
potential data alignment issues, or additional instructions required for handling edge cases.

It is important to benchmark your SIMD-optimized code on your target system. 
Performance can vary greatly between different systems and even between different processors from the same manufacturer. 
By benchmarking, you can ensure that your optimizations are indeed improving performance and not unintentionally slowing down your program.

Lastly, it's also important to verify the correctness of your SIMD-optimized code. 
SIMD optimizations can introduce new types of bugs, especially around the handling of edge cases and data alignment. 
Always thoroughly test your code to ensure it behaves as expected.

```c++
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


This is the slow AVX version:
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

// end using no simd method
auto pos = minSize - (minSize % 32);
return slow(lhs.data() + pos + (oneError && !oneSize), lhs.end(), rhs.data() + pos, rhs.end());
```

### Continue optimizations
- Divide the method to handle cases of equal-sized strings separately from those of different sizes

One approach to enhance performance in code is to split the method into two distinct cases, specifically when handling strings of the same size versus different sizes.
The primary advantage of this strategy is that it enables the compiler to avoid unnecessary branching and computations. 

If we know beforehand that two strings are of equal length, we can remove the conditional checks related to string size, 
thereby improving the speed of our function. 
Similarly, handling different size strings separately allows for specific optimizations relevant to that case only.

- Split method to handle cases without error and after first error

This technique has the advantage of removing branches within the inner loop, further increasing the function's speed. 
The case without error can run without any branching, optimizing the most common scenario. 
If an error occurs, we switch to the second case, which includes error handling code.

- Instead of using a full batch rollback for errors in cases of different sizes, locate the error position by counting the number of trailing least significant zero bits

When handling strings of different sizes, a common error to encounter is an out-of-bounds access.
A typical approach to handle this error might be to rollback the entire batch of operations, but this can be costly.
A more optimized approach is to handle this error in the same batch by finding the position of the error within the batch. 
This technique avoids unnecessary rollback operations, saving computation time. 
We can determine the position by counting the number of trailing least significant zero bits (using a function like _tzcnt_u32 mentioned earlier). 
Using this approach, we only have to handle the mismatch without reverting the entire batch of comparisons, thereby saving valuable computation time.

- [Possible] For data alignment, consider using the _mmX_load_siX instruction

I employ the `_mmX_loadu_siX` instruction to load data into registers, where the `loadu` suffix indicates the loading of unaligned data. For strings of equal lengths, there's potential for manual data alignment. We can iteratively process the first `n` elements until the address becomes divisible by the desired alignment.
In scenarios with unequal string lengths, our approach can remain consistent until the first discrepancy is encountered. Beyond this point, only one of the strings can be aligned. While this optimization might slightly reduce performance for shorter strings, it can potentially enhance speed for longer ones.
It's worth noting that, on my specific CPU, the performance difference between loading aligned and unaligned data is marginal. Given this observation, I opted not to integrate this alignment optimization into the implementation.


Same size sse implementation:
```c++
unsigned countOfErrors(__m128i v) noexcept {
    unsigned mask = _mm_movemask_epi8(v);
    return 16 - std::popcount(mask);
}

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
```
linux
Run on (16 X 4369.92 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x8)
  L1 Instruction 32 KiB (x8)
  L2 Unified 512 KiB (x8)
  L3 Unified 4096 KiB (x2)
```

| Method            | EQ 15    | Diff 15   | EQ 45    | DIFF 45   | EQ 1285   | DIFF 1285   | EQ 120kb   | DIFF 120kb   | Speedup(DIFF 1285)   |
|-------------------|----------|-----------|----------|-----------|-----------|-------------|------------|--------------|----------------------|
| slow windows      | 13.3ns   | 65.0ns    | 36.2ns   | 168ns     | 981ns     | 3.973us     | 89.888us   | 0.351ms      | ~0.53x               |
| fast windows      | 7.29ns   | 44.7ns    | 14.8ns   | 93.8ns    | 314ns     | 2.122us     | 29.091us   | 0.190ms      | 1.x                  |
| sse windows       | 11.3ns   | 76.6ns    | 12.8ns   | 93.8ns    | 72.1ns    | 405ns       | 6.011us    | 30.408us     | ~5.24x               |
| sseFast windows   | 12.7ns   | 70.4ns    | 12.5ns   | 66.2ns    | 48.5ns    | 267ns       | 3.938us    | 20.832us     | ~7.94x               |
| avx windows       | 12.1ns   | 78.7ns    | 12.2ns   | 93.6ns    | 41.2ns    | 319ns       | 3.177us    | 22.977us     | ~6.65x               |
| avxFast windows   | 13.2ns   | 69.1ns    | 12.5ns   | 67.2ns    | 28.1ns    | 152ns       | 2.074us    | 11.079ns     | ~13.96x              |
| ----------------- | -------- | --------- | -------- | --------- | --------- | ----------- | ---------- | ------------ | -------------------- |
| slow Linux        | 18.6ns   | 100ns     | 69.8ns   | 353ns     | 2.389us   | 11.879us    | 0.229ms    | 1.139ms      | ~0.21x               |
| fast Linux        | 5.79ns   | 42.3ns    | 13.4ns   | 100ns     | 305ns     | 2.575us     | 28.452us   | 0.241ms      | 1.x                  |
| sse  Linux        | 19.8ns   | 103ns     | 19.0ns   | 111ns     | 65.8ns    | 346ns       | 5.709us    | 28.753us     | ~7.44x               |
| sseFast Linux     | 6.48ns   | 41.5ns    | 7.84ns   | 48.2ns    | 59.6ns    | 278ns       | 5.383us    | 24.346us     | ~9.26x               |
| avx Linux         | 19.9ns   | 103ns     | 17.8ns   | 109ns     | 37.4ns    | 256ns       | 2.861us    | 20.409us     | ~10.x                |
| avxFast Linux     | 6.02ns   | 43.6ns    | 6.48ns   | 46.9ns    | 29.0ns    | 146ns       | 2.622us    | 12.611us     | ~17.63x              |

