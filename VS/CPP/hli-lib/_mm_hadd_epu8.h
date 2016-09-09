#pragma once

#ifdef _MSC_VER
#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
#if !defined(NOMINMAX)
#define NOMINMAX 1 
#endif
#if !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS 1
#endif
#endif

#include <limits>       // std::numeric_limits
#include <iostream>		// std::cout
#include <algorithm>	// std::min


//#include "mmintrin.h"  // mmx
#include "emmintrin.h"  // sse
#include "pmmintrin.h"  // sse3
#include "tmmintrin.h"  // ssse3
#include "smmintrin.h"  // sse4.1
#include "nmmintrin.h"  // sse4.2
#include "immintrin.h"  // avx, avx2, avx512, FP16C, KNCNI, FMA
//#include "ammintrin.h"  // AMD-specific intrinsics

#include "tools.h"
#include "timer.h"
#include "toString.h"
#include "_mm_hadd_epi64.h"
#include "_mm_rand_si128.h"

namespace hli {

	namespace priv {

		inline __m128i _mm_hadd_epu8_ref(
			const __m128i * const mem_addr,
			const size_t nBytes)
		{

			// AVX
			// vpsrldq     xmm4, xmm3, 4
			// vpmovzxbd   xmm2, xmm3
			// vpmovzxbd   xmm4, xmm4
			// vpaddd      xmm1, xmm1, xmm2
			// vpaddd      xmm0, xmm0, xmm4

			const unsigned __int8 * const ptr = reinterpret_cast<const unsigned __int8 * const>(mem_addr);
			unsigned __int32 sum = 0;
			for (size_t i = 0; i < nBytes; ++i) {
				sum += ptr[i];
				//std::cout << "INFO: hli::priv::_mm_hadd_epu8_ref: i=" << i << "; sum="<<sum << std::endl;
			}
			return _mm_set1_epi32(sum);
		}

		template <int N_BITS>
		inline __m128i _mm_hadd_epu8_method1(
			const __m128i * const mem_addr,
			const size_t nBytes)
		{
			static_assert((N_BITS > 0) && (N_BITS <= 8), "Number of bits must be in range 1 to 8.");
#			pragma warning( disable: 280) 
			switch (N_BITS) {
			case 8: return _mm_hadd_epu8_method1_nBits8(mem_addr, nBytes);
			case 7: return _mm_hadd_epu8_method1_nBits7(mem_addr, nBytes);
			case 6: return _mm_hadd_epu8_method1_nBits6(mem_addr, nBytes);
			case 5:
			case 4:
			case 3:
			case 2:
			case 1: return _mm_hadd_epu8_method1_nBits5(mem_addr, nBytes);
			default:
				return _mm_setzero_si128();
			}
		}

		inline __m128i _mm_hadd_epu8_method2(
			const __m128i * const mem_addr,
			const size_t nBytes)
		{
			//assume (nBytes < 2 ^ (32 - 8))

			const size_t nBlocks = nBytes >> 4; // divide by 16 to get the number of __m128i regs (blocks)
			const __m128i and_mask = _mm_set1_epi32(0b11111111);
			//const __m128i shuffle_mask_0 = _mm_set_epi8(15, 14, 13, 12,   11, 10, 9, 8,   7, 6, 5, 4,   3, 2, 1, 0);
			const __m128i shuffle_mask_1 = _mm_set_epi8(15, 14, 13, 13, 11, 10, 9, 9, 7, 6, 5, 5, 3, 2, 1, 1);
			const __m128i shuffle_mask_2 = _mm_set_epi8(15, 14, 13, 14, 11, 10, 9, 10, 7, 6, 5, 6, 3, 2, 1, 2);
			const __m128i shuffle_mask_3 = _mm_set_epi8(15, 14, 13, 15, 11, 10, 9, 11, 7, 6, 5, 7, 3, 2, 1, 3);

			__m128i sum = _mm_setzero_si128();

			for (size_t block = 0; block < nBlocks; ++block) {
				const __m128i d = _mm_load_si128(&mem_addr[block]);

				sum = _mm_add_epi32(sum, _mm_and_si128(d, and_mask));
				sum = _mm_add_epi32(sum, _mm_and_si128(_mm_shuffle_epi8(d, shuffle_mask_1), and_mask));
				sum = _mm_add_epi32(sum, _mm_and_si128(_mm_shuffle_epi8(d, shuffle_mask_2), and_mask));
				sum = _mm_add_epi32(sum, _mm_and_si128(_mm_shuffle_epi8(d, shuffle_mask_3), and_mask));
			}

			const __m128i sum2 = _mm_cvtepi32_epi64(_mm_hadd_epi32(sum, sum));
			return _mm_hadd_epi64(sum2);
		}

		inline __m128i _mm_hadd_epu8_method3(
			const __m128i * const mem_addr,
			const size_t nBytes)
		{
			const size_t nBlocks = nBytes >> 4; // divide by 16 to get the number of __m128i regs (blocks)
			__m128i sum = _mm_setzero_si128();
			for (size_t block = 0; block < nBlocks; ++block) {
				sum = _mm_add_epi64(sum, _mm_sad_epu8(mem_addr[block], _mm_setzero_si128()));
			}
			return _mm_hadd_epi64(sum);
		}

		inline __m128i _mm_hadd_epu8_method1_nBits8(
			const __m128i * const mem_addr,
			const size_t nBytes)
		{
			const size_t nBlocks = nBytes >> 4; // divide by 16 to get the number of __m128i regs (blocks)
			__m128i sum = _mm_setzero_si128();

			for (size_t block = 0; block < nBlocks; ++block) {
				sum = _mm_add_epi64(sum, _mm_sad_epu8(mem_addr[block], _mm_setzero_si128()));
			}
			return _mm_hadd_epi64(sum);
		}

		inline __m128i _mm_hadd_epu8_method1_nBits7(
			const __m128i * const mem_addr,
			const size_t nBytes)
		{
			const size_t nBlocks = nBytes >> 4; // divide by 16 to get the number of __m128i regs (blocks)

			__m128i sum = _mm_setzero_si128();

			for (size_t block = 0; block < nBlocks - 1; block += 2)
			{
				__m128i sum_p = _mm_add_epi8(mem_addr[block], mem_addr[block + 1]);
				sum = _mm_add_epi64(sum, _mm_sad_epu8(sum_p, _mm_setzero_si128()));
			}

			const size_t tail = nBlocks & 0b1;
			if (tail > 0) {
				for (size_t block = (nBlocks - tail); block < nBlocks; ++block) {
					//std::cout << "INFO: hli::_mm_hadd_epu8_class<7>: tail=" << tail << "; block=" << block << std::endl;
					sum = _mm_add_epi64(sum, _mm_sad_epu8(mem_addr[block], _mm_setzero_si128()));
				}
			}
			return _mm_hadd_epi64(sum);
		}

		inline __m128i _mm_hadd_epu8_method1_nBits6(
			const __m128i * const mem_addr,
			const size_t nBytes)
		{
			const int nBlocks = static_cast<int>(nBytes >> 4); // divide by 16 to get the number of __m128i regs (blocks)

			__m128i sum = _mm_setzero_si128();
			__m128i sum_p;

			for (int block = 0; block < (nBlocks - 3); block += 4)
			{
				sum_p = mem_addr[block];
				sum_p = _mm_add_epi8(sum_p, mem_addr[block + 1]);
				sum_p = _mm_add_epi8(sum_p, mem_addr[block + 2]);
				sum_p = _mm_add_epi8(sum_p, mem_addr[block + 3]);
				sum = _mm_add_epi64(sum, _mm_sad_epu8(sum_p, _mm_setzero_si128()));
			}
			const int tail = nBlocks & 0b11;
			if (tail > 0) {
				for (int block = (nBlocks - tail); block < nBlocks; ++block) {
					//std::cout << "INFO: hli:::_mm_hadd_epu8_method3: tail: block=" << block << std::endl;
					sum = _mm_add_epi64(sum, _mm_sad_epu8(mem_addr[block], _mm_setzero_si128()));
				}
			}
			return _mm_hadd_epi64(sum);
		}

		inline __m128i _mm_hadd_epu8_method1_nBits5(
			const __m128i * const mem_addr,
			const size_t nBytes)
		{
			const int nBlocks = static_cast<int>(nBytes >> 4); // divide by 16 to get the number of __m128i regs (blocks)

			__m128i sum = _mm_setzero_si128();
			__m128i sum_p;

			for (int block = 0; block < nBlocks - 7; block += 8)
			{
				sum_p = mem_addr[block];
				sum_p = _mm_add_epi8(sum_p, mem_addr[block + 1]);
				sum_p = _mm_add_epi8(sum_p, mem_addr[block + 2]);
				sum_p = _mm_add_epi8(sum_p, mem_addr[block + 3]);
				sum_p = _mm_add_epi8(sum_p, mem_addr[block + 4]);
				sum_p = _mm_add_epi8(sum_p, mem_addr[block + 5]);
				sum_p = _mm_add_epi8(sum_p, mem_addr[block + 6]);
				sum_p = _mm_add_epi8(sum_p, mem_addr[block + 7]);
				sum = _mm_add_epi64(sum, _mm_sad_epu8(sum_p, _mm_setzero_si128()));
			}

			const int tail = nBlocks & 0b111;
			if (tail > 0) {
				int startTail = nBlocks - tail;
				sum_p = mem_addr[startTail];
				for (int block = startTail + 1; block < nBlocks; ++block) {
					sum_p = _mm_add_epi8(sum_p, mem_addr[block]);
				}
				sum = _mm_add_epi64(sum, _mm_sad_epu8(sum_p, _mm_setzero_si128()));
			}
			return _mm_hadd_epi64(sum);
		}
	}

	namespace test {

		void test_mm_hadd_epu8(const size_t nBlocks, const size_t nExperiments, const bool doTests)
		{
			const size_t nBytes = resizeNBytes(16 * nBlocks, 16);
			__m128i * const mem_addr = static_cast<__m128i *>(_mm_malloc(nBytes, 16));
			fillRand_epu8<5>(mem_addr, nBytes);

			{
				double min_ref = std::numeric_limits<double>::max();
				double min1 = std::numeric_limits<double>::max();
				double min2 = std::numeric_limits<double>::max();
				double min3 = std::numeric_limits<double>::max();
				double min4 = std::numeric_limits<double>::max();
				double min5 = std::numeric_limits<double>::max();
				double min6 = std::numeric_limits<double>::max();

				for (size_t i = 0; i < nExperiments; ++i) 
				{
					timer::reset_and_start_timer();
					const __m128i result_ref = hli::priv::_mm_hadd_epu8_ref(mem_addr, nBytes);
					min_ref = std::min(min_ref, timer::get_elapsed_kcycles());

					{
						timer::reset_and_start_timer();
						const __m128i result = hli::priv::_mm_hadd_epu8_method1<8>(mem_addr, nBytes);
						min1 = std::min(min1, timer::get_elapsed_kcycles());

						if (doTests) {
							if (result_ref.m128i_u32[0] != result.m128i_u32[0]) {
								std::cout << "WARNING: test _mm_hadd_epu8_method1<8>: result-ref=" << hli::toString_u32(result_ref) << "; result=" << hli::toString_u32(result) << std::endl;
								return;
							}
						}
					}
					{
						timer::reset_and_start_timer();
						const __m128i result = hli::priv::_mm_hadd_epu8_method1<7>(mem_addr, nBytes);
						min2 = std::min(min2, timer::get_elapsed_kcycles());

						if (doTests) {
							if (result_ref.m128i_u32[0] != result.m128i_u32[0]) {
								std::cout << "WARNING: test _mm_hadd_epu8_method1<7>: result-ref=" << hli::toString_u32(result_ref) << "; result=" << hli::toString_u32(result) << std::endl;
								return;
							}
						}
					}
					{
						timer::reset_and_start_timer();
						const __m128i result = hli::priv::_mm_hadd_epu8_method1<6>(mem_addr, nBytes);
						min3 = std::min(min3, timer::get_elapsed_kcycles());

						if (doTests) {
							if (result_ref.m128i_u32[0] != result.m128i_u32[0]) {
								std::cout << "WARNING: test _mm_hadd_epu8_method1<6>: result-ref=" << hli::toString_u32(result_ref) << "; result=" << hli::toString_u32(result) << std::endl;
								return;
							}
						}
					}
					{
						timer::reset_and_start_timer();
						const __m128i result = hli::priv::_mm_hadd_epu8_method1<5>(mem_addr, nBytes);
						min4 = std::min(min4, timer::get_elapsed_kcycles());

						if (doTests) {
							if (result_ref.m128i_u32[0] != result.m128i_u32[0]) {
								std::cout << "WARNING: test _mm_hadd_epu8_method1<5>: result-ref=" << hli::toString_u32(result_ref) << "; result=" << hli::toString_u32(result) << std::endl;
								return;
							}
						}
					}
					{
						timer::reset_and_start_timer();
						const __m128i result = hli::priv::_mm_hadd_epu8_method2(mem_addr, nBytes);
						min5 = std::min(min5, timer::get_elapsed_kcycles());

						if (doTests) {
							if (result_ref.m128i_u32[0] != result.m128i_u32[0]) {
								std::cout << "WARNING: test _mm_hadd_epu8_method2: result-ref=" << hli::toString_u32(result_ref) << "; result=" << hli::toString_u32(result) << std::endl;
								return;
							}
						}
					}
					{
						timer::reset_and_start_timer();
						const __m128i result = hli::priv::_mm_hadd_epu8_method3(mem_addr, nBytes);
						min6 = std::min(min6, timer::get_elapsed_kcycles());

						if (doTests) {
							if (result_ref.m128i_u32[0] != result.m128i_u32[0]) {
								std::cout << "WARNING: test _mm_hadd_epu8_method3: result-ref=" << hli::toString_u32(result_ref) << "; result=" << hli::toString_u32(result) << std::endl;
								return;
							}
						}
					}
				}
				printf("[_mm_hadd_epu8 Ref]       : %2.5f Kcycles\n", min_ref);
				printf("[_mm_hadd_epu8_method1<8>]: %2.5f Kcycles; %2.3f times faster than ref\n", min1, min_ref / min1);
				printf("[_mm_hadd_epu8_method1<7>]: %2.5f Kcycles; %2.3f times faster than ref\n", min2, min_ref / min2);
				printf("[_mm_hadd_epu8_method1<6>]: %2.5f Kcycles; %2.3f times faster than ref\n", min3, min_ref / min3);
				printf("[_mm_hadd_epu8_method1<5>]: %2.5f Kcycles; %2.3f times faster than ref\n", min4, min_ref / min4);
				printf("[_mm_hadd_epu8_method2]:    %2.5f Kcycles; %2.3f times faster than ref\n", min5, min_ref / min5);
				printf("[_mm_hadd_epu8_method3]:    %2.5f Kcycles; %2.3f times faster than ref\n", min6, min_ref / min6);
			}

			_mm_free(mem_addr);
		}
	}

	// Horizontally add all 8-bit integers in mem_addr (with nBytes).
	// Operation:
	// tmp := sum(mem_addr)
	// dst[31:0] := tmp
	// dst[63:32] := tmp
	// dst[95:64] := tmp
	// dst[127:96] := tmp
	template <int N_BITS>
	inline __m128i _mm_hadd_epu8(
		const __m128i * const mem_addr,
		const size_t nBytes)
	{
		return priv::_mm_hadd_epu8_method1<N_BITS>(mem_addr, nBytes);
		//return priv::_mm_hadd_epu8_method2<N_BITS>(mem_addr, nBytes);
	}

	// Horizontally add all 8-bit integers in mem_addr (with nBytes).
	// Operation:
	// tmp := sum(mem_addr)
	// dst[31:0] := tmp
	// dst[63:32] := tmp
	// dst[95:64] := tmp
	// dst[127:96] := tmp
	template <int N_BITS>
	inline __m128i _mm_hadd_epu8(
		const std::tuple<__m128i * const, const size_t> data)
	{
		return _mm_hadd_epu8<N_BITS>(std::get<0>(data), std::get<1>(data));
		//return priv::_mm_hadd_epu8_method2<N_BITS>(mem_addr, nBytes);
	}

}