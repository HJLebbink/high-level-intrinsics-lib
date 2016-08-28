#pragma once

//#include "mmintrin.h"  // mmx
#include "emmintrin.h"  // sse
#include "pmmintrin.h"  // sse3
#include "tmmintrin.h"  // ssse3
#include "smmintrin.h"  // sse4.1
#include "nmmintrin.h"  // sse4.2
#include "immintrin.h"  // avx, avx2, avx512, FP16C, KNCNI, FMA
//#include "ammintrin.h"  // AMD-specific intrinsics

#include <math.h>       /* sqrt */

namespace hli {

	inline __m128i _mm_hadd_epu8_method1(
		const __m128i * const mem_addr,
		const size_t nBytes)
	{
		const size_t nBlocks = nBytes >> 4; // divide by 16 to get the number of _m128 regs (blocks)
		__m128i sum = _mm_setzero_si128();

		for (size_t block = 0; block < nBlocks; ++block) {
			sum = _mm_add_epi64(sum, _mm_sad_epu8(_mm_load_si128(&mem_addr[block]), _mm_setzero_si128()));
		}

		const __m128i sum_up = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(sum), _mm_castsi128_ps(sum)));
		const __m128i sum_lo = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(sum), _mm_castsi128_ps(sum)));
		return _mm_add_epi64(sum_lo, sum_up);
	}

	inline __m128i _mm_hadd_epu8_method2(
		const __m128i * const mem_addr,
		const size_t nBytes)
	{
		//assume (nBytes < 2 ^ (32 - 8))

		const size_t nBlocks = nBytes >> 4; // divide by 16 to get the number of _m128 regs (blocks)
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
		const __m128i sum_up = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(sum2), _mm_castsi128_ps(sum2)));
		const __m128i sum_lo = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(sum2), _mm_castsi128_ps(sum2)));
		return _mm_add_epi64(sum_lo, sum_up);
	}

	inline __m128i _mm_hadd_epu8(
		const __m128i * const mem_addr,
		const size_t nBytes)
	{
		return _mm_hadd_epu8_method1(mem_addr, nBytes);
	}

	inline __m128i _mm_hadd_epu8_ref(
		const __m128i * const mem_addr,
		const size_t nBytes)
	{
		const unsigned __int8 * const ptr = reinterpret_cast<const unsigned __int8 * const>(mem_addr);
		unsigned __int64 sum = 0;
		for (size_t i = 0; i < nBytes; ++i) {
			sum += ptr[i];
			//std::cout << "INFO: hli::ref::_mm_hadd_epi8: i=" << i << "; sum="<<sum << std::endl;
		}
		return _mm_set1_epi64x(sum);
	}


	/*
	const __m128i * mem_addr = mem_addr;

	__m128i sum = _mm_setzero_si128();
	__m128i sum_p;

	const long long nBlocks = nElements >> 4; // divide by 16 to get the number of _m128 regs (blocks)
	const long long nLoops = nBlocks >> 3; // divide nBlocks by 8 to get the number of time to execute the loop
	const long long tailLength = nBlocks & 0b111;
	std::cout << "INFO: Correlation-Intrinsic:: sum_6bit: nElements=" << nElements << "; nBlocks=" << nBlocks << "; nLoops=" << nLoops << "; tailLength=" << tailLength << std::endl;


	for (long long i = 0; i < nBlocks; ++i) {
		long long offset = (nBlocks - tailLength);
		sum_p = _mm_load_si128(&mem_addr[i + 0]);
		sum_p = _mm_add_epi8(sum_p, _mm_load_si128(&mem_addr[i + 1]));
		sum_p = _mm_add_epi8(sum_p, _mm_load_si128(&mem_addr[i + 2]));
		sum_p = _mm_add_epi8(sum_p, _mm_load_si128(&mem_addr[i + 3]));
		sum_p = _mm_add_epi8(sum_p, _mm_load_si128(&mem_addr[i + 4]));
		sum_p = _mm_add_epi8(sum_p, _mm_load_si128(&mem_addr[i + 5]));
		sum_p = _mm_add_epi8(sum_p, _mm_load_si128(&mem_addr[i + 6]));
		sum_p = _mm_add_epi8(sum_p, _mm_load_si128(&mem_addr[i + 7]));
		sum = _mm_add_epi64(sum, _mm_sad_epu8(sum_p, _mm_setzero_si128()));
	}

	if (tailLength > 0) {
		long long offset = (nBlocks - tailLength);

		sum_p = _mm_load_si128(&mem_addr[offset]);
		for (long long i = 1; i < tailLength; ++i) {
			sum_p = _mm_add_epi8(sum_p, _mm_load_si128(&mem_addr[offset + i]));
		}
		sum = _mm_add_epi64(sum, _mm_sad_epu8(sum_p, _mm_setzero_si128()));
	}

	const __m128i sum_up = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(sum), _mm_castsi128_ps(sum)));
	const __m128i sum_lo = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(sum), _mm_castsi128_ps(sum)));
	return _mm_add_epi64(sum_lo, sum_up);
	*/
}

