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

	inline __m128i _mm_hadd_epi8(
		const __m128i * const mem_addr,
		const long long nElements)
	{
		return _mm_setzero_si128();
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

}
