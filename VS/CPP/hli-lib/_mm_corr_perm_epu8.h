#pragma once

//#include "mmintrin.h"  // mmx
#include "emmintrin.h"  // sse
#include "pmmintrin.h"  // sse3
#include "tmmintrin.h"  // ssse3
#include "smmintrin.h"  // sse4.1
#include "nmmintrin.h"  // sse4.2
//#include "immintrin.h"  // avx, avx2, avx512, FP16C, KNCNI, FMA
//#include "ammintrin.h"  // AMD-specific intrinsics

#include "_mm_variance_epu8.h"
#include "_mm_covar_epu8.h"
#include "_mm_permute_epu8.h"

namespace hli {

	namespace priv {

		// Variance population reference
		inline __m128d _mm_corr_perm_epu8_ref(
			const __m128i * const mem_addr1,
			const __m128i * const mem_addr2,
			const size_t nBytes,
			__m128i& randInts)
		{
			//const double std_dev_d1 = sqrt(var_pop_ref(data1, nElements));
			//const double std_dev_d2 = sqrt(var_pop_ref(data1, nElements));
			//return covar_pop_ref(data1, data2, nElements) / (std_dev_d1 * std_dev_d2);
			
			__m128i * const mem_addr3 = static_cast<__m128i * const>(_mm_malloc(nBytes, 16));
			memcpy(mem_addr3, mem_addr2, nBytes);
			_mm_permute_epu8_ref(mem_addr3, nBytes, randInts);

			const __m128d corr = _mm_corr_epu8_ref(mem_addr1, mem_addr3, nBytes);

			_mm_free(mem_addr3);
			return corr;
		}

		template <int N_BITS>
		inline __m128d _mm_corr_perm_epu8_method1(
			const __m128i * const mem_addr1,
			const __m128i * const mem_addr2,
			const size_t nBytes,
			__m128i& randInts)
		{
			__m128i * const mem_addr3 = static_cast<__m128i * const>(_mm_malloc(nBytes, 16));
			memcpy(mem_addr3, mem_addr2, nBytes);
			_mm_permute_epu8_ref(mem_addr3, nBytes, randInts);

			const __m128d corr = _mm_corr_epu8<N_BITS>(mem_addr1, mem_addr3, nBytes);

			_mm_free(mem_addr3);
			return corr;
		}
	}

	template <int N_BITS>
	inline __m128d _mm_corr_perm_epu8(
		const __m128i * const mem_addr1,
		const __m128i * const mem_addr2,
		const size_t nBytes,
		__m128i& randInts)
	{
		//return priv::_mm_corr_perm_epu8_ref(mem_addr1, mem_addr2, nBytes, randInts);
		return priv::_mm_corr_perm_epu8_method1(mem_addr1, mem_addr2, nBytes, randInts);
	}

	template <int N_BITS>
	inline void _mm_corr_perm_epu8(
		const __m128i * const mem_addr1,
		const __m128i * const mem_addr2,
		const size_t nBytes,
		const __m128d * const results,
		const size_t nPermutations,
		__m128i& randInts)
	{
		//return priv::_mm_corr_perm_epu8_ref(mem_addr1, mem_addr2, nBytes, randInts);
		return priv::_mm_corr_perm_epu8(mem_addr1, mem_addr2, nBytes, randInts);
	}

}