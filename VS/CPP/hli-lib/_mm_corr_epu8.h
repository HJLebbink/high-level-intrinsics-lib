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

namespace hli {

	// Correlation population SSE: return 2x double var
	template <int N_BITS>
	inline __m128d _mm_corr_epu8(
		const __m128i * const mem_addr1,
		const __m128i * const mem_addr2,
		const size_t nBytes)
	{
		const __m128d std_dev_1 = _mm_sqrt_pd(_mm_variance_epu8<N_BITS>(mem_addr1, nBytes));
		const __m128d std_dev_2 = _mm_sqrt_pd(_mm_variance_epu8<N_BITS>(mem_addr2, nBytes));
		const __m128d corr = _mm_div_pd(_mm_covar_epu8<N_BITS>(mem_addr1, mem_addr2, nBytes), _mm_mul_pd(std_dev_1, std_dev_2));
		return corr;
	}

	namespace priv {

		// Variance population reference
		inline __m128d _mm_corr_epu8_ref(
			const __m128i * const mem_addr1,
			const __m128i * const mem_addr2,
			const size_t nBytes)
		{
			//const double std_dev_d1 = sqrt(var_pop_ref(data1, nElements));
			//const double std_dev_d2 = sqrt(var_pop_ref(data1, nElements));
			//return covar_pop_ref(data1, data2, nElements) / (std_dev_d1 * std_dev_d2);

			const __m128d std_dev_1 = _mm_sqrt_pd(_mm_variance_epu8_ref(mem_addr1, nBytes));
			const __m128d std_dev_2 = _mm_sqrt_pd(_mm_variance_epu8_ref(mem_addr2, nBytes));
			const __m128d corr = _mm_div_pd(_mm_covar_epu8_ref(mem_addr1, mem_addr2, nBytes), _mm_mul_pd(std_dev_1, std_dev_2));
			return corr;
		}
	}
}