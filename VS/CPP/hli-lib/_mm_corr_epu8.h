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

		template <int N_BITS>
		inline __m128d _mm_corr_epu8_method1(
			const __m128i * const mem_addr1,
			const __m128i * const mem_addr2,
			const size_t nBytes)
		{
			const __m128d nElements = _mm_set1_pd(static_cast<double>(nBytes));
			const __m128d average1 = _mm_div_pd(_mm_cvtepi32_pd(_mm_hadd_epu8<N_BITS>(mem_addr1, nBytes)), nElements);
			const __m128d average2 = _mm_div_pd(_mm_cvtepi32_pd(_mm_hadd_epu8<N_BITS>(mem_addr2, nBytes)), nElements);

			__m128d covar = _mm_setzero_pd();
			__m128d var1 = _mm_setzero_pd();
			__m128d var2 = _mm_setzero_pd();

			const size_t nBlocks = nBytes >> 4;
			for (size_t block = 0; block < nBlocks; ++block) {
				const __m128i data1 = mem_addr1[block];
				const __m128i data2 = mem_addr2[block];
				{
					const __m128i d1 = _mm_cvtepu8_epi32(data1);
					const __m128d d1a = _mm_sub_pd(_mm_cvtepi32_pd(d1), average1);
					const __m128d d1b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d1, 0b01011011)), average1);
					var1 = _mm_add_pd(var1, _mm_mul_pd(d1a, d1a));
					var1 = _mm_add_pd(var1, _mm_mul_pd(d1b, d1b));
					const __m128i d2 = _mm_cvtepu8_epi32(data2);
					const __m128d d2a = _mm_sub_pd(_mm_cvtepi32_pd(d2), average2);
					const __m128d d2b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d2, 0b01011011)), average2);
					var2 = _mm_add_pd(var2, _mm_mul_pd(d2a, d2a));
					var2 = _mm_add_pd(var2, _mm_mul_pd(d2b, d2b));
					covar = _mm_add_pd(covar, _mm_mul_pd(d1a, d2a));
					covar = _mm_add_pd(covar, _mm_mul_pd(d1b, d2b));
				}
				{
					const __m128i d1 = _mm_cvtepu8_epi32(_mm_shuffle_epi32(data1, 0b01010101));
					const __m128d d1a = _mm_sub_pd(_mm_cvtepi32_pd(d1), average1);
					const __m128d d1b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d1, 0b01011011)), average1);
					var1 = _mm_add_pd(var1, _mm_mul_pd(d1a, d1a));
					var1 = _mm_add_pd(var1, _mm_mul_pd(d1b, d1b));
					const __m128i d2 = _mm_cvtepu8_epi32(_mm_shuffle_epi32(data2, 0b01010101));
					const __m128d d2a = _mm_sub_pd(_mm_cvtepi32_pd(d2), average2);
					const __m128d d2b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d2, 0b01011011)), average2);
					var2 = _mm_add_pd(var2, _mm_mul_pd(d2a, d2a));
					var2 = _mm_add_pd(var2, _mm_mul_pd(d2b, d2b));
					covar = _mm_add_pd(covar, _mm_mul_pd(d1a, d2a));
					covar = _mm_add_pd(covar, _mm_mul_pd(d1b, d2b));
				}
				{
					const __m128i d1 = _mm_cvtepu8_epi32(_mm_shuffle_epi32(data1, 0b10101010));
					const __m128d d1a = _mm_sub_pd(_mm_cvtepi32_pd(d1), average1);
					const __m128d d1b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d1, 0b01011011)), average1);
					var1 = _mm_add_pd(var1, _mm_mul_pd(d1a, d1a));
					var1 = _mm_add_pd(var1, _mm_mul_pd(d1b, d1b));
					const __m128i d2 = _mm_cvtepu8_epi32(_mm_shuffle_epi32(data2, 0b10101010));
					const __m128d d2a = _mm_sub_pd(_mm_cvtepi32_pd(d2), average2);
					const __m128d d2b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d2, 0b01011011)), average2);
					var2 = _mm_add_pd(var2, _mm_mul_pd(d2a, d2a));
					var2 = _mm_add_pd(var2, _mm_mul_pd(d2b, d2b));
					covar = _mm_add_pd(covar, _mm_mul_pd(d1a, d2a));
					covar = _mm_add_pd(covar, _mm_mul_pd(d1b, d2b));
				}
				{
					const __m128i d1 = _mm_cvtepu8_epi32(_mm_shuffle_epi32(data1, 0b11111111));
					const __m128d d1a = _mm_sub_pd(_mm_cvtepi32_pd(d1), average1);
					const __m128d d1b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d1, 0b01011011)), average1);
					var1 = _mm_add_pd(var1, _mm_mul_pd(d1a, d1a));
					var1 = _mm_add_pd(var1, _mm_mul_pd(d1b, d1b));
					const __m128i d2 = _mm_cvtepu8_epi32(_mm_shuffle_epi32(data2, 0b11111111));
					const __m128d d2a = _mm_sub_pd(_mm_cvtepi32_pd(d2), average2);
					const __m128d d2b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d2, 0b01011011)), average2);
					var2 = _mm_add_pd(var2, _mm_mul_pd(d2a, d2a));
					var2 = _mm_add_pd(var2, _mm_mul_pd(d2b, d2b));
					covar = _mm_add_pd(covar, _mm_mul_pd(d1a, d2a));
					covar = _mm_add_pd(covar, _mm_mul_pd(d1b, d2b));
				}
				//std::cout << "INFO: hli:::_mm_variance_epu8: data=" << toString_u8(data) << "; d1=" << toString_u32(d1) << "; d2=" << toString_u32(d2) << "; d3=" << toString_u32(d3) << "; d4=" << toString_u32(d4) << std::endl;
			}
			covar = _mm_div_pd(_mm_hadd_pd(covar, covar), nElements);
			var1 = _mm_div_pd(_mm_hadd_pd(var1, var1), nElements);
			var2 = _mm_div_pd(_mm_hadd_pd(var2, var2), nElements);

			const __m128d corr = _mm_div_pd(covar, _mm_sqrt_pd(_mm_mul_pd(var1, var2)));
			return corr;
		}

		// Correlation population SSE: return 2x double var
		inline __m128d _mm_corr_epu8_method2(
			const __m128i * const mem_addr1,
			const __m128i * const mem_addr2,
			const size_t nBytes,
			const __m128d average1,
			const __m128d average2)
		{
			const __m128d var1 = _mm_variance_epu8(mem_addr1, nBytes, average1);
			const __m128d var2 = _mm_variance_epu8(mem_addr2, nBytes, average2);
			const __m128d corr = _mm_div_pd(_mm_covar_epu8(mem_addr1, mem_addr2, nBytes, average1, average2), _mm_sqrt_pd(_mm_mul_pd(var1, var2)));
			return corr;
		}

		template <int N_BITS>
		inline __m128d _mm_corr_epu8_method2(
			const __m128i * const mem_addr1,
			const __m128i * const mem_addr2,
			const size_t nBytes)
		{
			const __m128d nElements = _mm_set1_pd(static_cast<double>(nBytes));
			const __m128d average1 = _mm_div_pd(_mm_cvtepi32_pd(_mm_hadd_epu8<N_BITS>(mem_addr1, nBytes)), nElements);
			const __m128d average2 = _mm_div_pd(_mm_cvtepi32_pd(_mm_hadd_epu8<N_BITS>(mem_addr2, nBytes)), nElements);

			return _mm_corr_epu8_method2(mem_addr1, mem_addr2, nBytes, average1, average2);
		}

	}

	template <int N_BITS>
	inline __m128d _mm_corr_epu8(
		const __m128i * const mem_addr1,
		const __m128i * const mem_addr2,
		const size_t nBytes)
	{
		return priv::_mm_corr_epu8_method1<N_BITS>(mem_addr1, mem_addr2, nBytes);
	}
}