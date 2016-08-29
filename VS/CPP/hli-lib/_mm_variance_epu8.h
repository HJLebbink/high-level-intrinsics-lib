#pragma once

//#include "mmintrin.h"  // mmx
#include "emmintrin.h"  // sse
#include "pmmintrin.h"  // sse3
#include "tmmintrin.h"  // ssse3
#include "smmintrin.h"  // sse4.1
#include "nmmintrin.h"  // sse4.2
//#include "immintrin.h"  // avx, avx2, avx512, FP16C, KNCNI, FMA
//#include "ammintrin.h"  // AMD-specific intrinsics

#include "_mm_hadd_epu8.h"

namespace hli {

	// Variance population SSE: return 2x double var
	template <int N_BITS>
	inline __m128d _mm_variance_epu8(
		const __m128i * const mem_addr,
		const size_t nBytes)
	{
		const __m128d nElements = _mm_set1_pd(static_cast<double>(nBytes));
		const __m128d sum = _mm_cvtepi32_pd(_mm_hadd_epu8<N_BITS>(mem_addr, nBytes));
		const __m128d average = _mm_div_pd(sum, nElements);

		__m128d result_a = _mm_setzero_pd();
		__m128d result_b = _mm_setzero_pd();

		const size_t nBlocks = nBytes >> 4;
		for (size_t block = 0; block < nBlocks; ++block) {
			const __m128i data1 = _mm_load_si128(&mem_addr[block]);
			{
				const __m128i d1 = _mm_cvtepi8_epi32(data1);
				const __m128d d1a = _mm_sub_pd(_mm_cvtepi32_pd(d1), average);
				const __m128d d1b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d1, 0b01011011)), average);
				result_a = _mm_add_pd(result_a, _mm_mul_pd(d1a, d1a));
				result_b = _mm_add_pd(result_b, _mm_mul_pd(d1b, d1b));
			}
			{
				const __m128i d1 = _mm_cvtepi8_epi32(_mm_shuffle_epi32(data1, 0b01010101));
				const __m128d d1a = _mm_sub_pd(_mm_cvtepi32_pd(d1), average);
				const __m128d d1b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d1, 0b01011011)), average);
				result_a = _mm_add_pd(result_a, _mm_mul_pd(d1a, d1a));
				result_b = _mm_add_pd(result_b, _mm_mul_pd(d1b, d1b));
			}
			{
				const __m128i d1 = _mm_cvtepi8_epi32(_mm_shuffle_epi32(data1, 0b10101010));
				const __m128d d1a = _mm_sub_pd(_mm_cvtepi32_pd(d1), average);
				const __m128d d1b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d1, 0b01011011)), average);
				result_a = _mm_add_pd(result_a, _mm_mul_pd(d1a, d1a));
				result_b = _mm_add_pd(result_b, _mm_mul_pd(d1b, d1b));
			}
			{
				const __m128i d1 = _mm_cvtepi8_epi32(_mm_shuffle_epi32(data1, 0b11111111));
				const __m128d d1a = _mm_sub_pd(_mm_cvtepi32_pd(d1), average);
				const __m128d d1b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d1, 0b01011011)), average);
				result_a = _mm_add_pd(result_a, _mm_mul_pd(d1a, d1a));
				result_b = _mm_add_pd(result_b, _mm_mul_pd(d1b, d1b));
			}
			//std::cout << "INFO: hli:::_mm_variance_epu8: data=" << toString_u8(data) << "; d1=" << toString_u32(d1) << "; d2=" << toString_u32(d2) << "; d3=" << toString_u32(d3) << "; d4=" << toString_u32(d4) << std::endl;
		}
		result_a = _mm_add_pd(result_a, result_b);
		return _mm_div_pd(_mm_hadd_pd(result_a, result_a), nElements);
	}

	namespace priv {

		// Variance population reference
		inline __m128d _mm_variance_epu8_ref(
			const __m128i * const mem_addr,
			const size_t nBytes)
		{
			const double average = static_cast<double>(_mm_hadd_epu8_ref(mem_addr, nBytes).m128i_u32[0]) / nBytes;
			const unsigned __int8 * const ptr = reinterpret_cast<const unsigned __int8 * const>(mem_addr);
			double sum = 0;
			for (size_t i = 0; i < nBytes; ++i) {
				double d = static_cast<double>(ptr[i]) - average;
				sum += (d * d);
			}
			return _mm_set1_pd(sum / nBytes);
		}
	}
}