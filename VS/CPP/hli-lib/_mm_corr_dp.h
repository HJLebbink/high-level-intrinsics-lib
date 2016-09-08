#pragma once

#include <tuple>

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
#include "_mm_permute_array.h"

namespace hli {

	namespace priv {

		inline __m128d _mm_corr_dp_method3(
			const std::tuple<__m128d * const, const size_t> data1,
			const std::tuple<__m128d * const, const size_t> data2,
			const __m128d var1_2)
		{
			const size_t nBytes = std::get<1>(data1);
			const size_t nBlocks = nBytes >> 4;
			//std::cout << "INFO: _mm_corr_epu8::_mm_corr_dp_method3: tail=" << tail << std::endl;

			__m128d covar = _mm_setzero_pd();

			if (false) { // unrolling is NOT faster
				const size_t nLoops = nBlocks >> 2;
				const size_t tail = nBlocks & 0b11;
				for (size_t block = 0; block < nBlocks; block += 4) {
					const __m128d d1a = std::get<0>(data1)[block + 0];
					const __m128d d1b = std::get<0>(data1)[block + 1];
					const __m128d d1c = std::get<0>(data1)[block + 2];
					const __m128d d1d = std::get<0>(data1)[block + 3];
					const __m128d d2a = std::get<0>(data2)[block + 0];
					const __m128d d2b = std::get<0>(data2)[block + 1];
					const __m128d d2c = std::get<0>(data2)[block + 2];
					const __m128d d2d = std::get<0>(data2)[block + 3];
					covar = _mm_add_pd(covar, _mm_mul_pd(d1a, d2a));
					covar = _mm_add_pd(covar, _mm_mul_pd(d1b, d2b));
					covar = _mm_add_pd(covar, _mm_mul_pd(d1c, d2c));
					covar = _mm_add_pd(covar, _mm_mul_pd(d1d, d2d));
				}

				for (size_t block = nBlocks - tail; block < nBlocks; ++block) {
					const __m128d d1a = std::get<0>(data1)[block + 0];
					const __m128d d2a = std::get<0>(data2)[block + 0];
					covar = _mm_add_pd(covar, _mm_mul_pd(d1a, d2a));
					std::cout << "INFO: _mm_corr_epu8::_mm_corr_dp_method3: tail block=" << block << std::endl;
				}
			} else {
				for (size_t block = 0; block < nBlocks; ++block) {
					const __m128d d1a = std::get<0>(data1)[block + 0];
					const __m128d d2a = std::get<0>(data2)[block + 0];
					covar = _mm_add_pd(covar, _mm_mul_pd(d1a, d2a));
				}
			}

			const __m128d nElements = _mm_set1_pd(static_cast<double>(nBytes >> 3));
			covar = _mm_div_pd(_mm_hadd_pd(covar, covar), nElements);
			const __m128d corr = _mm_div_pd(covar, var1_2);
			//std::cout << "INFO: _mm_corr_epu8::_mm_corr_dp_method3: covar=" << covar.m128d_f64[0] << "; corr=" << corr.m128d_f64[0] << std::endl;
			return corr;
		}

	}
}