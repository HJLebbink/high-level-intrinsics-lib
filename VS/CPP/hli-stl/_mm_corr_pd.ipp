#pragma once

#include <algorithm>	// std::min
#include <limits>		// std::numeric_limits
#include <iostream>		// std::cout
#include <tuple>

//#include "mmintrin.h"  // mmx
#include "emmintrin.h"  // sse
#include "pmmintrin.h"  // sse3
#include "tmmintrin.h"  // ssse3
#include "smmintrin.h"  // sse4.1
#include "nmmintrin.h"  // sse4.2
//#include "immintrin.h"  // avx, avx2, avx512, FP16C, KNCNI, FMA
//#include "ammintrin.h"  // AMD-specific intrinsics

#include "_mm_variance_epu8.ipp"
#include "_mm_covar_epu8.ipp"
#include "_mm_permute_pd_array.ipp"


namespace hli {

	namespace priv {

		template <bool HAS_MV, int MV>
		inline __m128d _mm_corr_pd_method0(
			const std::tuple<const __m128d * const, const size_t>& data1,
			const std::tuple<const __m128d * const, const size_t>& data2,
			const size_t nElements)
		{
			const double * const ptr1 = reinterpret_cast<const double * const>(std::get<0>(data1));
			const double * const ptr2 = reinterpret_cast<const double * const>(std::get<0>(data2));

			double s12 = 0;
			double s11 = 0;
			double s22 = 0;
			double s1 = 0;
			double s2 = 0;

			for (size_t i = 0; i < nElements; ++i)
			{
				const double d1 = ptr1[i];
				const double d2 = ptr2[i];
				s12 += d1 * d2;
				s11 += d1 * d1;
				s22 += d2 * d2;
				s1 += d1;
				s2 += d2;
			}

			double corr = ((nElements * s12) - (s1*s2)) / (sqrt((nElements*s11) - (s1*s1)) * sqrt((nElements * s22) - (s2*s2)));
			return _mm_set1_pd(corr);
		}

		template <bool HAS_MV, int MV>
		inline __m128d _mm_corr_pd_method1(
			const std::tuple<const __m128d * const, const size_t>& data1,
			const std::tuple<const __m128d * const, const size_t>& data2,
			const size_t nElements)
		{
			const size_t nBytes = std::get<1>(data1);
			const size_t nBlocks = nBytes >> 4;

			__m128d s12 = _mm_setzero_pd();
			__m128d s11 = _mm_setzero_pd();
			__m128d s22 = _mm_setzero_pd();
			__m128d s1 = _mm_setzero_pd();
			__m128d s2 = _mm_setzero_pd();

			for (size_t i = 0; i < nBlocks; ++i)
			{
				const __m128d d1 = std::get<0>(data1)[i];
				const __m128d d2 = std::get<0>(data2)[i];
				
				s12 = _mm_add_pd(s12, _mm_mul_pd(d1, d2));
				s11 = _mm_add_pd(s11, _mm_mul_pd(d1, d1));
				s22 = _mm_add_pd(s22, _mm_mul_pd(d2, d2));
				s1 = _mm_add_pd(s1, d1);
				s2 = _mm_add_pd(s2, d2);
			}

			const __m128d nElementsD = _mm_set1_pd(static_cast<double>(nElements));
			const __m128d s11_s22 = _mm_hadd_pd(s11, s22);
			const __m128d s1_s2 = _mm_hadd_pd(s1, s2);
			const __m128d s11_s22_s = _mm_sqrt_pd(_mm_sub_pd(_mm_mul_pd(s11_s22, nElementsD), _mm_mul_pd(s1_s2, s1_s2)));
			const __m128d s11_s22_p = _mm_mul_pd(s11_s22_s, _mm_swap_64(s11_s22_s));
			const __m128d covar = _mm_sub_pd(_mm_mul_pd(_mm_hadd_pd(s12, s12), nElementsD), _mm_mul_pd(s1_s2, _mm_swap_64(s1_s2)));
			const __m128d corr = _mm_div_pd(covar, s11_s22_p);
			//double corr = ((nElements*s12) - (s1*s2)) / (sqrt((nElements*s11) - (s1*s1)) * sqrt((nElements*s22) - (s2*s2)));
			return corr;
		}

		template <bool HAS_MV, int MV>
		inline __m128d _mm_corr_dp_method3(
			const std::tuple<const __m128d * const, const size_t>& data1,
			const std::tuple<const __m128d * const, const size_t>& data2,
			const size_t nElements,
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

			const __m128d nElementsD = _mm_set1_pd(static_cast<double>(nElements));
			covar = _mm_div_pd(_mm_hadd_pd(covar, covar), nElementsD);
			const __m128d corr = _mm_div_pd(covar, var1_2);
			//std::cout << "INFO: _mm_corr_epu8::_mm_corr_dp_method3: covar=" << covar.m128d_f64[0] << "; corr=" << corr.m128d_f64[0] << std::endl;
			return corr;
		}
	}

	namespace test {

		void _mm_corr_pd_speed_test_1(
			const size_t nBlocks, 
			const size_t nExperiments, 
			const bool doTests)
		{
			const double delta = 0.0000001;
			const bool HAS_MV = false;
			const int MV = 99999;
			const size_t nElements = nBlocks * 2;

			auto data1_r = _mm_malloc_m128d(nElements * 8);
			auto data2_r = _mm_malloc_m128d(nElements * 8);

			fillRand_pd(data1_r);
			fillRand_pd(data2_r);

			const std::tuple<const __m128d * const, const size_t> data1 = data1_r;
			const std::tuple<const __m128d * const, const size_t> data2 = data2_r;


			double min0 = std::numeric_limits<double>::max();
			double min1 = std::numeric_limits<double>::max();

			__m128d result0, result1;

			for (size_t i = 0; i < nExperiments; ++i) {

				timer::reset_and_start_timer();
				result0 = hli::priv::_mm_corr_pd_method0<HAS_MV, MV>(data1, data2, nElements);
				min0 = std::min(min0, timer::get_elapsed_kcycles());

				{
					timer::reset_and_start_timer();
					result1 = hli::priv::_mm_corr_pd_method1<HAS_MV, MV>(data1, data2, nElements);
					min1 = std::min(min1, timer::get_elapsed_kcycles());

					if (doTests) {
						if (std::abs(result0.m128d_f64[0] - result1.m128d_f64[0]) > delta) {
							std::cout << "WARNING: test _mm_corr_pd_method0: result0=" << hli::toString_f64(result0) << "; result1=" << hli::toString_f64(result1) << std::endl;
							return;
						}
					}
				}
			}
			printf("[_mm_corr_pd_method0]: %2.5f Kcycles; %0.14f\n", min0, result0.m128d_f64[0]);
			printf("[_mm_corr_pd_method1]: %2.5f Kcycles; %0.14f; %2.3f times faster than ref\n", min1, result1.m128d_f64[0], min0 / min1);

			_mm_free2(data1);
			_mm_free2(data2);
		}
	}

	template <bool HAS_MV, int MV>
	inline __m128d _mm_corr_pd(
		const std::tuple<const __m128d * const, const size_t>& data1,
		const std::tuple<const __m128d * const, const size_t>& data2,
		const size_t nElements)
	{
		return priv::_mm_corr_pd_method0<HAS_MV, MV>(data1, data2, nElements);
		//return priv::_mm_corr_pd_method1<HAS_MV>(data1, data2);
	}
}