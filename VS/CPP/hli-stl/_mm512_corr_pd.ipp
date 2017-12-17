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

#include "timing.ipp"


#include "_mm_variance_epu8.ipp"
#include "_mm_covar_epu8.ipp"
#include "_mm_permute_pd_array.ipp"

namespace hli
{
	//array of doubles 512
	using d512 = std::tuple<const __m512d * const, const int>;

	namespace priv
	{
		template <bool HAS_MV, int MV>
		inline double _mm512_corr_pd_method0(
			const d512& data1,
			const d512& data2,
			const int nElements)
		{
			const double * const ptr1 = reinterpret_cast<const double * const>(std::get<0>(data1));
			const double * const ptr2 = reinterpret_cast<const double * const>(std::get<0>(data2));

			double s12 = 0;
			double s11 = 0;
			double s22 = 0;
			double s1 = 0;
			double s2 = 0;

			for (int i = 0; i < nElements; ++i)
			{
				const double d1 = ptr1[i];
				const double d2 = ptr2[i];
				s12 += d1 * d2;
				s11 += d1 * d1;
				s22 += d2 * d2;
				s1 += d1;
				s2 += d2;
			}
			const double corr = ((nElements * s12) - (s1*s2)) / (sqrt((nElements*s11) - (s1*s1)) * sqrt((nElements * s22) - (s2*s2)));
			return corr;
		}

		template <bool HAS_MV, int MV>
		inline double _mm512_corr_pd_method1(
			const d512& data1,
			const d512& data2,
			const int nElements)
		{
			const int nBytes = std::get<1>(data1);
			const int nBlocks = nElements >> 3; // there are 8 doubles in a zmm register
			const int tail = nElements & 0b111;

			__m512d zeros = _mm512_setzero_pd();
			__m512d s12 = zeros;
			__m512d s11 = zeros;
			__m512d s22 = zeros;
			__m512d s1 = zeros;
			__m512d s2 = zeros;

			for (int i = 0; i < nBlocks; ++i)
			{
				const __m512d d1 = std::get<0>(data1)[i];
				const __m512d d2 = std::get<0>(data2)[i];

				s12 = _mm512_fmadd_pd(d1, d2, s12);
				s11 = _mm512_fmadd_pd(d1, d1, s11);
				s22 = _mm512_fmadd_pd(d2, d2, s22);

				s1 = _mm512_add_pd(s1, d1);
				s2 = _mm512_add_pd(s2, d2);
			}

			if (tail > 0)	//tail 
			{
				const __mmask8 mask = (1 << tail) - 1;

				const __m512d d1 = std::get<0>(data1)[nBlocks];
				const __m512d d2 = std::get<0>(data2)[nBlocks];

				s12 = _mm512_mask_fmadd_pd(d1, mask, d2, s12);
				s11 = _mm512_mask_fmadd_pd(d1, mask, d1, s11);
				s22 = _mm512_mask_fmadd_pd(d2, mask, d2, s22);

				s1 = _mm512_mask_add_pd(zeros, mask, s1, d1);
				s2 = _mm512_mask_add_pd(zeros, mask, s2, d2);
			}

			const double s12d = _mm512_reduce_add_pd(s12);
			const double s11d = _mm512_reduce_add_pd(s11);
			const double s22d = _mm512_reduce_add_pd(s22);
			const double s1d = _mm512_reduce_add_pd(s1);
			const double s2d = _mm512_reduce_add_pd(s2);

			if (true)
			{
				const double corr = ((nElements * s12d) - (s1d*s2d)) / (sqrt((nElements*s11d) - (s1d*s1d)) * sqrt((nElements * s22d) - (s2d*s2d)));
				return corr;
			}
			else
			{
				const __m128d nElementsD = _mm_set1_pd(static_cast<double>(nElements));
				const __m128d s11_s22 = _mm_set_pd(s11d, s22d);
				const __m128d s1_s2 = _mm_set_pd(s1d, s2d);
				const __m128d s2_s1 = _mm_swap_64(s1_s2);
				const __m128d s12_s12 = _mm_set_pd(s12d, s12d);

				const __m128d s11_s22_s = _mm_sqrt_pd(_mm_sub_pd(_mm_mul_pd(s11_s22, nElementsD), _mm_mul_pd(s1_s2, s1_s2)));
				const __m128d s22_s11_s = _mm_swap_64(s11_s22_s);
				const __m128d s11_s22_p = _mm_mul_pd(s11_s22_s, s22_s11_s);

				const __m128d covar = _mm_sub_sd(_mm_mul_pd(s12_s12, nElementsD), _mm_mul_pd(s1_s2, s2_s1));
				const __m128d corr = _mm_div_sd(covar, s11_s22_p);
				//double corr = ((nElements*s12) - (s1*s2)) / (sqrt((nElements*s11) - (s1*s1)) * sqrt((nElements*s22) - (s2*s2)));
				return _mm_cvtsd_f64(corr);
			}
		}

		template <bool HAS_MV, int MV>
		inline __m512d _mm512_corr_dp_method3(
			const d512& data1,
			const d512& data2,
			const int nElements,
			const __m512d var1_2)
		{
			const int nBytes = std::get<1>(data1);
			const int nBlocks = nBytes >> 4;
			//std::cout << "INFO: _mm_corr_epu8::_mm_corr_dp_method3: tail=" << tail << std::endl;

			__m128d covar = _mm_setzero_pd();

			if (false)
			{ // unrolling is NOT faster
				const int nLoops = nBlocks >> 2;
				const int tail = nBlocks & 0b11;
				for (int block = 0; block < nBlocks; block += 4)
				{
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

				for (int block = nBlocks - tail; block < nBlocks; ++block)
				{
					const __m128d d1a = std::get<0>(data1)[block + 0];
					const __m128d d2a = std::get<0>(data2)[block + 0];
					covar = _mm_add_pd(covar, _mm_mul_pd(d1a, d2a));
					std::cout << "INFO: _mm_corr_epu8::_mm_corr_dp_method3: tail block=" << block << std::endl;
				}
			}
			else
			{
				for (int block = 0; block < nBlocks; ++block)
				{
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

	namespace test
	{
		using namespace tools::timing;

		void _mm512_corr_pd_speed_test_1(
			const int nElements,
			const int nExperiments,
			const bool doTests)
		{
			const double delta = 0.0000001;
			const bool HAS_MV = false;
			const int MV = 99999;
			const int nBlocks = nElements >> 3;
			const int nBytes = nElements * sizeof(double);

			auto data1_r = _mm_malloc_m512d(nBytes);
			auto data2_r = _mm_malloc_m512d(nBytes);

			fillRand_pd(data1_r);
			fillRand_pd(data2_r);

			const d512 data1 = data1_r; // create new variable just to make it const
			const d512 data2 = data2_r;
			const auto data1_x = std::make_tuple(reinterpret_cast<const __m128d * const>(std::get<0>(data1)), std::get<1>(data1));
			const auto data2_x = std::make_tuple(reinterpret_cast<const __m128d * const>(std::get<0>(data2)), std::get<1>(data2));

			auto min0 = std::numeric_limits<double>::max();
			auto min1 = std::numeric_limits<double>::max();
			auto min2 = std::numeric_limits<double>::max();
			auto min3 = std::numeric_limits<double>::max();

			double result0, result1, result2, result3;

			for (int i = 0; i < nExperiments; ++i)
			{
				reset_and_start_timer();
				result0 = hli::priv::_mm512_corr_pd_method0<HAS_MV, MV>(data1, data2, nElements);
				min0 = std::min(min0, get_elapsed_kcycles());

				{
					reset_and_start_timer();
					result1 = hli::priv::_mm512_corr_pd_method1<HAS_MV, MV>(data1, data2, nElements);
					min1 = std::min(min1, get_elapsed_kcycles());

					if (doTests)
					{
						if (std::abs(result0 - result1) > delta)
						{
							std::cout << "WARNING: test _mm512_corr_pd_method0: result0=" << result0 << "; result1=" << result1 << std::endl;
							return;
						}
					}
				}
				{
					reset_and_start_timer();
					result2 = hli::priv::_mm_corr_pd_method0<HAS_MV, MV>(data1_x, data2_x, nElements);
					min2 = std::min(min2, get_elapsed_kcycles());

					if (doTests)
					{
						if (std::abs(result0 - result2) > delta)
						{
							std::cout << "WARNING: test _mm512_corr_pd_method0: result0=" << result0 << "; result2=" << result2 << std::endl;
							return;
						}
					}
				}
				{
					reset_and_start_timer();
					result3 = hli::priv::_mm_corr_pd_method1<HAS_MV, MV>(data1_x, data2_x, nElements);
					min3 = std::min(min3, get_elapsed_kcycles());

					if (doTests)
					{
						if (std::abs(result0 - result3) > delta)
						{
							std::cout << "WARNING: test _mm512_corr_pd_method0: result0=" << result0 << "; result3=" << result3 << std::endl;
							return;
						}
					}
				}
			}
			printf("[_mm512_corr_pd_method0]: %2.5f Kcycles; %0.14f\n", min0, result0);
			printf("[_mm512_corr_pd_method1]: %2.5f Kcycles; %0.14f; %2.3f times faster than ref\n", min1, result1, min0 / min1);
			printf("[_mm_corr_pd_method0]:    %2.5f Kcycles; %0.14f; %2.3f times faster than ref\n", min2, result2, min0 / min2);
			printf("[_mm_corr_pd_method1]:    %2.5f Kcycles; %0.14f; %2.3f times faster than ref\n", min3, result3, min0 / min3);

			_mm_free2(data1);
			_mm_free2(data2);
		}
	}

	template <bool HAS_MV, int MV>
	inline __m512d _mm512_corr_pd(
		const d512& data1,
		const d512& data2,
		const int nElements)
	{
		return priv::_mm512_corr_pd_method0<HAS_MV, MV>(data1, data2, nElements);
	}
}