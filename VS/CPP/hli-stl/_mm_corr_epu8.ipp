#pragma once

#include <algorithm>	// std::min
#include <limits>		// std::numeric_limits
#include <iostream>		// std::cout
#include <tuple>
#include <math.h>

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
#include "_mm_permute_epu8_array.ipp"
#include "_mm_corr_pd.ipp"

namespace hli
{
	namespace priv
	{
		template <int N_BITS1, int N_BITS2, bool HAS_MV, U8 MV>
		inline __m128d _mm_corr_epu8_ref(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const std::tuple<const __m128i * const, const size_t>& data2,
			const size_t nElements)
		{
			if constexpr (HAS_MV)
			{
				auto data1b = deepCopy(data1);
				auto data2b = deepCopy(data2);
				auto data1b_ptr = reinterpret_cast<U8 * const>(std::get<0>(data1b));
				auto data2b_ptr = reinterpret_cast<U8 * const>(std::get<0>(data2b));

				// copy the missing value to both data columns
				for (int i = 0; i < nElements; ++i)
				{
					if ((data1b_ptr[i] == MV) || (data2b_ptr[i] == MV))
					{
						data1b_ptr[i] = MV;
						data2b_ptr[i] = MV;
					}
				}

				const __m128d var1 = _mm_variance_epu8_method0<HAS_MV, MV>(data1b, nElements);
				const __m128d var2 = _mm_variance_epu8_method0<HAS_MV, MV>(data2b, nElements);
				const __m128d covar = _mm_covar_epu8_method0<N_BITS1, N_BITS2, HAS_MV, MV>(data1b, data2b, nElements);
				const __m128d corr = _mm_div_pd(covar, _mm_mul_pd(_mm_sqrt_pd(var1), _mm_sqrt_pd(var2)));
				//std::cout << "INFO: _mm_corr_epu8::_mm_corr_epu8_ref: var1=" << var1.m128d_f64[0] << "; var2=" << var2.m128d_f64[0] << "; covar=" << covar.m128d_f64[0] << "; corr=" << corr.m128d_f64[0] << std::endl;

				_mm_free2(data1b);
				_mm_free2(data2b);

				return corr;
			}
			else
			{
				const __m128d var1 = _mm_variance_epu8_method0<HAS_MV, MV>(data1, nElements);
				const __m128d var2 = _mm_variance_epu8_method0<HAS_MV, MV>(data2, nElements);
				const __m128d covar = _mm_covar_epu8_method0<N_BITS1, N_BITS2, HAS_MV, MV>(data1, data2, nElements);
				const __m128d corr = _mm_div_pd(covar, _mm_mul_pd(_mm_sqrt_pd(var1), _mm_sqrt_pd(var2)));
				//std::cout << "INFO: _mm_corr_epu8::_mm_corr_epu8_ref: var1=" << var1.m128d_f64[0] << "; var2=" << var2.m128d_f64[0] << "; covar=" << covar.m128d_f64[0] << "; corr=" << corr.m128d_f64[0] << std::endl;
				return corr;
			}
		}

		template <bool HAS_MV, U8 MV>
		inline __m128d _mm_corr_epu8_method0(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const std::tuple<const __m128i * const, const size_t>& data2,
			const size_t nElements,
			const __m128d average1,
			const __m128d average2)
		{
			if constexpr (HAS_MV)
			{ //TODO
				std::cout << "WARNING: _mm_corr_epu8_method0: Not implemented yet" << std::endl;
				return _mm_setzero_pd();
			}

			double covar = 0;
			double var1 = 0;
			double var2 = 0;
			const double a1 = average1.m128d_f64[0];
			const double a2 = average2.m128d_f64[0];

			auto ptr1 = reinterpret_cast<const U8 * const>(std::get<0>(data1));
			auto ptr2 = reinterpret_cast<const U8 * const>(std::get<0>(data2));

			const size_t nBytes = std::get<1>(data1);

			for (size_t element = 0; element < nElements; ++element)
			{
				const double d1 = static_cast<double>(ptr1[element]) - a1;
				const double d2 = static_cast<double>(ptr2[element]) - a2;

				var1 += d1 * d1;
				var2 += d2 * d2;
				covar += d1 * d2;
			}

			var1 /= nElements;
			var2 /= nElements;
			covar /= nElements;

			const double corr = covar / sqrt(var1 * var2);
			return _mm_set1_pd(corr);
		}

		template <int N_BITS1, int N_BITS2, bool HAS_MV, U8 MV>
		inline __m128d _mm_corr_epu8_method0(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const std::tuple<const __m128i * const, const size_t>& data2,
			const size_t nElements)
		{
			const auto tup1 = _mm_hadd_epu8<N_BITS1, HAS_MV, MV>(data1, nElements);
			const auto tup2 = _mm_hadd_epu8<N_BITS2, HAS_MV, MV>(data2, nElements);
			const __m128d average1 = _mm_div_pd(_mm_cvtepi32_pd(std::get<0>(tup1)), _mm_cvtepi32_pd(std::get<1>(tup1)));
			const __m128d average2 = _mm_div_pd(_mm_cvtepi32_pd(std::get<0>(tup2)), _mm_cvtepi32_pd(std::get<1>(tup2)));
			return _mm_corr_epu8_method0<HAS_MV, MV>(data1, data2, nElements, average1, average2);
		}

		template <bool HAS_MV, U8 MV>
		inline __m128d _mm_corr_epu8_method1(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const std::tuple<const __m128i * const, const size_t>& data2,
			const size_t nElements,
			const __m128d average1,
			const __m128d average2)
		{
			if constexpr (HAS_MV)
			{ //TODO
				std::cout << "WARNING: _mm_corr_epu8_method1: Not implemented yet" << std::endl;
				return _mm_setzero_pd();
			}

			__m128d covar = _mm_setzero_pd();
			__m128d var1 = _mm_setzero_pd();
			__m128d var2 = _mm_setzero_pd();

			const size_t nBytes = std::get<1>(data1);
			const size_t nBlocks = nBytes >> 4;
			for (size_t block = 0; block < nBlocks; ++block)
			{
				const auto d1 = _mm_sub_pd(_mm_cvt_epu8_pd(std::get<0>(data1)[block]), average1);

				var1 = _mm_add_pd(var1, _mm_mul_pd(std::get<0>(d1), std::get<0>(d1)));
				var1 = _mm_add_pd(var1, _mm_mul_pd(std::get<1>(d1), std::get<1>(d1)));
				var1 = _mm_add_pd(var1, _mm_mul_pd(std::get<2>(d1), std::get<2>(d1)));
				var1 = _mm_add_pd(var1, _mm_mul_pd(std::get<3>(d1), std::get<3>(d1)));
				var1 = _mm_add_pd(var1, _mm_mul_pd(std::get<4>(d1), std::get<4>(d1)));
				var1 = _mm_add_pd(var1, _mm_mul_pd(std::get<5>(d1), std::get<5>(d1)));
				var1 = _mm_add_pd(var1, _mm_mul_pd(std::get<6>(d1), std::get<6>(d1)));
				var1 = _mm_add_pd(var1, _mm_mul_pd(std::get<7>(d1), std::get<7>(d1)));

				const auto d2 = _mm_sub_pd(_mm_cvt_epu8_pd(std::get<0>(data2)[block]), average2);

				var2 = _mm_add_pd(var2, _mm_mul_pd(std::get<0>(d2), std::get<0>(d2)));
				var2 = _mm_add_pd(var2, _mm_mul_pd(std::get<1>(d2), std::get<1>(d2)));
				var2 = _mm_add_pd(var2, _mm_mul_pd(std::get<2>(d2), std::get<2>(d2)));
				var2 = _mm_add_pd(var2, _mm_mul_pd(std::get<3>(d2), std::get<3>(d2)));
				var2 = _mm_add_pd(var2, _mm_mul_pd(std::get<4>(d2), std::get<4>(d2)));
				var2 = _mm_add_pd(var2, _mm_mul_pd(std::get<5>(d2), std::get<5>(d2)));
				var2 = _mm_add_pd(var2, _mm_mul_pd(std::get<6>(d2), std::get<6>(d2)));
				var2 = _mm_add_pd(var2, _mm_mul_pd(std::get<7>(d2), std::get<7>(d2)));

				covar = _mm_add_pd(covar, _mm_mul_pd(std::get<0>(d1), std::get<0>(d2)));
				covar = _mm_add_pd(covar, _mm_mul_pd(std::get<1>(d1), std::get<1>(d2)));
				covar = _mm_add_pd(covar, _mm_mul_pd(std::get<2>(d1), std::get<2>(d2)));
				covar = _mm_add_pd(covar, _mm_mul_pd(std::get<3>(d1), std::get<3>(d2)));
				covar = _mm_add_pd(covar, _mm_mul_pd(std::get<4>(d1), std::get<4>(d2)));
				covar = _mm_add_pd(covar, _mm_mul_pd(std::get<5>(d1), std::get<5>(d2)));
				covar = _mm_add_pd(covar, _mm_mul_pd(std::get<6>(d1), std::get<6>(d2)));
				covar = _mm_add_pd(covar, _mm_mul_pd(std::get<7>(d1), std::get<7>(d2)));
			}
			const __m128d nElementsD = _mm_set1_pd(static_cast<double>(nElements));
			covar = _mm_div_pd(_mm_hadd_pd(covar, covar), nElementsD);
			var1 = _mm_div_pd(_mm_hadd_pd(var1, var1), nElementsD);
			var2 = _mm_div_pd(_mm_hadd_pd(var2, var2), nElementsD);
			const __m128d corr = _mm_div_pd(covar, _mm_sqrt_pd(_mm_mul_pd(var1, var2)));
			//std::cout << "INFO: _mm_corr_epu8::_mm_corr_epu8_method1: var1=" << var1.m128d_f64[0] << "; var2=" << var2.m128d_f64[0] << "; covar=" << covar.m128d_f64[0] << "; corr=" << corr.m128d_f64[0] << std::endl;
			return corr;
		}

		template <int N_BITS1, int N_BITS2, bool HAS_MV, U8 MV>
		inline __m128d _mm_corr_epu8_method1(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const std::tuple<const __m128i * const, const size_t>& data2,
			const size_t nElements)
		{
			const auto tup1 = _mm_hadd_epu8<N_BITS1, HAS_MV, MV>(data1, nElements);
			const auto tup2 = _mm_hadd_epu8<N_BITS2, HAS_MV, MV>(data2, nElements);
			const __m128d average1 = _mm_div_pd(_mm_cvtepi32_pd(std::get<0>(tup1)), _mm_cvtepi32_pd(std::get<1>(tup1)));
			const __m128d average2 = _mm_div_pd(_mm_cvtepi32_pd(std::get<0>(tup2)), _mm_cvtepi32_pd(std::get<1>(tup2)));
			return _mm_corr_epu8_method1<HAS_MV, MV>(data1, data2, nElements, average1, average2);
		}

		template <bool HAS_MV, U8 MV>
		inline __m128d _mm_corr_epu8_method2(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const std::tuple<const __m128i * const, const size_t>& data2,
			const size_t nElements,
			const __m128d average1,
			const __m128d average2)
		{
			const __m128d var1 = _mm_variance_epu8<HAS_MV, MV>(data1, average1);
			const __m128d var2 = _mm_variance_epu8<HAS_MV, MV>(data2, average2);
			const __m128d covar = _mm_covar_epu8<HAS_MV, MV>(data1, data2, nElements, average1, average2);
			const __m128d corr = _mm_div_pd(covar, _mm_sqrt_pd(_mm_mul_pd(var1, var2)));
			//std::cout << "INFO: _mm_corr_epu8::_mm_corr_epu8_method2: var1=" << var1.m128d_f64[0] << "; var2=" << var2.m128d_f64[0] << "; covar=" << covar.m128d_f64[0] << "; corr=" << corr.m128d_f64[0] << std::endl;
			return corr;
		}

		template <int N_BITS1, int N_BITS2, bool HAS_MV, U8 MV>
		inline __m128d _mm_corr_epu8_method2(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const std::tuple<const __m128i * const, const size_t>& data2,
			const size_t nElements)
		{
			const auto tup1 = _mm_hadd_epu8<N_BITS1, HAS_MV, MV>(data1, nElements);
			const auto tup2 = _mm_hadd_epu8<N_BITS2, HAS_MV, MV>(data2, nElements);
			const __m128d average1 = _mm_div_pd(_mm_cvtepi32_pd(std::get<0>(tup1)), _mm_cvtepi32_pd(std::get<1>(tup1)));
			const __m128d average2 = _mm_div_pd(_mm_cvtepi32_pd(std::get<0>(tup2)), _mm_cvtepi32_pd(std::get<1>(tup2)));
			return _mm_corr_epu8_method2<HAS_MV, MV>(data1, data2, nElements, average1, average2);
		}

		template <bool HAS_MV, U8 MV>
		inline __m128d _mm_corr_epu8_method3(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const std::tuple<const __m128i * const, const size_t>& data2,
			const size_t nElements)
		{
			if constexpr (HAS_MV)
			{ //TODO
				std::cout << "WARNING: _mm_corr_epu8_method3: Not implemented yet" << std::endl;
				return _mm_setzero_pd();
			}

			const size_t nBytes = std::get<1>(data1);
			const size_t nBlocks = nBytes >> 4;

			if (nElements > 0xFFFF)
			{
				std::cout << "WARNING: _mm_corr_epu8: _mm_corr_epu8_method3: nElements=" << nElements << " which is larger than 0xFFFF." << std::endl;
			}

			auto ptr1 = reinterpret_cast<const U8 * const>(std::get<0>(data1));
			auto ptr2 = reinterpret_cast<const U8 * const>(std::get<0>(data2));

			__int32 s12 = 0;
			__int32 s11 = 0;
			__int32 s22 = 0;
			__int32 s1 = 0;
			__int32 s2 = 0;

			if (HAS_MV)
			{
				size_t nElements_No_MV = 0;

				for (size_t i = 0; i < nElements; ++i)
				{
					const U8 d1 = ptr1[i];
					const U8 d2 = ptr2[i];
					if ((d1 != MV) && (d2 != MV))
					{

						s12 += d1 * d2;
						s11 += d1 * d1;
						s22 += d2 * d2;
						s1 += d1;
						s2 += d2;

						nElements_No_MV++;
					}
				}

				const double s12d = static_cast<double>(s12);
				const double s11d = static_cast<double>(s11);
				const double s22d = static_cast<double>(s22);
				const double s1d = static_cast<double>(s1);
				const double s2d = static_cast<double>(s2);

				double corr = ((nElements_No_MV * s12d) - (s1d*s2d)) / (sqrt((nElements_No_MV*s11d) - (s1d*s1d)) * sqrt((nElements_No_MV * s22d) - (s2d*s2d)));
				return _mm_set1_pd(corr);
			}
			else
			{
				for (size_t i = 0; i < nElements; ++i)
				{
					const U8 d1 = ptr1[i];
					const U8 d2 = ptr2[i];

					s12 += d1 * d2;
					s11 += d1 * d1;
					s22 += d2 * d2;
					s1 += d1;
					s2 += d2;
				}
			}

			const double s12d = static_cast<double>(s12);
			const double s11d = static_cast<double>(s11);
			const double s22d = static_cast<double>(s22);
			const double s1d = static_cast<double>(s1);
			const double s2d = static_cast<double>(s2);

			double corr = ((nElements * s12d) - (s1d*s2d)) / (sqrt((nElements*s11d) - (s1d*s1d)) * sqrt((nElements * s22d) - (s2d*s2d)));
			return _mm_set1_pd(corr);
		}

		template <int N_BITS1, int N_BITS2, bool HAS_MV, U8 MV>
		inline __m128d _mm_corr_epu8_method4(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const std::tuple<const __m128i * const, const size_t>& data2,
			const size_t nElements)
		{
			const size_t nBytes = std::get<1>(data1);
			const size_t nBlocksInput = nBytes >> 4;
			const size_t nBlocksDouble = nBlocksInput * 8;

			auto data1_d = _mm_malloc_m128d(nBlocksDouble * 16);
			auto data2_d = _mm_malloc_m128d(nBlocksDouble * 16);

			const __m128d var1 = priv::_mm_variance_epu8_method1<N_BITS1, HAS_MV, MV>(data1, nElements, data1_d);
			const __m128d var2 = priv::_mm_variance_epu8_method1<N_BITS2, HAS_MV, MV>(data2, nElements, data2_d);
			const __m128d var1_2 = _mm_sqrt_pd(_mm_mul_pd(var1, var2));
			const __m128d corr = _mm_corr_dp_method3<HAS_MV, MV>(data1_d, data2_d, nElements, var1_2);

			_mm_free2(data1_d);
			_mm_free2(data2_d);
			return corr;
		}
	}

	namespace test
	{

		void _mm_corr_epu8_speed_test_1(
			const size_t nBlocks,
			const size_t nExperiments,
			const bool doTests)
		{
			const double delta = 0.0000001;
			const bool HAS_MV = false;
			const U8 MV = 0xFF;

			const size_t nElements = nBlocks * 16;
			auto data1_r = _mm_malloc_m128i(nElements);
			auto data2_r = _mm_malloc_m128i(nElements);
			auto data1_D_r = _mm_malloc_m128d(nElements * 8);
			auto data2_D_r = _mm_malloc_m128d(nElements * 8);

			{	// initialize data
				fillRand_epu8<5>(data1_r);
				fillRand_epu8<5>(data2_r);

				auto ptr1i = reinterpret_cast<U8 * const>(std::get<0>(data1_r));
				auto ptr1d = reinterpret_cast<double * const>(std::get<0>(data1_D_r));
				auto ptr2i = reinterpret_cast<U8 * const>(std::get<0>(data2_r));
				auto ptr2d = reinterpret_cast<double * const>(std::get<0>(data2_D_r));

				for (size_t i = 0; i < std::get<1>(data1_r); ++i)
				{
					ptr1d[i] = static_cast<double>(ptr1i[i]);
					ptr2d[i] = static_cast<double>(ptr2i[i]);
				}
			}

			const std::tuple<const __m128i * const, const size_t> data1 = data1_r;
			const std::tuple<const __m128i * const, const size_t> data2 = data2_r;
			const std::tuple<const __m128d * const, const size_t> data1_D = data1_D_r;
			const std::tuple<const __m128d * const, const size_t> data2_D = data2_D_r;

			double min_ref = std::numeric_limits<double>::max();
			double min1 = std::numeric_limits<double>::max();
			double min2 = std::numeric_limits<double>::max();
			double min3 = std::numeric_limits<double>::max();
			double min4 = std::numeric_limits<double>::max();
			double min5 = std::numeric_limits<double>::max();
			double min6 = std::numeric_limits<double>::max();
			double min7 = std::numeric_limits<double>::max();
			double min8 = std::numeric_limits<double>::max();
			double min9 = std::numeric_limits<double>::max();

			__m128d result_ref, result1, result2, result3, result4, result5, result6, result7, result8, result9;

			for (size_t i = 0; i < nExperiments; ++i)
			{
				timer::reset_and_start_timer();
				result_ref = hli::priv::_mm_corr_epu8_ref<8, 8, HAS_MV, MV>(data1, data2, nElements);
				min_ref = std::min(min_ref, timer::get_elapsed_kcycles());

				{
					timer::reset_and_start_timer();
					result1 = hli::priv::_mm_corr_epu8_method0<8, 8, HAS_MV, MV>(data1, data2, nElements);
					min1 = std::min(min1, timer::get_elapsed_kcycles());

					if (doTests)
					{
						if (std::abs(result_ref.m128d_f64[0] - result1.m128d_f64[0]) > delta)
						{
							std::cout << "WARNING: test _mm_corr_epu8_method0<8>: result-ref=" << hli::toString_f64(result_ref) << "; result0=" << hli::toString_f64(result1) << std::endl;
							return;
						}
					}
				}
				{
					timer::reset_and_start_timer();
					result2 = hli::priv::_mm_corr_epu8_method0<6, 6, HAS_MV, MV>(data1, data2, nElements);
					min2 = std::min(min2, timer::get_elapsed_kcycles());

					if (doTests)
					{
						if (std::abs(result_ref.m128d_f64[0] - result2.m128d_f64[0]) > delta)
						{
							std::cout << "WARNING: test _mm_corr_epu8_method0<6>: result-ref=" << hli::toString_f64(result_ref) << "; result0=" << hli::toString_f64(result2) << std::endl;
							return;
						}
					}
				}

				{
					timer::reset_and_start_timer();
					result3 = hli::priv::_mm_corr_epu8_method1<8, 8, HAS_MV, MV>(data1, data2, nElements);
					min3 = std::min(min3, timer::get_elapsed_kcycles());

					if (doTests)
					{
						if (std::abs(result_ref.m128d_f64[0] - result3.m128d_f64[0]) > delta)
						{
							std::cout << "WARNING: test _mm_corr_epu8_method1<8>: result-ref=" << hli::toString_f64(result_ref) << "; result1=" << hli::toString_f64(result3) << std::endl;
							return;
						}
					}
				}
				{
					timer::reset_and_start_timer();
					result4 = hli::priv::_mm_corr_epu8_method1<6, 6, HAS_MV, MV>(data1, data2, nElements);
					min4 = std::min(min4, timer::get_elapsed_kcycles());

					if (doTests)
					{
						if (std::abs(result_ref.m128d_f64[0] - result4.m128d_f64[0]) > delta)
						{
							std::cout << "WARNING: test _mm_corr_epu8_method1<6>: result-ref=" << hli::toString_f64(result_ref) << "; result1=" << hli::toString_f64(result4) << std::endl;
							return;
						}
					}
				}

				{
					timer::reset_and_start_timer();
					result5 = hli::priv::_mm_corr_epu8_method2<8, 8, HAS_MV, MV>(data1, data2, nElements);
					min5 = std::min(min5, timer::get_elapsed_kcycles());

					if (doTests)
					{
						if (std::abs(result_ref.m128d_f64[0] - result5.m128d_f64[0]) > delta)
						{
							std::cout << "WARNING: test _mm_corr_epu8_method2<8>: result-ref=" << hli::toString_f64(result_ref) << "; result2=" << hli::toString_f64(result5) << std::endl;
							return;
						}
					}
				}
				{
					timer::reset_and_start_timer();
					result6 = hli::priv::_mm_corr_epu8_method0<6, 6, HAS_MV, MV>(data1, data2, nElements);
					min6 = std::min(min6, timer::get_elapsed_kcycles());

					if (doTests)
					{
						if (std::abs(result_ref.m128d_f64[0] - result6.m128d_f64[0]) > delta)
						{
							std::cout << "WARNING: test _mm_corr_epu8_method2<6>: result-ref=" << hli::toString_f64(result_ref) << "; result2=" << hli::toString_f64(result6) << std::endl;
							return;
						}
					}
				}
				{
					timer::reset_and_start_timer();
					result7 = hli::priv::_mm_corr_epu8_method3<HAS_MV, MV>(data1, data2, nElements);
					min7 = std::min(min7, timer::get_elapsed_kcycles());

					if (doTests)
					{
						if (std::abs(result_ref.m128d_f64[0] - result7.m128d_f64[0]) > delta)
						{
							std::cout << "WARNING: test _mm_corr_epu8_method3<8>: result-ref=" << hli::toString_f64(result_ref) << "; result3=" << hli::toString_f64(result7) << std::endl;
							return;
						}
					}
				}
				{
					timer::reset_and_start_timer();
					result8 = hli::priv::_mm_corr_epu8_method4<8, 8, HAS_MV, MV>(data1, data2, nElements);
					min8 = std::min(min8, timer::get_elapsed_kcycles());

					if (doTests)
					{
						if (std::abs(result_ref.m128d_f64[0] - result8.m128d_f64[0]) > delta)
						{
							std::cout << "WARNING: test _mm_corr_epu8_method4<8>: result-ref=" << hli::toString_f64(result_ref) << "; result4=" << hli::toString_f64(result8) << std::endl;
							return;
						}
					}
				}
				{
					timer::reset_and_start_timer();
					result9 = hli::priv::_mm_corr_pd_method0<HAS_MV, MV>(data1_D, data2_D, nElements);
					min9 = std::min(min9, timer::get_elapsed_kcycles());

					if (doTests)
					{
						if (std::abs(result_ref.m128d_f64[0] - result9.m128d_f64[0]) > delta)
						{
							std::cout << "WARNING: test _mm_corr_pd_method0<8>: result-ref=" << hli::toString_f64(result_ref) << "; result4=" << hli::toString_f64(result9) << std::endl;
							return;
						}
					}
				}
			}
			printf("[_mm_corr_epu8 Ref]       : %9.5f Kcycles; %0.14f\n", min_ref, result_ref.m128d_f64[0]);
			printf("[_mm_corr_epu8_method0<8>]: %9.5f Kcycles; %0.14f; %2.3f times faster than ref\n", min1, result1.m128d_f64[0], min_ref / min1);
			printf("[_mm_corr_epu8_method0<6>]: %9.5f Kcycles; %0.14f; %2.3f times faster than ref\n", min2, result2.m128d_f64[0], min_ref / min2);
			printf("[_mm_corr_epu8_method1<8>]: %9.5f Kcycles; %0.14f; %2.3f times faster than ref\n", min3, result3.m128d_f64[0], min_ref / min3);
			printf("[_mm_corr_epu8_method1<6>]: %9.5f Kcycles; %0.14f; %2.3f times faster than ref\n", min4, result4.m128d_f64[0], min_ref / min4);
			printf("[_mm_corr_epu8_method2<8>]: %9.5f Kcycles; %0.14f; %2.3f times faster than ref\n", min5, result5.m128d_f64[0], min_ref / min5);
			printf("[_mm_corr_epu8_method2<6>]: %9.5f Kcycles; %0.14f; %2.3f times faster than ref\n", min6, result6.m128d_f64[0], min_ref / min6);
			printf("[_mm_corr_epu8_method3]   : %9.5f Kcycles; %0.14f; %2.3f times faster than ref\n", min7, result7.m128d_f64[0], min_ref / min7);
			printf("[_mm_corr_epu8_method4<8>]: %9.5f Kcycles; %0.14f; %2.3f times faster than ref\n", min8, result8.m128d_f64[0], min_ref / min8);
			printf("[_mm_corr_pd_method0]     : %9.5f Kcycles; %0.14f; %2.3f times faster than ref\n", min9, result9.m128d_f64[0], min_ref / min9);

			_mm_free2(data1);
			_mm_free2(data2);
		}
	}

	template <int N_BITS1, int N_BITS2, bool HAS_MV, U8 MV>
	inline __m128d _mm_corr_epu8(
		const std::tuple<const __m128i * const, const size_t>& data1,
		const std::tuple<const __m128i * const, const size_t>& data2,
		const size_t nElements)
	{
		if constexpr (HAS_MV)
		{
			return priv::_mm_corr_epu8_ref<N_BITS1, N_BITS2, HAS_MV, MV>(data1, data2, nElements);
		}
		else
		{
			return priv::_mm_corr_epu8_method3<HAS_MV, MV>(data1, data2, nElements);
		}
	}

	template <bool HAS_MV, U8 MV>
	inline __m128d _mm_corr_epu8(
		const std::tuple<const __m128i * const, const size_t>& data1,
		const int nBits1,
		const std::tuple<const __m128i * const, const size_t>& data2,
		const int nBits2,
		const size_t nElements)
	{
		#if _DEBUG
		if ((nBits1 > 8) || (nBits1 < 1)) std::cout << "WARNING: _mm_corr_epu8: nBits1=" << nBits1 << " has to be in range[1..8]" << std::endl;
		if ((nBits2 > 8) || (nBits2 < 1)) std::cout << "WARNING: _mm_corr_epu8: nBits2=" << nBits2 << " has to be in range[1..8]" << std::endl;
		#endif

		switch (nBits1)
		{
			case 1:
				switch (nBits2)
				{
					case 1: return _mm_corr_epu8<1, 1, HAS_MV, MV>(data1, data2, nElements);
					case 2: return _mm_corr_epu8<1, 2, HAS_MV, MV>(data1, data2, nElements);
					case 3: return _mm_corr_epu8<1, 3, HAS_MV, MV>(data1, data2, nElements);
					case 4: return _mm_corr_epu8<1, 4, HAS_MV, MV>(data1, data2, nElements);
					case 5: return _mm_corr_epu8<1, 5, HAS_MV, MV>(data1, data2, nElements);
					case 6: return _mm_corr_epu8<1, 6, HAS_MV, MV>(data1, data2, nElements);
					case 7: return _mm_corr_epu8<1, 7, HAS_MV, MV>(data1, data2, nElements);
					default: return _mm_setzero_pd();
				}
			case 2:
				switch (nBits2)
				{
					case 1: return _mm_corr_epu8<2, 1, HAS_MV, MV>(data1, data2, nElements);
					case 2: return _mm_corr_epu8<2, 2, HAS_MV, MV>(data1, data2, nElements);
					case 3: return _mm_corr_epu8<2, 3, HAS_MV, MV>(data1, data2, nElements);
					case 4: return _mm_corr_epu8<2, 4, HAS_MV, MV>(data1, data2, nElements);
					case 5: return _mm_corr_epu8<2, 5, HAS_MV, MV>(data1, data2, nElements);
					case 6: return _mm_corr_epu8<2, 6, HAS_MV, MV>(data1, data2, nElements);
					default: return _mm_setzero_pd();
				}
			case 3:
				switch (nBits2)
				{
					case 1: return _mm_corr_epu8<3, 1, HAS_MV, MV>(data1, data2, nElements);
					case 2: return _mm_corr_epu8<3, 2, HAS_MV, MV>(data1, data2, nElements);
					case 3: return _mm_corr_epu8<3, 3, HAS_MV, MV>(data1, data2, nElements);
					case 4: return _mm_corr_epu8<3, 4, HAS_MV, MV>(data1, data2, nElements);
					case 5: return _mm_corr_epu8<3, 5, HAS_MV, MV>(data1, data2, nElements);
					default: return _mm_setzero_pd();
				}
			case 4:
				switch (nBits2)
				{
					case 1: return _mm_corr_epu8<4, 1, HAS_MV, MV>(data1, data2, nElements);
					case 2: return _mm_corr_epu8<4, 2, HAS_MV, MV>(data1, data2, nElements);
					case 3: return _mm_corr_epu8<4, 3, HAS_MV, MV>(data1, data2, nElements);
					case 4: return _mm_corr_epu8<4, 4, HAS_MV, MV>(data1, data2, nElements);
					default: return _mm_setzero_pd();
				}
			case 5:
				switch (nBits2)
				{
					case 1: return _mm_corr_epu8<5, 1, HAS_MV, MV>(data1, data2, nElements);
					case 2: return _mm_corr_epu8<5, 2, HAS_MV, MV>(data1, data2, nElements);
					case 3: return _mm_corr_epu8<5, 3, HAS_MV, MV>(data1, data2, nElements);
					default: return _mm_setzero_pd();
				}
			case 6:
				switch (nBits2)
				{
					case 1: return _mm_corr_epu8<6, 1, HAS_MV, MV>(data1, data2, nElements);
					case 2: return _mm_corr_epu8<6, 2, HAS_MV, MV>(data1, data2, nElements);
					default: return _mm_setzero_pd();
				}
			case 7:
				switch (nBits2)
				{
					case 1: return _mm_corr_epu8<7, 1, HAS_MV, MV>(data1, data2, nElements);
					default: return _mm_setzero_pd();
				}
			default: return _mm_setzero_pd();
		}
	}
}