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
#include "_mm_permute_array.ipp"
#include "_mm_corr_pd.ipp"

namespace hli {

	namespace priv {

		template <bool HAS_MV, U8 MV>
		inline __m128d _mm_corr_epu8_ref(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const std::tuple<const __m128i * const, const size_t>& data2,
			const size_t nElements)
		{
			//const double std_dev_d1 = sqrt(var_pop_ref(data1, nElements));
			//const double std_dev_d2 = sqrt(var_pop_ref(data1, nElements));
			//return covar_pop_ref(data1, data2, nElements) / (std_dev_d1 * std_dev_d2);

			const __m128d var1 = _mm_variance_epu8_method0<HAS_MV, MV>(data1, nElements);
			const __m128d var2 = _mm_variance_epu8_method0<HAS_MV, MV>(data2, nElements);
			const __m128d covar = _mm_covar_epu8_ref<HAS_MV, MV>(data1, data2, nElements);
			const __m128d corr = _mm_div_pd(covar, _mm_mul_pd(_mm_sqrt_pd(var1), _mm_sqrt_pd(var2)));
			//std::cout << "INFO: _mm_corr_epu8::_mm_corr_epu8_ref: var1=" << var1.m128d_f64[0] << "; var2=" << var2.m128d_f64[0] << "; covar=" << covar.m128d_f64[0] << "; corr=" << corr.m128d_f64[0] << std::endl;
			return corr;
		}

		template <int N_BITS, bool HAS_MV, U8 MV>
		inline __m128d _mm_corr_epu8_method0(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const std::tuple<const __m128i * const, const size_t>& data2,
			const size_t nElements,
			const __m128d average1,
			const __m128d average2)
		{
			double covar = 0;
			double var1 = 0;
			double var2 = 0;
			const double a1 = average1.m128d_f64[0];
			const double a2 = average2.m128d_f64[0];

			const __int8 * const ptr1 = reinterpret_cast<const __int8 * const>(std::get<0>(data1));
			const __int8 * const ptr2 = reinterpret_cast<const __int8 * const>(std::get<0>(data2));

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

		template <int N_BITS, bool HAS_MV, U8 MV>
		inline __m128d _mm_corr_epu8_method0(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const std::tuple<const __m128i * const, const size_t>& data2,
			const size_t nElements)
		{
			const auto tup1 = _mm_hadd_epu8<N_BITS, HAS_MV, MV>(data1, nElements);
			const auto tup2 = _mm_hadd_epu8<N_BITS, HAS_MV, MV>(data2, nElements);
			const __m128d average1 = _mm_div_pd(_mm_cvtepi32_pd(std::get<0>(tup1)), _mm_cvtepi32_pd(std::get<1>(tup1)));
			const __m128d average2 = _mm_div_pd(_mm_cvtepi32_pd(std::get<0>(tup2)), _mm_cvtepi32_pd(std::get<1>(tup2)));
			return _mm_corr_epu8_method0<N_BITS, HAS_MV, MV>(data1, data2, nElements, average1, average2);
		}

		template <int N_BITS, bool HAS_MV, U8 MV>
		inline __m128d _mm_corr_epu8_method1(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const std::tuple<const __m128i * const, const size_t>& data2,
			const size_t nElements,
			const __m128d average1,
			const __m128d average2)
		{
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

		template <int N_BITS, bool HAS_MV, U8 MV>
		inline __m128d _mm_corr_epu8_method1(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const std::tuple<const __m128i * const, const size_t>& data2,
			const size_t nElements)
		{
			const auto tup1 = _mm_hadd_epu8<N_BITS, HAS_MV, MV>(data1, nElements);
			const auto tup2 = _mm_hadd_epu8<N_BITS, HAS_MV, MV>(data2, nElements);
			const __m128d average1 = _mm_div_pd(_mm_cvtepi32_pd(std::get<0>(tup1)), _mm_cvtepi32_pd(std::get<1>(tup1)));
			const __m128d average2 = _mm_div_pd(_mm_cvtepi32_pd(std::get<0>(tup2)), _mm_cvtepi32_pd(std::get<1>(tup2)));
			return _mm_corr_epu8_method1<N_BITS, HAS_MV, MV>(data1, data2, nElements, average1, average2);
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

		template <int N_BITS, bool HAS_MV, U8 MV>
		inline __m128d _mm_corr_epu8_method2(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const std::tuple<const __m128i * const, const size_t>& data2,
			const size_t nElements)
		{
			const auto tup1 = _mm_hadd_epu8<N_BITS, HAS_MV, MV>(data1, nElements);
			const auto tup2 = _mm_hadd_epu8<N_BITS, HAS_MV, MV>(data2, nElements);
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
			const size_t nBytes = std::get<1>(data1);
			const size_t nBlocks = nBytes >> 4;

			if (nElements > 0xFFFF) {
				std::cout << "WARNING: _mm_corr_epu8: _mm_corr_epu8_method3: nElements=" << nElements << " which is larger than 0xFFFF." << std::endl;
			}

			const unsigned __int8 * const ptr1 = reinterpret_cast<const unsigned __int8 * const>(std::get<0>(data1));
			const unsigned __int8 * const ptr2 = reinterpret_cast<const unsigned __int8 * const>(std::get<0>(data2));

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
					if ((d1 != MV) && (d2 != MV)) {

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

		template <int N_BITS, bool HAS_MV, U8 MV>
		inline __m128d calc_variance(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const size_t nElements,
			const std::tuple<__m128d * const, const size_t>& data_double_out)
		{
			const auto tup1 = _mm_hadd_epu8<N_BITS, HAS_MV, MV>(data1, nElements);
			const __m128d nTrueElements = _mm_cvtepi32_pd(std::get<1>(tup1));
			const __m128d average = _mm_div_pd(_mm_cvtepi32_pd(std::get<0>(tup1)), nTrueElements);

			const size_t nBytes = std::get<1>(data1);
			const size_t nBlocksData = nBytes >> 4;

			__m128d * const data_double = std::get<0>(data_double_out);
			__m128d var = _mm_setzero_pd();

			for (size_t block = 0; block < nBlocksData; ++block) {
				const __m128i data = std::get<0>(data1)[block];
				{
					const __m128i d = _mm_cvtepu8_epi32(data);
					const __m128d da = _mm_sub_pd(_mm_cvtepi32_pd(d), average);
					const __m128d db = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d, 0b01011110)), average);
					var = _mm_add_pd(var, _mm_mul_pd(da, da));
					var = _mm_add_pd(var, _mm_mul_pd(db, db));
					data_double[(8 * block) + 0] = da;
					data_double[(8 * block) + 1] = db;
					//std::cout << "INFO: _mm_corr_epu8::calc_variance: block=" << ((8 * block) + 0) << "; d=" << toString_f64(da) << std::endl;
					//std::cout << "INFO: _mm_corr_epu8::calc_variance: block=" << ((8 * block) + 1) << "; d=" << toString_f64(db) << std::endl;
				}
				{
					const __m128i d = _mm_cvtepu8_epi32(_mm_shuffle_epi32(data, 0b01010101));
					const __m128d da = _mm_sub_pd(_mm_cvtepi32_pd(d), average);
					const __m128d db = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d, 0b01011110)), average);
					var = _mm_add_pd(var, _mm_mul_pd(da, da));
					var = _mm_add_pd(var, _mm_mul_pd(db, db));
					data_double[(8 * block) + 2] = da;
					data_double[(8 * block) + 3] = db;
					//std::cout << "INFO: _mm_corr_epu8::calc_variance: block=" << ((8 * block) + 2) << "; d=" << toString_f64(da) << std::endl;
					//std::cout << "INFO: _mm_corr_epu8::calc_variance: block=" << ((8 * block) + 3) << "; d=" << toString_f64(db) << std::endl;
				}
				{
					const __m128i d = _mm_cvtepu8_epi32(_mm_shuffle_epi32(data, 0b10101010));
					const __m128d da = _mm_sub_pd(_mm_cvtepi32_pd(d), average);
					const __m128d db = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d, 0b01011110)), average);
					var = _mm_add_pd(var, _mm_mul_pd(da, da));
					var = _mm_add_pd(var, _mm_mul_pd(db, db));
					data_double[(8 * block) + 4] = da;
					data_double[(8 * block) + 5] = db;
					//std::cout << "INFO: _mm_corr_epu8::calc_variance: block=" << ((8 * block) + 4) << "; d=" << toString_f64(da) << std::endl;
					//std::cout << "INFO: _mm_corr_epu8::calc_variance: block=" << ((8 * block) + 5) << "; d=" << toString_f64(db) << std::endl;
				}
				{
					const __m128i d = _mm_cvtepu8_epi32(_mm_shuffle_epi32(data, 0b11111111));
					const __m128d da = _mm_sub_pd(_mm_cvtepi32_pd(d), average);
					const __m128d db = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d, 0b01011110)), average);
					var = _mm_add_pd(var, _mm_mul_pd(da, da));
					var = _mm_add_pd(var, _mm_mul_pd(db, db));
					data_double[(8 * block) + 6] = da;
					data_double[(8 * block) + 7] = db;
					//std::cout << "INFO: _mm_corr_epu8::calc_variance: block=" << ((8 * block) + 6) << "; d=" << toString_f64(da) << std::endl;
					//std::cout << "INFO: _mm_corr_epu8::calc_variance: block=" << ((8 * block) + 7) << "; d=" << toString_f64(db) << std::endl;
				}
			}
			//for (size_t block = 0; block < (nBlocksData * 8); ++block) {
			//	std::cout << "INFO: _mm_corr_epu8::calc_variance: block=" << block << "; d=" << toString_f64(data_double[block]) << std::endl;
			//}
			return _mm_div_pd(_mm_hadd_pd(var, var), nTrueElements);
		}

		template <int N_BITS, bool HAS_MV, U8 MV>
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

			const __m128d var1 = calc_variance<N_BITS, HAS_MV, MV>(data1, nElements, data1_d);
			const __m128d var2 = calc_variance<N_BITS, HAS_MV, MV>(data2, nElements, data2_d);
			const __m128d var1_2 = _mm_sqrt_pd(_mm_mul_pd(var1, var2));
			const __m128d corr = _mm_corr_dp_method3<HAS_MV, MV>(data1_d, data2_d, nElements, var1_2);

			_mm_free2(data1_d);
			_mm_free2(data2_d);
			return corr;
		}

	}

	namespace test {

		void test_mm_corr_epu8(
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

				__int8 * const ptr1i = reinterpret_cast<__int8 * const>(std::get<0>(data1_r));
				double * const ptr1d = reinterpret_cast<double * const>(std::get<0>(data1_D_r));
				__int8 * const ptr2i = reinterpret_cast<__int8 * const>(std::get<0>(data2_r));
				double * const ptr2d = reinterpret_cast<double * const>(std::get<0>(data2_D_r));
				for (size_t i = 0; i < std::get<1>(data1_r); ++i) {
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
				result_ref = hli::priv::_mm_corr_epu8_ref<HAS_MV, MV>(data1, data2, nElements);
				min_ref = std::min(min_ref, timer::get_elapsed_kcycles());

				{
					timer::reset_and_start_timer();
					result1 = hli::priv::_mm_corr_epu8_method0<8, HAS_MV, MV>(data1, data2, nElements);
					min1 = std::min(min1, timer::get_elapsed_kcycles());

					if (doTests) {
						if (std::abs(result_ref.m128d_f64[0] - result1.m128d_f64[0]) > delta) {
							std::cout << "WARNING: test _mm_corr_epu8_method0<8>: result-ref=" << hli::toString_f64(result_ref) << "; result0=" << hli::toString_f64(result1) << std::endl;
							return;
						}
					}
				}
				{
					timer::reset_and_start_timer();
					result2 = hli::priv::_mm_corr_epu8_method0<6, HAS_MV, MV>(data1, data2, nElements);
					min2 = std::min(min2, timer::get_elapsed_kcycles());

					if (doTests) {
						if (std::abs(result_ref.m128d_f64[0] - result2.m128d_f64[0]) > delta) {
							std::cout << "WARNING: test _mm_corr_epu8_method0<6>: result-ref=" << hli::toString_f64(result_ref) << "; result0=" << hli::toString_f64(result2) << std::endl;
							return;
						}
					}
				}

				{
					timer::reset_and_start_timer();
					result3 = hli::priv::_mm_corr_epu8_method1<8, HAS_MV, MV>(data1, data2, nElements);
					min3 = std::min(min3, timer::get_elapsed_kcycles());

					if (doTests) {
						if (std::abs(result_ref.m128d_f64[0] - result3.m128d_f64[0]) > delta) {
							std::cout << "WARNING: test _mm_corr_epu8_method1<8>: result-ref=" << hli::toString_f64(result_ref) << "; result1=" << hli::toString_f64(result3) << std::endl;
							return;
						}
					}
				}
				{
					timer::reset_and_start_timer();
					result4 = hli::priv::_mm_corr_epu8_method1<6, HAS_MV, MV>(data1, data2, nElements);
					min4 = std::min(min4, timer::get_elapsed_kcycles());

					if (doTests) {
						if (std::abs(result_ref.m128d_f64[0] - result4.m128d_f64[0]) > delta) {
							std::cout << "WARNING: test _mm_corr_epu8_method1<6>: result-ref=" << hli::toString_f64(result_ref) << "; result1=" << hli::toString_f64(result4) << std::endl;
							return;
						}
					}
				}

				{
					timer::reset_and_start_timer();
					result5 = hli::priv::_mm_corr_epu8_method2<8, HAS_MV, MV>(data1, data2, nElements);
					min5 = std::min(min5, timer::get_elapsed_kcycles());

					if (doTests) {
						if (std::abs(result_ref.m128d_f64[0] - result5.m128d_f64[0]) > delta) {
							std::cout << "WARNING: test _mm_corr_epu8_method2<8>: result-ref=" << hli::toString_f64(result_ref) << "; result2=" << hli::toString_f64(result5) << std::endl;
							return;
						}
					}
				}
				{
					timer::reset_and_start_timer();
					result6 = hli::priv::_mm_corr_epu8_method0<6, HAS_MV, MV>(data1, data2, nElements);
					min6 = std::min(min6, timer::get_elapsed_kcycles());

					if (doTests) {
						if (std::abs(result_ref.m128d_f64[0] - result6.m128d_f64[0]) > delta) {
							std::cout << "WARNING: test _mm_corr_epu8_method2<6>: result-ref=" << hli::toString_f64(result_ref) << "; result2=" << hli::toString_f64(result6) << std::endl;
							return;
						}
					}
				}
				{
					timer::reset_and_start_timer();
					result7 = hli::priv::_mm_corr_epu8_method3<HAS_MV, MV>(data1, data2, nElements);
					min7 = std::min(min7, timer::get_elapsed_kcycles());

					if (doTests) {
						if (std::abs(result_ref.m128d_f64[0] - result7.m128d_f64[0]) > delta) {
							std::cout << "WARNING: test _mm_corr_epu8_method3<8>: result-ref=" << hli::toString_f64(result_ref) << "; result3=" << hli::toString_f64(result7) << std::endl;
							return;
						}
					}
				}
				{
					timer::reset_and_start_timer();
					result8 = hli::priv::_mm_corr_epu8_method4<8, HAS_MV, MV>(data1, data2, nElements);
					min8 = std::min(min8, timer::get_elapsed_kcycles());

					if (doTests) {
						if (std::abs(result_ref.m128d_f64[0] - result8.m128d_f64[0]) > delta) {
							std::cout << "WARNING: test _mm_corr_epu8_method4<8>: result-ref=" << hli::toString_f64(result_ref) << "; result4=" << hli::toString_f64(result8) << std::endl;
							return;
						}
					}
				}
				{
					timer::reset_and_start_timer();
					result9 = hli::priv::_mm_corr_pd_method0<HAS_MV, MV>(data1_D, data2_D, nElements);
					min9 = std::min(min9, timer::get_elapsed_kcycles());

					if (doTests) {
						if (std::abs(result_ref.m128d_f64[0] - result9.m128d_f64[0]) > delta) {
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

	template <int N_BITS, bool HAS_MV, U8 MV>
	inline __m128d _mm_corr_epu8(
		const std::tuple<const __m128i * const, const size_t>& data1,
		const std::tuple<const __m128i * const, const size_t>& data2,
		const size_t nElements)
	{
//		return priv::_mm_corr_epu8_method2<N_BITS, HAS_MV, MV>(data1, data2, nElements);
		return priv::_mm_corr_epu8_method3<HAS_MV, MV>(data1, data2, nElements);
	}

}