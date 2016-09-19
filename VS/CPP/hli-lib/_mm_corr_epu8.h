#pragma once

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

#include "_mm_variance_epu8.h"
#include "_mm_covar_epu8.h"
#include "_mm_permute_array.h"
#include "_mm_corr_pd.h"

namespace hli {

	namespace priv {

		inline __m128d _mm_corr_epu8_ref(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const std::tuple<const __m128i * const, const size_t>& data2,
			const size_t nElements)
		{
			//const double std_dev_d1 = sqrt(var_pop_ref(data1, nElements));
			//const double std_dev_d2 = sqrt(var_pop_ref(data1, nElements));
			//return covar_pop_ref(data1, data2, nElements) / (std_dev_d1 * std_dev_d2);

			const __m128d var1 = _mm_variance_epu8_method0(data1, nElements);
			const __m128d var2 = _mm_variance_epu8_method0(data2, nElements);
			const __m128d covar = _mm_covar_epu8_ref(data1, data2, nElements);
			const __m128d corr = _mm_div_pd(covar, _mm_mul_pd(_mm_sqrt_pd(var1), _mm_sqrt_pd(var2)));
			//std::cout << "INFO: _mm_corr_epu8::_mm_corr_epu8_ref: var1=" << var1.m128d_f64[0] << "; var2=" << var2.m128d_f64[0] << "; covar=" << covar.m128d_f64[0] << "; corr=" << corr.m128d_f64[0] << std::endl;
			return corr;
		}

		template <int N_BITS>
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

			for (int element = 0; element < nElements; ++element)
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

		template <int N_BITS>
		inline __m128d _mm_corr_epu8_method0(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const std::tuple<const __m128i * const, const size_t>& data2,
			const size_t nElements)
		{
			const __m128d nElementsD = _mm_set1_pd(static_cast<double>(nElements));
			const __m128d average1 = _mm_div_pd(_mm_cvtepi32_pd(_mm_hadd_epu8<N_BITS>(data1, nElements)), nElementsD);
			const __m128d average2 = _mm_div_pd(_mm_cvtepi32_pd(_mm_hadd_epu8<N_BITS>(data2, nElements)), nElementsD);

			return _mm_corr_epu8_method0<N_BITS>(data1, data2, nElements, average1, average2);
		}

		template <int N_BITS>
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

		template <int N_BITS>
		inline __m128d _mm_corr_epu8_method1(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const std::tuple<const __m128i * const, const size_t>& data2,
			const size_t nElements)
		{
			const __m128d nElementsD = _mm_set1_pd(static_cast<double>(nElements));
			const __m128d average1 = _mm_div_pd(_mm_cvtepi32_pd(_mm_hadd_epu8<N_BITS>(data1, nElements)), nElementsD);
			const __m128d average2 = _mm_div_pd(_mm_cvtepi32_pd(_mm_hadd_epu8<N_BITS>(data2, nElements)), nElementsD);

			return _mm_corr_epu8_method1<N_BITS>(data1, data2, nElements, average1, average2);
		}

		inline __m128d _mm_corr_epu8_method2(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const std::tuple<const __m128i * const, const size_t>& data2,
			const size_t nElements,
			const __m128d average1,
			const __m128d average2)
		{
			const __m128d var1 = _mm_variance_epu8(data1, average1);
			const __m128d var2 = _mm_variance_epu8(data2, average2);
			const __m128d covar = _mm_covar_epu8(data1, data2, nElements, average1, average2);
			const __m128d corr = _mm_div_pd(covar, _mm_sqrt_pd(_mm_mul_pd(var1, var2)));
			//std::cout << "INFO: _mm_corr_epu8::_mm_corr_epu8_method2: var1=" << var1.m128d_f64[0] << "; var2=" << var2.m128d_f64[0] << "; covar=" << covar.m128d_f64[0] << "; corr=" << corr.m128d_f64[0] << std::endl;
			return corr;
		}

		template <int N_BITS>
		inline __m128d _mm_corr_epu8_method2(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const std::tuple<const __m128i * const, const size_t>& data2,
			const size_t nElements)
		{
			const __m128d nElementsD = _mm_set1_pd(static_cast<double>(nElements));
			const __m128d average1 = _mm_div_pd(_mm_cvtepi32_pd(_mm_hadd_epu8<N_BITS>(data1, nElements)), nElementsD);
			const __m128d average2 = _mm_div_pd(_mm_cvtepi32_pd(_mm_hadd_epu8<N_BITS>(data2, nElements)), nElementsD);
			return _mm_corr_epu8_method2(data1, data2, nElements, average1, average2);
		}

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

			const __int8 * const ptr1 = reinterpret_cast<const __int8 * const>(std::get<0>(data1));
			const __int8 * const ptr2 = reinterpret_cast<const __int8 * const>(std::get<0>(data2));

			__int32 s12 = 0;
			__int32 s11 = 0;
			__int32 s22 = 0;
			__int32 s1 = 0;
			__int32 s2 = 0;

			for (size_t i = 0; i < nElements; ++i)
			{
				const unsigned __int8 d1 = ptr1[i];
				const unsigned __int8 d2 = ptr2[i];

				s12 += d1 * d2;
				s11 += d1 * d1;
				s22 += d2 * d2;
				s1 += d1;
				s2 += d2;
			}

			const double s12d = static_cast<double>(s12);
			const double s11d = static_cast<double>(s11);
			const double s22d = static_cast<double>(s22);
			const double s1d = static_cast<double>(s1);
			const double s2d = static_cast<double>(s2);

			double corr = ((nElements * s12d) - (s1d*s2d)) / (sqrt((nElements*s11d) - (s1d*s1d)) * sqrt((nElements * s22d) - (s2d*s2d)));
			return _mm_set1_pd(corr);
		}

		template <int N_BITS>
		inline __m128d calc_variance(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const size_t nElements,
			const std::tuple<__m128d * const, const size_t>& data_double_out)
		{
			const size_t nBytes = std::get<1>(data1);
			const __m128d nElementsD = _mm_set1_pd(static_cast<double>(nElements));
			const __m128d average = _mm_div_pd(_mm_cvtepi32_pd(_mm_hadd_epu8<N_BITS>(data1, nElements)), nElementsD);
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
			return _mm_div_pd(_mm_hadd_pd(var, var), nElementsD);
		}

		template <int N_BITS>
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

			const __m128d var1 = calc_variance<N_BITS>(data1, nElements, data1_d);
			const __m128d var2 = calc_variance<N_BITS>(data2, nElements, data2_d);
			const __m128d var1_2 = _mm_sqrt_pd(_mm_mul_pd(var1, var2));
			const __m128d corr = _mm_corr_dp_method3(data1_d, data2_d, nElements, var1_2);

			_mm_free2(data1_d);
			_mm_free2(data2_d);
			return corr;
		}

		namespace perm {

			inline void _mm_corr_perm_epu8_method0(
				const std::tuple<const __m128i * const, const size_t>& data1,
				const std::tuple<const __m128i * const, const size_t>& data2,
				const size_t nElements,
				const std::tuple<__m128d * const, const size_t>& results,
				const size_t nPermutations,
				__m128i& randInts)
			{
				auto data3 = deepCopy(data2);
				auto swap = _mm_malloc_m128i(nElements << 1);

				double * const results_double = reinterpret_cast<double * const>(std::get<0>(results));
				for (size_t permutation = 0; permutation < nPermutations; ++permutation)
				{
					_mm_permute_epu8_array_ref(data3, swap, randInts);
					const __m128d corr1 = _mm_corr_epu8_ref(data1, data3, nElements);
					results_double[permutation] = corr1.m128d_f64[0];
				}
				_mm_free2(data3);
				_mm_free2(swap);
			}

			template <int N_BITS>
			inline void _mm_corr_perm_epu8_method1(
				const std::tuple<const __m128i * const, const size_t>& data1,
				const std::tuple<const __m128i * const, const size_t>& data2,
				const size_t nElements,
				const std::tuple<__m128d * const, const size_t>& results,
				const size_t nPermutations,
				__m128i& randInts)
			{
				auto data3 = deepCopy(data2);
				const size_t nBytes = std::get<1>(data1);
				const size_t swap_array_nBytes = nBytes << 1;
				auto swap = _mm_malloc_m128i(swap_array_nBytes);

				const __m128d nElementsD = _mm_set1_pd(static_cast<double>(nElements));
				//std::cout << "INFO: _mm_corr_epu8::_mm_corr_perm_epu8_method1: nElements=" << toString_f64(nElements) << std::endl;
				const __m128d average1 = _mm_div_pd(_mm_cvtepi32_pd(_mm_hadd_epu8<N_BITS>(data1, nElements)), nElementsD);
				const __m128d average2 = _mm_div_pd(_mm_cvtepi32_pd(_mm_hadd_epu8<N_BITS>(data2, nElements)), nElementsD);

				double * const results_double = reinterpret_cast<double * const>(std::get<0>(results));
				for (size_t permutation = 0; permutation < nPermutations; ++permutation) {
					_mm_permute_epu8_array(data3, swap, randInts);
					const __m128d corr = _mm_corr_epu8_method1<N_BITS>(data1, data3, nElements, average1, average2);
					results_double[permutation] = corr.m128d_f64[0];
				}
				_mm_free2(data3);
				_mm_free2(swap);
			}

			template <int N_BITS>
			inline void _mm_corr_perm_epu8_method2(
				const std::tuple<const __m128i * const, const size_t>& data1,
				const std::tuple<const __m128i * const, const size_t>& data2,
				const size_t nElements,
				const std::tuple<__m128d * const, const size_t>& results,
				const size_t nPermutations,
				__m128i& randInts)
			{
				const size_t nBytes = std::get<1>(data1);
				auto data1_Double = _mm_malloc_m128d(8 * nBytes);
				auto data2_Double = _mm_malloc_m128d(8 * nBytes);
				auto swap = _mm_malloc_m128i(2 * nBytes);

				const __m128d var1 = calc_variance<N_BITS>(data1, nElements, data1_Double);
				const __m128d var2 = calc_variance<N_BITS>(data2, nElements, data2_Double);
				const __m128d var1_2 = _mm_sqrt_pd(_mm_mul_pd(var1, var2));

				//std::cout << "INFO: _mm_corr_epu8::_mm_corr_perm_epu8_method3: var1=" << var1.m128d_f64[0] << "; var2=" << var2.m128d_f64[0] << std::endl;

				double * const results_double = reinterpret_cast<double * const>(std::get<0>(results));
				for (size_t permutation = 0; permutation < nPermutations; ++permutation) {
					_mm_permute_dp_array(data2_Double, swap, randInts);
					const __m128d corr = _mm_corr_dp_method3(data1_Double, data2_Double, nElements, var1_2);
					//std::cout << "INFO: _mm_corr_epu8::_mm_corr_perm_epu8_method3: corr=" << corr.m128d_f64[0] << std::endl;
					results_double[permutation] = corr.m128d_f64[0];
				}
				_mm_free2(data1_Double);
				_mm_free2(data2_Double);
				_mm_free2(swap);
			}

			inline void _mm_corr_perm_epu8_method3(
				const std::tuple<const __m128i * const, const size_t>& data1,
				const std::tuple<const __m128i * const, const size_t>& data2,
				const size_t nElements,
				const std::tuple<__m128d * const, const size_t>& results,
				const size_t nPermutations,
				__m128i& randInts)
			{
				const size_t nBytes = std::get<1>(data1);
				const size_t nBlocks = nBytes >> 4;

				if (nElements > 0xFFFF) {
					std::cout << "WARNING: _mm_corr_epu8: _mm_corr_epu8_method3: nElements=" << nElements << " which is larger than 0xFFFF." << std::endl;
				}

				const __int8 * const ptr1 = reinterpret_cast<const __int8 * const>(std::get<0>(data1));
				const __int8 * const ptr2 = reinterpret_cast<const __int8 * const>(std::get<0>(data2));

				__int32 s11 = 0;
				__int32 s22 = 0;
				__int32 s1 = 0;
				__int32 s2 = 0;

				for (size_t i = 0; i < nElements; ++i)
				{
					const unsigned __int8 d1 = ptr1[i];
					const unsigned __int8 d2 = ptr2[i];
					s11 += d1 * d1;
					s22 += d2 * d2;
					s1 += d1;
					s2 += d2;
				}

				const double s11d = static_cast<double>(s11);
				const double s22d = static_cast<double>(s22);
				const double s1d = static_cast<double>(s1);
				const double s2d = static_cast<double>(s2);

				const size_t swap_array_nBytes = nBytes << 1;
				auto swap = _mm_malloc_m128i(swap_array_nBytes);
				auto data3 = deepCopy(data2);

				const __int8 * const ptr3 = reinterpret_cast<const __int8 * const>(std::get<0>(data3));
				double * const results_Double = reinterpret_cast<double * const>(std::get<0>(results));

				for (size_t permutation = 0; permutation < nPermutations; ++permutation) 
				{
					_mm_permute_epu8_array(data3, swap, randInts);

					__int32 s12 = 0;
					for (size_t i = 0; i < nElements; ++i)
					{
						const unsigned __int8 d1 = ptr1[i];
						const unsigned __int8 d2 = ptr3[i];
						s12 += d1 * d2;
					}
					const double s12d = static_cast<double>(s12);
					double corr = ((nElements * s12d) - (s1d*s2d)) / (sqrt((nElements*s11d) - (s1d*s1d)) * sqrt((nElements * s22d) - (s2d*s2d)));
					results_Double[permutation] = corr;
				}
				_mm_free2(data3);
				_mm_free2(swap);
			}
		}
	}

	namespace test {

		void test_mm_corr_epu8(
			const size_t nBlocks, 
			const size_t nExperiments, 
			const bool doTests)
		{
			const double delta = 0.0000001;

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
				result_ref = hli::priv::_mm_corr_epu8_ref(data1, data2, nElements);
				min_ref = std::min(min_ref, timer::get_elapsed_kcycles());

				{
					timer::reset_and_start_timer();
					result1 = hli::priv::_mm_corr_epu8_method0<8>(data1, data2, nElements);
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
					result2 = hli::priv::_mm_corr_epu8_method0<6>(data1, data2, nElements);
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
					result3 = hli::priv::_mm_corr_epu8_method1<8>(data1, data2, nElements);
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
					result4 = hli::priv::_mm_corr_epu8_method1<6>(data1, data2, nElements);
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
					result5 = hli::priv::_mm_corr_epu8_method2<8>(data1, data2, nElements);
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
					result6 = hli::priv::_mm_corr_epu8_method0<6>(data1, data2, nElements);
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
					result7 = hli::priv::_mm_corr_epu8_method3(data1, data2, nElements);
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
					result8 = hli::priv::_mm_corr_epu8_method4<8>(data1, data2, nElements);
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
					result9 = hli::priv::_mm_corr_pd_method0(data1_D, data2_D, nElements);
					min9 = std::min(min9, timer::get_elapsed_kcycles());

					if (doTests) {
						if (std::abs(result_ref.m128d_f64[0] - result9.m128d_f64[0]) > delta) {
							std::cout << "WARNING: test _mm_corr_pd_method0<8>: result-ref=" << hli::toString_f64(result_ref) << "; result4=" << hli::toString_f64(result9) << std::endl;
							return;
						}
					}
				}
			}
			printf("[_mm_corr_epu8 Ref]       : %2.5f Kcycles; %0.14f\n", min_ref, result_ref.m128d_f64[0]);
			printf("[_mm_corr_epu8_method0<8>]: %2.5f Kcycles; %0.14f; %2.3f times faster than ref\n", min1, result1.m128d_f64[0], min_ref / min1);
			printf("[_mm_corr_epu8_method0<6>]: %2.5f Kcycles; %0.14f; %2.3f times faster than ref\n", min2, result2.m128d_f64[0], min_ref / min2);
			printf("[_mm_corr_epu8_method1<8>]: %2.5f Kcycles; %0.14f; %2.3f times faster than ref\n", min3, result3.m128d_f64[0], min_ref / min3);
			printf("[_mm_corr_epu8_method1<6>]: %2.5f Kcycles; %0.14f; %2.3f times faster than ref\n", min4, result4.m128d_f64[0], min_ref / min4);
			printf("[_mm_corr_epu8_method2<8>]: %2.5f Kcycles; %0.14f; %2.3f times faster than ref\n", min5, result5.m128d_f64[0], min_ref / min5);
			printf("[_mm_corr_epu8_method2<6>]: %2.5f Kcycles; %0.14f; %2.3f times faster than ref\n", min6, result6.m128d_f64[0], min_ref / min6);
			printf("[_mm_corr_epu8_method3]   : %2.5f Kcycles; %0.14f; %2.3f times faster than ref\n", min7, result7.m128d_f64[0], min_ref / min7);
			printf("[_mm_corr_epu8_method4<8>]: %2.5f Kcycles; %0.14f; %2.3f times faster than ref\n", min8, result8.m128d_f64[0], min_ref / min8);
			printf("[_mm_corr_pd_method0]     : %2.5f Kcycles; %0.14f; %2.3f times faster than ref\n", min9, result9.m128d_f64[0], min_ref / min9);

			_mm_free2(data1);
			_mm_free2(data2);
		}

		void test_mm_corr_perm_epu8(
			const size_t nBlocks, 
			const size_t nPermutations, 
			const size_t nExperiments, 
			const bool doTests)
		{
			const double delta = 0.000001;
			const size_t nElements = nBlocks * 16;
			auto data1_r = _mm_malloc_m128i(16 * nBlocks);
			auto data2_r = _mm_malloc_m128i(16 * nBlocks);

			const size_t nBytesResults = resizeNBytes(8 * nPermutations, 16);
			//std::cout << "INFO: test_mm_corr_perm_epu8: nPermutations=" << nPermutations << "; nBytesResults=" << nBytesResults << std::endl;
			auto results = _mm_malloc_m128d(nBytesResults);
			auto results1 = _mm_malloc_m128d(nBytesResults);
			auto results2 = _mm_malloc_m128d(nBytesResults);
			auto results3 = _mm_malloc_m128d(nBytesResults);

			const __m128i seed = _mm_set_epi16(rand(), rand(), rand(), rand(), rand(), rand(), rand(), rand());
			__m128i randInt = seed;
			__m128i randInt1 = seed;
			__m128i randInt2 = seed;
			__m128i randInt3 = seed;

			const int N_BITS = 5;
			fillRand_epu8<N_BITS>(data1_r);
			fillRand_epu8<N_BITS>(data2_r);

			const std::tuple<const __m128i * const, const size_t> data1 = data1_r;
			const std::tuple<const __m128i * const, const size_t> data2 = data2_r;

			{
				double min_ref = std::numeric_limits<double>::max();
				double min0 = std::numeric_limits<double>::max();
				double min1 = std::numeric_limits<double>::max();
				double min2 = std::numeric_limits<double>::max();
				double min3 = std::numeric_limits<double>::max();

				for (size_t i = 0; i < nExperiments; ++i) 
				{
					timer::reset_and_start_timer();
					hli::priv::perm::_mm_corr_perm_epu8_method0(data1, data2, nElements, results, nPermutations, randInt);
					min_ref = std::min(min_ref, timer::get_elapsed_kcycles());

					{
						timer::reset_and_start_timer();
						hli::priv::perm::_mm_corr_perm_epu8_method1<6>(data1, data2, nElements, results1, nPermutations, randInt1);
						min1 = std::min(min1, timer::get_elapsed_kcycles());

						//for (size_t block = 0; block < (nBytesResults >> 4); ++block) {
						//	std::cout << "WARNING: test_mm_corr_perm_epu8_ref<6>: results[" << block << "] =" << hli::toString_f64(results[block]) << std::endl;
						//	std::cout << "WARNING: test_mm_corr_perm_epu8_ref<6>: results1[" << block << "]=" << hli::toString_f64(results1[block]) << std::endl;
						//}

						if (doTests) {
							if (!equal(randInt, randInt1)) {
								std::cout << "WARNING: test_mm_corr_perm_epu8_ref: randInt=" << hli::toString_u32(randInt) << "; randInt1=" << hli::toString_u32(randInt1) << std::endl;
								return;
							}
							if (i == 0) {
								for (size_t i = 0; i < (nBytesResults >> 3); ++i) {
									double diff = std::abs(getDouble(results, i) - getDouble(results1, i));
									if (diff > delta) {
										std::cout << "WARNING: _mm_corr_perm_epu8_method1<6>: i=" << i << "; diff=" << std::setprecision(30) << diff << "; result-ref=" << getDouble(results, i) << "; result1=" << getDouble(results1, i) << std::endl;
										return;
									}
								}
							}
						}
					}
					{
						timer::reset_and_start_timer();
						hli::priv::perm::_mm_corr_perm_epu8_method2<5>(data1, data2, nElements, results2, nPermutations, randInt2);
						min2 = std::min(min2, timer::get_elapsed_kcycles());

						if (doTests) {
							if (!equal(randInt, randInt2)) {
								std::cout << "WARNING: _mm_corr_perm_epu8_method2<6>: randInt=" << hli::toString_u32(randInt) << "; randInt2=" << hli::toString_u32(randInt2) << std::endl;
								return;
							}
							if (i == 0) {
								for (size_t i = 0; i < (nBytesResults >> 3); ++i) {
									double diff = std::abs(getDouble(results, i) - getDouble(results2, i));
									if (diff > delta) {
										std::cout << "WARNING: _mm_corr_perm_epu8_method3<6>: i=" << i << "; diff=" << std::setprecision(30) << diff << "; result-ref=" << getDouble(results, i) << "; result2=" << getDouble(results2, i) << std::endl;
										return;
									}
								}
							}
						}
					}
					{
						timer::reset_and_start_timer();
						hli::priv::perm::_mm_corr_perm_epu8_method3(data1, data2, nElements, results3, nPermutations, randInt3);
						min3 = std::min(min3, timer::get_elapsed_kcycles());

						if (doTests) {
							if (!equal(randInt, randInt3)) {
								std::cout << "WARNING: _mm_corr_perm_epu8_method3<6>: randInt=" << hli::toString_u32(randInt) << "; randInt3=" << hli::toString_u32(randInt3) << std::endl;
								return;
							}
							if (i == 0) {
								for (size_t i = 0; i < (nBytesResults >> 3); ++i) {
									double diff = std::abs(getDouble(results, i) - getDouble(results3, i));
									if (diff > delta) {
										std::cout << "WARNING: _mm_corr_perm_epu8_method3<6>: i=" << i << "; diff=" << std::setprecision(30) << diff << "; result-ref=" << getDouble(results, i) << "; result3=" << getDouble(results3, i) << std::endl;
										return;
									}
								}
							}
						}
					}
				}
				printf("[_mm_corr_perm_epu8 Ref]       : %2.5f Kcycles\n", min_ref);
				printf("[_mm_corr_perm_epu8_method1<8>]: %2.5f Kcycles; %2.3f times faster than ref\n", min1, min_ref / min1);
				printf("[_mm_corr_perm_epu8_method2<8>]: %2.5f Kcycles; %2.3f times faster than ref\n", min2, min_ref / min2);
				printf("[_mm_corr_perm_epu8_method3]   : %2.5f Kcycles; %2.3f times faster than ref\n", min3, min_ref / min3);
			}
			_mm_free2(data1);
			_mm_free2(data2);
			_mm_free2(results);
			_mm_free2(results1);
			_mm_free2(results2);
			_mm_free2(results3);
		}
	}

	template <int N_BITS>
	inline __m128d _mm_corr_epu8(
		const std::tuple<const __m128i * const, const size_t>& data1,
		const std::tuple<const __m128i * const, const size_t>& data2,
		const size_t nElements)
	{
//		return priv::_mm_corr_epu8_method2<N_BITS>(data1, data2, nElements);
		return priv::_mm_corr_epu8_method3(data1, data2, nElements);
	}

	template <int N_BITS>
	inline void _mm_corr_perm_epu8(
		const std::tuple<const __m128i * const, const size_t>& data1,
		const std::tuple<const __m128i * const, const size_t>& data2,
		const size_t nElements,
		const std::tuple<__m128d * const, const size_t>& results,
		const size_t nPermutations,
		__m128i& randInts)
	{
		priv::perm::_mm_corr_perm_epu8_method3(data1, data2, nElements, results, nPermutations, randInts);
#		if _DEBUG
		const double * const ptr = reinterpret_cast<double * const>(std::get<0>(results));
		for (size_t i = 0; i < nPermutations; ++i) {
			if ((ptr[i] < -1) || (ptr[i] > 1)) {
				std::cout << "WARNING: _mm_corr_epu8: _mm_corr_perm_epu8: permutation " << i << " has an invalid correlation value " << ptr[i] << std::endl;
			}
		}
#		endif
	}
}