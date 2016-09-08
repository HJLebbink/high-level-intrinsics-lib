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
#include "_mm_corr_dp.h"

namespace hli {

	namespace priv {

		inline __m128d _mm_corr_epu8_ref(
			const __m128i * const mem_addr1,
			const __m128i * const mem_addr2,
			const size_t nBytes)
		{
			//const double std_dev_d1 = sqrt(var_pop_ref(data1, nElements));
			//const double std_dev_d2 = sqrt(var_pop_ref(data1, nElements));
			//return covar_pop_ref(data1, data2, nElements) / (std_dev_d1 * std_dev_d2);

			const __m128d var1 = _mm_variance_epu8_ref(mem_addr1, nBytes);
			const __m128d var2 = _mm_variance_epu8_ref(mem_addr2, nBytes);
			const __m128d covar = _mm_covar_epu8_ref(mem_addr1, mem_addr2, nBytes);
			const __m128d corr = _mm_div_pd(covar, _mm_mul_pd(_mm_sqrt_pd(var1), _mm_sqrt_pd(var2)));
			//std::cout << "INFO: _mm_corr_epu8::_mm_corr_epu8_ref: var1=" << var1.m128d_f64[0] << "; var2=" << var2.m128d_f64[0] << "; covar=" << covar.m128d_f64[0] << "; corr=" << corr.m128d_f64[0] << std::endl;
			return corr;
		}

		template <int N_BITS>
		inline __m128d _mm_corr_epu8_method1(
			const __m128i * const mem_addr1,
			const __m128i * const mem_addr2,
			const size_t nBytes,
			const __m128d average1,
			const __m128d average2)
		{
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
					const __m128d d1b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_swap_64(d1)), average1);
					var1 = _mm_add_pd(var1, _mm_mul_pd(d1a, d1a));
					var1 = _mm_add_pd(var1, _mm_mul_pd(d1b, d1b));
					const __m128i d2 = _mm_cvtepu8_epi32(data2);
					const __m128d d2a = _mm_sub_pd(_mm_cvtepi32_pd(d2), average2);
					const __m128d d2b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_swap_64(d2)), average2);
					var2 = _mm_add_pd(var2, _mm_mul_pd(d2a, d2a));
					var2 = _mm_add_pd(var2, _mm_mul_pd(d2b, d2b));
					covar = _mm_add_pd(covar, _mm_mul_pd(d1a, d2a));
					covar = _mm_add_pd(covar, _mm_mul_pd(d1b, d2b));
					//std::cout << "INFO: _mm_corr_epu8::_mm_corr_epu8_method1: block=" << ((8 * block) + 0) << "; d1=" << toString_f64(d1a) << "; d2=" << toString_f64(d2a) << std::endl;
					//std::cout << "INFO: _mm_corr_epu8::_mm_corr_epu8_method1: block=" << ((8 * block) + 1) << "; d1=" << toString_f64(d1b) << "; d2=" << toString_f64(d2b) << std::endl;
				}
				{
					const __m128i d1 = _mm_cvtepu8_epi32(_mm_shuffle_epi32(data1, _MM_SHUFFLE_EPI32_INT(1, 1, 1, 1)));
					const __m128d d1a = _mm_sub_pd(_mm_cvtepi32_pd(d1), average1);
					const __m128d d1b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_swap_64(d1)), average1);
					var1 = _mm_add_pd(var1, _mm_mul_pd(d1a, d1a));
					var1 = _mm_add_pd(var1, _mm_mul_pd(d1b, d1b));
					const __m128i d2 = _mm_cvtepu8_epi32(_mm_shuffle_epi32(data2, _MM_SHUFFLE_EPI32_INT(1, 1, 1, 1)));
					const __m128d d2a = _mm_sub_pd(_mm_cvtepi32_pd(d2), average2);
					const __m128d d2b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_swap_64(d2)), average2);
					var2 = _mm_add_pd(var2, _mm_mul_pd(d2a, d2a));
					var2 = _mm_add_pd(var2, _mm_mul_pd(d2b, d2b));
					covar = _mm_add_pd(covar, _mm_mul_pd(d1a, d2a));
					covar = _mm_add_pd(covar, _mm_mul_pd(d1b, d2b));
					//std::cout << "INFO: _mm_corr_epu8::_mm_corr_epu8_method1: block=" << ((8 * block) + 2) << "; d1=" << toString_f64(d1a) << "; d2=" << toString_f64(d2a) << std::endl;
					//std::cout << "INFO: _mm_corr_epu8::_mm_corr_epu8_method1: block=" << ((8 * block) + 3) << "; d1=" << toString_f64(d1b) << "; d2=" << toString_f64(d2b) << std::endl;
				}
				{
					const __m128i d1 = _mm_cvtepu8_epi32(_mm_shuffle_epi32(data1, _MM_SHUFFLE_EPI32_INT(2, 2, 2, 2)));
					const __m128d d1a = _mm_sub_pd(_mm_cvtepi32_pd(d1), average1);
					const __m128d d1b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_swap_64(d1)), average1);
					var1 = _mm_add_pd(var1, _mm_mul_pd(d1a, d1a));
					var1 = _mm_add_pd(var1, _mm_mul_pd(d1b, d1b));
					const __m128i d2 = _mm_cvtepu8_epi32(_mm_shuffle_epi32(data2, _MM_SHUFFLE_EPI32_INT(2, 2, 2, 2)));
					const __m128d d2a = _mm_sub_pd(_mm_cvtepi32_pd(d2), average2);
					const __m128d d2b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_swap_64(d2)), average2);
					var2 = _mm_add_pd(var2, _mm_mul_pd(d2a, d2a));
					var2 = _mm_add_pd(var2, _mm_mul_pd(d2b, d2b));
					covar = _mm_add_pd(covar, _mm_mul_pd(d1a, d2a));
					covar = _mm_add_pd(covar, _mm_mul_pd(d1b, d2b));
					//std::cout << "INFO: _mm_corr_epu8::_mm_corr_epu8_method1: block=" << ((8 * block) + 4) << "; d1=" << toString_f64(d1a) << "; d2=" << toString_f64(d2a) << std::endl;
					//std::cout << "INFO: _mm_corr_epu8::_mm_corr_epu8_method1: block=" << ((8 * block) + 5) << "; d1=" << toString_f64(d1b) << "; d2=" << toString_f64(d2b) << std::endl;
				}
				{
					const __m128i d1 = _mm_cvtepu8_epi32(_mm_shuffle_epi32(data1, _MM_SHUFFLE_EPI32_INT(3, 3, 3, 3)));
					const __m128d d1a = _mm_sub_pd(_mm_cvtepi32_pd(d1), average1);
					const __m128d d1b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_swap_64(d1)), average1);
					var1 = _mm_add_pd(var1, _mm_mul_pd(d1a, d1a));
					var1 = _mm_add_pd(var1, _mm_mul_pd(d1b, d1b));
					const __m128i d2 = _mm_cvtepu8_epi32(_mm_shuffle_epi32(data2, _MM_SHUFFLE_EPI32_INT(3, 3, 3, 3)));
					const __m128d d2a = _mm_sub_pd(_mm_cvtepi32_pd(d2), average2);
					const __m128d d2b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_swap_64(d2)), average2);
					var2 = _mm_add_pd(var2, _mm_mul_pd(d2a, d2a));
					var2 = _mm_add_pd(var2, _mm_mul_pd(d2b, d2b));
					covar = _mm_add_pd(covar, _mm_mul_pd(d1a, d2a));
					covar = _mm_add_pd(covar, _mm_mul_pd(d1b, d2b));
					//std::cout << "INFO: _mm_corr_epu8::_mm_corr_epu8_method1: block=" << ((8 * block) + 6) << "; d1=" << toString_f64(d1a) << "; d2=" << toString_f64(d2a) << std::endl;
					//std::cout << "INFO: _mm_corr_epu8::_mm_corr_epu8_method1: block=" << ((8 * block) + 7) << "; d1=" << toString_f64(d1b) << "; d2=" << toString_f64(d2b) << std::endl;
				}
				//std::cout << "INFO: hli:::_mm_variance_epu8: data=" << toString_u8(data) << "; d1=" << toString_u32(d1) << "; d2=" << toString_u32(d2) << "; d3=" << toString_u32(d3) << "; d4=" << toString_u32(d4) << std::endl;
			}
			const __m128d nElements = _mm_set1_pd(static_cast<double>(nBytes));
			covar = _mm_div_pd(_mm_hadd_pd(covar, covar), nElements);
			var1 = _mm_div_pd(_mm_hadd_pd(var1, var1), nElements);
			var2 = _mm_div_pd(_mm_hadd_pd(var2, var2), nElements);
			const __m128d corr = _mm_div_pd(covar, _mm_sqrt_pd(_mm_mul_pd(var1, var2)));
			//std::cout << "INFO: _mm_corr_epu8::_mm_corr_epu8_method1: var1=" << var1.m128d_f64[0] << "; var2=" << var2.m128d_f64[0] << "; covar=" << covar.m128d_f64[0] << "; corr=" << corr.m128d_f64[0] << std::endl;
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

			return _mm_corr_epu8_method1<N_BITS>(mem_addr1, mem_addr2, nBytes, average1, average2);
		}

		inline __m128d _mm_corr_epu8_method2(
			const __m128i * const mem_addr1,
			const __m128i * const mem_addr2,
			const size_t nBytes,
			const __m128d average1,
			const __m128d average2)
		{
			const __m128d var1 = _mm_variance_epu8(mem_addr1, nBytes, average1);
			const __m128d var2 = _mm_variance_epu8(mem_addr2, nBytes, average2);
			const __m128d covar = _mm_covar_epu8(mem_addr1, mem_addr2, nBytes, average1, average2);
			const __m128d corr = _mm_div_pd(covar, _mm_sqrt_pd(_mm_mul_pd(var1, var2)));
			//std::cout << "INFO: _mm_corr_epu8::_mm_corr_epu8_method2: var1=" << var1.m128d_f64[0] << "; var2=" << var2.m128d_f64[0] << "; covar=" << covar.m128d_f64[0] << "; corr=" << corr.m128d_f64[0] << std::endl;
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


		template <int N_BITS>
		inline __m128d calc_variance(
			const __m128i * const mem_addr1,
			const size_t nBytes,
			std::tuple<__m128d * const, const size_t> data_double_tup)
		{
			const __m128d nElements = _mm_set1_pd(static_cast<double>(nBytes));
			const __m128d average = _mm_div_pd(_mm_cvtepi32_pd(_mm_hadd_epu8<N_BITS>(mem_addr1, nBytes)), nElements);
			const size_t nBlocksData = nBytes >> 4;

			__m128d * const data_double = std::get<0>(data_double_tup);
			__m128d var = _mm_setzero_pd();

			for (size_t block = 0; block < nBlocksData; ++block) {
				const __m128i data = mem_addr1[block];
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
			return _mm_div_pd(_mm_hadd_pd(var, var), nElements);
		}

		template <int N_BITS>
		inline __m128d _mm_corr_epu8_method3(
			const __m128i * const mem_addr1,
			const __m128i * const mem_addr2,
			const size_t nBytes)
		{
			const size_t nBlocksInput = nBytes >> 4;
			const size_t nBlocksDouble = nBlocksInput * 8;

			auto data1 = _mm_malloc_m128d(nBlocksDouble * 16);
			auto data2 = _mm_malloc_m128d(nBlocksDouble * 16);

			const __m128d var1 = calc_variance<N_BITS>(mem_addr1, nBytes, data1);
			const __m128d var2 = calc_variance<N_BITS>(mem_addr2, nBytes, data2);
			const __m128d var1_2 = _mm_sqrt_pd(_mm_mul_pd(var1, var2));
			const __m128d corr = _mm_corr_dp_method3(data1, data2, var1_2);

			_mm_free2(data1);
			_mm_free2(data2);
			return corr;
		}

		template <int N_BITS>
		inline __m128d _mm_corr_epu8_method4(
			const std::tuple<__m128i * const, const size_t> data1,
			const std::tuple<__m128i * const, const size_t> data2)
		{
			static_assert(N_BITS < 7, "only works if N_BITS is smaller than 7");

			const size_t nBytes = std::get<1>(data1);
			const size_t nBlocks = nBytes >> 4;
			if (nBlocks == 0) return _mm_setzero_pd();

			const __m128i * const ptr1 = std::get<0>(data1);
			const __m128i * const ptr2 = std::get<0>(data2);

			__m128i sum1 = _mm_setzero_si128();
			__m128i sum2 = _mm_setzero_si128();
			__m128i sumProd12 = _mm_setzero_si128();
			__m128i sumProd11 = _mm_setzero_si128();
			__m128i sumProd22 = _mm_setzero_si128();


			for (size_t block = 0; block < nBlocks; ++block) {
				const __m128i d1 = ptr1[block];
				const __m128i d2 = ptr2[block];

				sum1 = _mm_add_epi64(sum1, _mm_sad_epu8(d1, _mm_setzero_si128()));
				sum2 = _mm_add_epi64(sum2, _mm_sad_epu8(d2, _mm_setzero_si128()));
				{
					const __m128i prod = _mm_maddubs_epi16(d1, d2);
					sumProd12 = _mm_add_epi64(_mm_and_si128(prod, _mm_set1_epi64x(0xFFFF)));
					sumProd12 = _mm_add_epi64(_mm_and_si128(_mm_shuffle_ (prod, 16), _mm_set1_epi64x(0xFFFF)));
					sumProd12 = _mm_add_epi64(_mm_and_si128(prod, _mm_set1_epi64x(0xFFFF)));
					sumProd12 = _mm_add_epi64(_mm_and_si128(prod, _mm_set1_epi64x(0xFFFF)));
				}
				{
					const __m128i prod = _mm_maddubs_epi16(d1, d1);
					sumProd11 = _mm_add_epi32(sumProd11, _mm_unpacklo_epi16(_mm_setzero_si128(), prod));
					sumProd11 = _mm_add_epi32(sumProd11, _mm_unpackhi_epi16(_mm_setzero_si128(), prod));
				}
				{
					const __m128i prod = _mm_maddubs_epi16(d2, d2);
					sumProd22 = _mm_add_epi32(sumProd22, _mm_unpacklo_epi16(_mm_setzero_si128(), prod));
					sumProd22 = _mm_add_epi32(sumProd22, _mm_unpackhi_epi16(_mm_setzero_si128(), prod));
				}
			}

			sum1 = _mm_hadd_epi64(sum1);
			sum2 = _mm_hadd_epi64(sum2);



			const __m128d covar = _mm_sub_epi64(_mm_mul_epu64(sumProd12, n) __mm_mul_epi64(sum1, sum2));

			const __m128d corr = 
			return corr;
		}

		namespace perm {

			inline void _mm_corr_perm_epu8_ref(
				const __m128i * const mem_addr1,
				const __m128i * const mem_addr2,
				const size_t nBytes,
				__m128d * const results,
				const size_t nPermutations,
				__m128i& randInts)
			{
				__m128i * const mem_addr3 = static_cast<__m128i * const>(_mm_malloc(nBytes, 16));
				memcpy(mem_addr3, mem_addr2, nBytes);

				const size_t nElements = nBytes;
				const size_t swap_array_nBytes = nElements << 1;
				__m128i * const swap_array = static_cast<__m128i * const>(_mm_malloc(swap_array_nBytes, 16));

				double * const results_double = reinterpret_cast<double * const>(results);
				for (size_t permutation = 0; permutation < nPermutations; ++permutation)
				{
					_mm_permute_epu8_array_ref(mem_addr3, nBytes, swap_array, swap_array_nBytes, randInts);
					const __m128d corr1 = _mm_corr_epu8_ref(mem_addr1, mem_addr3, nBytes);
					results_double[permutation] = corr1.m128d_f64[0];
				}
				_mm_free(mem_addr3);
				_mm_free(swap_array);
			}

			template <int N_BITS>
			inline void _mm_corr_perm_epu8_method1(
				const __m128i * const mem_addr1,
				const __m128i * const mem_addr2,
				const size_t nBytes,
				__m128d * const results,
				const size_t nPermutations,
				__m128i& randInts)
			{
				__m128i * const mem_addr3 = static_cast<__m128i * const>(_mm_malloc(nBytes, 16));
				memcpy(mem_addr3, mem_addr2, nBytes);

				const size_t swap_array_nBytes = nBytes << 1;
				__m128i * const swap_array = static_cast<__m128i * const>(_mm_malloc(swap_array_nBytes, 16));

				const __m128d nElements = _mm_set1_pd(static_cast<double>(nBytes));
				//std::cout << "INFO: _mm_corr_epu8::_mm_corr_perm_epu8_method1: nElements=" << toString_f64(nElements) << std::endl;
				const __m128d average1 = _mm_div_pd(_mm_cvtepi32_pd(_mm_hadd_epu8<N_BITS>(mem_addr1, nBytes)), nElements);
				const __m128d average2 = _mm_div_pd(_mm_cvtepi32_pd(_mm_hadd_epu8<N_BITS>(mem_addr2, nBytes)), nElements);

				double * const results_double = reinterpret_cast<double * const>(results);
				for (size_t permutation = 0; permutation < nPermutations; ++permutation) {
					_mm_permute_epu8_array(mem_addr3, nBytes, swap_array, swap_array_nBytes, randInts);
					const __m128d corr = _mm_corr_epu8_method1<N_BITS>(mem_addr1, mem_addr3, nBytes, average1, average2);
					results_double[permutation] = corr.m128d_f64[0];
				}
				_mm_free(mem_addr3);
				_mm_free(swap_array);
			}

			template <int N_BITS>
			inline void _mm_corr_perm_epu8_method3(
				const __m128i * const mem_addr1,
				const __m128i * const mem_addr2,
				const size_t nBytes,
				__m128d * const results,
				const size_t nPermutations,
				__m128i& randInts)
			{
				auto data1 = _mm_malloc_m128d(8 * nBytes);
				auto data2 = _mm_malloc_m128d(8 * nBytes);
				auto swap = _mm_malloc_m128i(2 * nBytes);

				const __m128d var1 = calc_variance<N_BITS>(mem_addr1, nBytes, data1);
				const __m128d var2 = calc_variance<N_BITS>(mem_addr2, nBytes, data2);
				const __m128d var1_2 = _mm_sqrt_pd(_mm_mul_pd(var1, var2));

				//std::cout << "INFO: _mm_corr_epu8::_mm_corr_perm_epu8_method3: var1=" << var1.m128d_f64[0] << "; var2=" << var2.m128d_f64[0] << std::endl;

				double * const results_double = reinterpret_cast<double * const>(results);
				for (size_t permutation = 0; permutation < nPermutations; ++permutation) {
					_mm_permute_dp_array(data2, swap, randInts);
					const __m128d corr = _mm_corr_dp_method3(data1, data2, var1_2);
					//std::cout << "INFO: _mm_corr_epu8::_mm_corr_perm_epu8_method3: corr=" << corr.m128d_f64[0] << std::endl;
					results_double[permutation] = corr.m128d_f64[0];
				}
				_mm_free2(data1);
				_mm_free2(data2);
				_mm_free2(swap);
			}
		}
	}

	namespace test {

		void test_mm_corr_epu8(const size_t nBlocks, const size_t nExperiments, const bool doTests)
		{
			const double delta = 0.0000001;
			const size_t nBytes = resizeNBytes(16 * nBlocks, 16);
			__m128i * const mem_addr1 = static_cast<__m128i *>(_mm_malloc(nBytes, 16));
			__m128i * const mem_addr2 = static_cast<__m128i *>(_mm_malloc(nBytes, 16));
			fillRand_epu8<5>(mem_addr1, nBytes);
			fillRand_epu8<5>(mem_addr2, nBytes);

			double min_ref = std::numeric_limits<double>::max();
			double min1 = std::numeric_limits<double>::max();
			double min2 = std::numeric_limits<double>::max();
			double min3 = std::numeric_limits<double>::max();
			double min4 = std::numeric_limits<double>::max();
			double min5 = std::numeric_limits<double>::max();
			double min6 = std::numeric_limits<double>::max();

			__m128d result_ref, result1, result2, result3, result4, result5, result6;

			for (size_t i = 0; i < nExperiments; ++i) {

				timer::reset_and_start_timer();
				result_ref = hli::priv::_mm_corr_epu8_ref(mem_addr1, mem_addr2, nBytes);
				min_ref = std::min(min_ref, timer::get_elapsed_kcycles());

				{
					timer::reset_and_start_timer();
					result1 = hli::priv::_mm_corr_epu8_method1<8>(mem_addr1, mem_addr2, nBytes);
					min1 = std::min(min1, timer::get_elapsed_kcycles());

					if (doTests) {
						if (std::abs(result_ref.m128d_f64[0] - result1.m128d_f64[0]) > delta) {
							std::cout << "WARNING: test _mm_corr_epu8_method1<8>: result-ref=" << hli::toString_f64(result_ref) << "; result=" << hli::toString_f64(result1) << std::endl;
							return;
						}
					}
				}
				{
					timer::reset_and_start_timer();
					result2 = hli::priv::_mm_corr_epu8_method1<6>(mem_addr1, mem_addr2, nBytes);
					min2 = std::min(min2, timer::get_elapsed_kcycles());

					if (doTests) {
						if (std::abs(result_ref.m128d_f64[0] - result2.m128d_f64[0]) > delta) {
							std::cout << "WARNING: test _mm_corr_epu8_method1<6>: result-ref=" << hli::toString_f64(result_ref) << "; result=" << hli::toString_f64(result2) << std::endl;
							return;
						}
					}
				}
				{
					timer::reset_and_start_timer();
					result3 = hli::priv::_mm_corr_epu8_method2<8>(mem_addr1, mem_addr2, nBytes);
					min3 = std::min(min3, timer::get_elapsed_kcycles());

					if (doTests) {
						if (std::abs(result_ref.m128d_f64[0] - result3.m128d_f64[0]) > delta) {
							std::cout << "WARNING: test _mm_corr_epu8_method2<8>: result-ref=" << hli::toString_f64(result_ref) << "; result=" << hli::toString_f64(result3) << std::endl;
							return;
						}
					}
				}
				{
					timer::reset_and_start_timer();
					result4 = hli::priv::_mm_corr_epu8_method2<6>(mem_addr1, mem_addr2, nBytes);
					min4 = std::min(min4, timer::get_elapsed_kcycles());

					if (doTests) {
						if (std::abs(result_ref.m128d_f64[0] - result4.m128d_f64[0]) > delta) {
							std::cout << "WARNING: test _mm_corr_epu8_method2<6>: result-ref=" << hli::toString_f64(result_ref) << "; result=" << hli::toString_f64(result4) << std::endl;
							return;
						}
					}
				}
				{
					timer::reset_and_start_timer();
					result5 = hli::priv::_mm_corr_epu8_method3<8>(mem_addr1, mem_addr2, nBytes);
					min5 = std::min(min5, timer::get_elapsed_kcycles());

					if (doTests) {
						if (std::abs(result_ref.m128d_f64[0] - result5.m128d_f64[0]) > delta) {
							std::cout << "WARNING: test _mm_corr_epu8_method3<8>: result-ref=" << hli::toString_f64(result_ref) << "; result=" << hli::toString_f64(result5) << std::endl;
							return;
						}
					}
				}
				{
					timer::reset_and_start_timer();
					result6 = hli::priv::_mm_corr_epu8_method3<6>(mem_addr1, mem_addr2, nBytes);
					min6 = std::min(min6, timer::get_elapsed_kcycles());

					if (doTests) {
						if (std::abs(result_ref.m128d_f64[0] - result6.m128d_f64[0]) > delta) {
							std::cout << "WARNING: test _mm_corr_epu8_method3<6>: result-ref=" << hli::toString_f64(result_ref) << "; result=" << hli::toString_f64(result6) << std::endl;
							return;
						}
					}
				}
			}
			printf("[_mm_corr_epu8 Ref]       : %2.5f Kcycles; %0.14f\n", min_ref, result_ref.m128d_f64[0]);
			printf("[_mm_corr_epu8_method1<8>]: %2.5f Kcycles; %0.14f; %2.3f times faster than ref\n", min1, result1.m128d_f64[0], min_ref / min1);
			printf("[_mm_corr_epu8_method1<6>]: %2.5f Kcycles; %0.14f; %2.3f times faster than ref\n", min2, result2.m128d_f64[0], min_ref / min2);
			printf("[_mm_corr_epu8_method2<8>]: %2.5f Kcycles; %0.14f; %2.3f times faster than ref\n", min3, result3.m128d_f64[0], min_ref / min3);
			printf("[_mm_corr_epu8_method2<6>]: %2.5f Kcycles; %0.14f; %2.3f times faster than ref\n", min4, result4.m128d_f64[0], min_ref / min4);
			printf("[_mm_corr_epu8_method3<8>]: %2.5f Kcycles; %0.14f; %2.3f times faster than ref\n", min5, result3.m128d_f64[0], min_ref / min5);
			printf("[_mm_corr_epu8_method3<6>]: %2.5f Kcycles; %0.14f; %2.3f times faster than ref\n", min6, result4.m128d_f64[0], min_ref / min6);

			_mm_free(mem_addr1);
			_mm_free(mem_addr2);
		}

		void test_mm_corr_perm_epu8(const size_t nBlocks, const size_t nPermutations, const size_t nExperiments, const bool doTests)
		{
			const double delta = 0.000001;
			const size_t nBytesData = resizeNBytes(16 * nBlocks, 16);
			__m128i * const data1 = static_cast<__m128i *>(_mm_malloc(nBytesData, 16));
			__m128i * const data2 = static_cast<__m128i *>(_mm_malloc(nBytesData, 16));

			const size_t nBytesResults = resizeNBytes(8 * nPermutations, 16);
			//std::cout << "INFO: test_mm_corr_perm_epu8: nPermutations=" << nPermutations << "; nBytesResults=" << nBytesResults << std::endl;
			__m128d * const results = static_cast<__m128d *>(_mm_malloc(nBytesResults, 16));
			__m128d * const results1 = static_cast<__m128d *>(_mm_malloc(nBytesResults, 16));
			__m128d * const results2 = static_cast<__m128d *>(_mm_malloc(nBytesResults, 16));

			const __m128i seed = _mm_set_epi32(rand(), rand(), rand(), rand());
			__m128i randInt = seed;
			__m128i randInt1 = seed;
			__m128i randInt2 = seed;

			const int N_BITS = 5;
			fillRand_epu8<N_BITS>(data1, nBytesData);
			fillRand_epu8<N_BITS>(data2, nBytesData);

			{
				double min_ref = std::numeric_limits<double>::max();
				double min1 = std::numeric_limits<double>::max();
				double min2 = std::numeric_limits<double>::max();

				for (size_t i = 0; i < nExperiments; ++i) {

					timer::reset_and_start_timer();
					hli::priv::perm::_mm_corr_perm_epu8_ref(data1, data2, nBytesData, results, nPermutations, randInt);
					min_ref = std::min(min_ref, timer::get_elapsed_kcycles());

					{
						timer::reset_and_start_timer();
						hli::priv::perm::_mm_corr_perm_epu8_method1<6>(data1, data2, nBytesData, results1, nPermutations, randInt1);
						min1 = std::min(min1, timer::get_elapsed_kcycles());

						//for (size_t block = 0; block < (nBytesResults >> 4); ++block) {
						//	std::cout << "WARNING: test_mm_corr_perm_epu8_ref<6>: results[" << block << "] =" << hli::toString_f64(results[block]) << std::endl;
						//	std::cout << "WARNING: test_mm_corr_perm_epu8_ref<6>: results1[" << block << "]=" << hli::toString_f64(results1[block]) << std::endl;
						//}

						if (doTests) {
							if (!equal(randInt, randInt1)) {
								std::cout << "WARNING: test_mm_corr_perm_epu8_ref<6>: randInt=" << hli::toString_u32(randInt) << "; randInt1=" << hli::toString_u32(randInt1) << std::endl;
								return;
							}
							if (i == 0) {
								for (size_t block = 0; block < (nBytesResults >> 4); ++block) {
									double diff = std::abs(results[block].m128d_f64[0] - results1[block].m128d_f64[0]);
									if (diff > delta) {
										std::cout << "WARNING: _mm_corr_perm_epu8_method3<6>: block=" << block << "; diff=" << std::setprecision(30) << diff << "; result-ref=" << results[block].m128d_f64[0] << "; result1=" << results1[block].m128d_f64[0] << std::endl;
										return;
									}
									diff = std::abs(results[block].m128d_f64[1] - results1[block].m128d_f64[1]);
									if (diff > delta) {
										std::cout << "WARNING: _mm_corr_perm_epu8_method3<6>: block=" << block << "; diff=" << std::setprecision(30) << diff << "; result-ref=" << results[block].m128d_f64[1] << "; result1=" << results1[block].m128d_f64[1] << std::endl;
										return;
									}
								}
							}
						}
					}
					{
						timer::reset_and_start_timer();
						hli::priv::perm::_mm_corr_perm_epu8_method3<5>(data1, data2, nBytesData, results2, nPermutations, randInt2);
						min2 = std::min(min2, timer::get_elapsed_kcycles());

						if (doTests) {
							if (!equal(randInt, randInt2)) {
								std::cout << "WARNING: _mm_corr_perm_epu8_method3<6>: randInt=" << hli::toString_u32(randInt) << "; randInt2=" << hli::toString_u32(randInt2) << std::endl;
								return;
							}
							if (i == 0) {
								for (size_t block = 0; block < (nBytesResults >> 4); ++block) {
									double diff = std::abs(results[block].m128d_f64[0] - results2[block].m128d_f64[0]);
									if (diff > delta) {
										std::cout << "WARNING: _mm_corr_perm_epu8_method3<6>: block=" << block << "; diff=" << std::setprecision(30) << "; result1=" << results1[block].m128d_f64[0] << diff << "; result-ref=" << results[block].m128d_f64[0] << "; result2=" << results2[block].m128d_f64[0] << std::endl;
										return;
									}
									diff = std::abs(results[block].m128d_f64[1] - results2[block].m128d_f64[1]);
									if (diff > delta) {
										std::cout << "WARNING: _mm_corr_perm_epu8_method3<6>: block=" << block << "; diff=" << std::setprecision(30) << "; result1=" << results1[block].m128d_f64[1] << diff << "; result-ref=" << results[block].m128d_f64[1] << "; result2=" << results2[block].m128d_f64[1] << std::endl;
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
			}
			_mm_free(data1);
			_mm_free(data2);
			_mm_free(results);
			_mm_free(results1);
			_mm_free(results2);
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

	template <int N_BITS>
	inline void _mm_corr_perm_epu8(
		const __m128i * const mem_addr1,
		const __m128i * const mem_addr2,
		const size_t nBytes,
		const __m128d * const results,
		const size_t nPermutations,
		__m128i& randInts)
	{
		//return priv::_mm_corr_perm_epu8_method1(mem_addr1, mem_addr2, nBytes, results, nPermutations, randInts);
		return priv::_mm_corr_perm_epu8_method3(mem_addr1, mem_addr2, nBytes, results, nPermutations, randInts);
	}
}