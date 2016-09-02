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
#include "_mm_permute_array.h"

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
					const __m128d d1b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d1, 0b01011110)), average1);
					var1 = _mm_add_pd(var1, _mm_mul_pd(d1a, d1a));
					var1 = _mm_add_pd(var1, _mm_mul_pd(d1b, d1b));
					const __m128i d2 = _mm_cvtepu8_epi32(data2);
					const __m128d d2a = _mm_sub_pd(_mm_cvtepi32_pd(d2), average2);
					const __m128d d2b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d2, 0b01011110)), average2);
					var2 = _mm_add_pd(var2, _mm_mul_pd(d2a, d2a));
					var2 = _mm_add_pd(var2, _mm_mul_pd(d2b, d2b));
					covar = _mm_add_pd(covar, _mm_mul_pd(d1a, d2a));
					covar = _mm_add_pd(covar, _mm_mul_pd(d1b, d2b));
					//std::cout << "INFO: _mm_corr_epu8::_mm_corr_epu8_method1: block=" << ((8 * block) + 0) << "; d1=" << toString_f64(d1a) << "; d2=" << toString_f64(d2a) << std::endl;
					//std::cout << "INFO: _mm_corr_epu8::_mm_corr_epu8_method1: block=" << ((8 * block) + 1) << "; d1=" << toString_f64(d1b) << "; d2=" << toString_f64(d2b) << std::endl;
				}
				{
					const __m128i d1 = _mm_cvtepu8_epi32(_mm_shuffle_epi32(data1, 0b01010101));
					const __m128d d1a = _mm_sub_pd(_mm_cvtepi32_pd(d1), average1);
					const __m128d d1b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d1, 0b01011110)), average1);
					var1 = _mm_add_pd(var1, _mm_mul_pd(d1a, d1a));
					var1 = _mm_add_pd(var1, _mm_mul_pd(d1b, d1b));
					const __m128i d2 = _mm_cvtepu8_epi32(_mm_shuffle_epi32(data2, 0b01010101));
					const __m128d d2a = _mm_sub_pd(_mm_cvtepi32_pd(d2), average2);
					const __m128d d2b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d2, 0b01011110)), average2);
					var2 = _mm_add_pd(var2, _mm_mul_pd(d2a, d2a));
					var2 = _mm_add_pd(var2, _mm_mul_pd(d2b, d2b));
					covar = _mm_add_pd(covar, _mm_mul_pd(d1a, d2a));
					covar = _mm_add_pd(covar, _mm_mul_pd(d1b, d2b));
					//std::cout << "INFO: _mm_corr_epu8::_mm_corr_epu8_method1: block=" << ((8 * block) + 2) << "; d1=" << toString_f64(d1a) << "; d2=" << toString_f64(d2a) << std::endl;
					//std::cout << "INFO: _mm_corr_epu8::_mm_corr_epu8_method1: block=" << ((8 * block) + 3) << "; d1=" << toString_f64(d1b) << "; d2=" << toString_f64(d2b) << std::endl;
				}
				{
					const __m128i d1 = _mm_cvtepu8_epi32(_mm_shuffle_epi32(data1, 0b10101010));
					const __m128d d1a = _mm_sub_pd(_mm_cvtepi32_pd(d1), average1);
					const __m128d d1b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d1, 0b01011110)), average1);
					var1 = _mm_add_pd(var1, _mm_mul_pd(d1a, d1a));
					var1 = _mm_add_pd(var1, _mm_mul_pd(d1b, d1b));
					const __m128i d2 = _mm_cvtepu8_epi32(_mm_shuffle_epi32(data2, 0b10101010));
					const __m128d d2a = _mm_sub_pd(_mm_cvtepi32_pd(d2), average2);
					const __m128d d2b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d2, 0b01011110)), average2);
					var2 = _mm_add_pd(var2, _mm_mul_pd(d2a, d2a));
					var2 = _mm_add_pd(var2, _mm_mul_pd(d2b, d2b));
					covar = _mm_add_pd(covar, _mm_mul_pd(d1a, d2a));
					covar = _mm_add_pd(covar, _mm_mul_pd(d1b, d2b));
					//std::cout << "INFO: _mm_corr_epu8::_mm_corr_epu8_method1: block=" << ((8 * block) + 4) << "; d1=" << toString_f64(d1a) << "; d2=" << toString_f64(d2a) << std::endl;
					//std::cout << "INFO: _mm_corr_epu8::_mm_corr_epu8_method1: block=" << ((8 * block) + 5) << "; d1=" << toString_f64(d1b) << "; d2=" << toString_f64(d2b) << std::endl;
				}
				{
					const __m128i d1 = _mm_cvtepu8_epi32(_mm_shuffle_epi32(data1, 0b11111111));
					const __m128d d1a = _mm_sub_pd(_mm_cvtepi32_pd(d1), average1);
					const __m128d d1b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d1, 0b01011110)), average1);
					var1 = _mm_add_pd(var1, _mm_mul_pd(d1a, d1a));
					var1 = _mm_add_pd(var1, _mm_mul_pd(d1b, d1b));
					const __m128i d2 = _mm_cvtepu8_epi32(_mm_shuffle_epi32(data2, 0b11111111));
					const __m128d d2a = _mm_sub_pd(_mm_cvtepi32_pd(d2), average2);
					const __m128d d2b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d2, 0b01011110)), average2);
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

		inline __m128d _mm_corr_dp_method3(
			const __m128d * const data_array1,
			const __m128d * const data_array2,
			const size_t nBytes,
			const __m128d var1_2)
		{
			const size_t nBlocks = nBytes >> 4;
			const __m128d nElements = _mm_set1_pd(static_cast<double>(nBytes >> 3));
			//std::cout << "INFO: _mm_corr_epu8::_mm_corr_dp_method3: nBlocks=" << nBlocks << "; nElements=" << toString_f64(nElements) << std::endl;


			//for (size_t block = 0; block < nBlocks; ++block) {
			//	std::cout << "INFO: _mm_corr_epu8::_mm_corr_dp_method3: block=" << block << "; d1=" << toString_f64(data_array1[block]) << std::endl;
			//}
			//for (size_t block = 0; block < nBlocks; ++block) {
			//	std::cout << "INFO: _mm_corr_epu8::_mm_corr_dp_method3: block=" << block << "; d2=" << toString_f64(data_array2[block]) << std::endl;
			//}

			__m128d covar = _mm_setzero_pd();
			for (size_t block = 0; block < nBlocks; ++block) {
				//std::cout << "INFO: _mm_corr_epu8::calc_variance: block=" << block << ": d1=" << toString_f64(data_array1[block]) << "; d2=" << toString_f64(data_array2[block]) << std::endl;
				covar = _mm_add_pd(covar, _mm_mul_pd(data_array1[block], data_array2[block]));
			}
			covar = _mm_hadd_pd(covar, covar);
			covar = _mm_div_pd(covar, nElements);
			const __m128d corr = _mm_div_pd(covar, var1_2);
			//std::cout << "INFO: _mm_corr_epu8::_mm_corr_dp_method3: covar=" << covar.m128d_f64[0] << "; corr=" << corr.m128d_f64[0] << std::endl;
			return corr;
		}

		template <int N_BITS>
		inline __m128d calc_variance(
			const __m128i * const mem_addr1,
			const size_t nBytes,
			__m128d * const data_double)
		{
			const __m128d nElements = _mm_set1_pd(static_cast<double>(nBytes));
			const __m128d average = _mm_div_pd(_mm_cvtepi32_pd(_mm_hadd_epu8<N_BITS>(mem_addr1, nBytes)), nElements);
			const size_t nBlocksData = nBytes >> 4;

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

			__m128d * const data_array1 = static_cast<__m128d * const>(_mm_malloc(nBlocksDouble * 16, 16));
			__m128d * const data_array2 = static_cast<__m128d * const>(_mm_malloc(nBlocksDouble * 16, 16));
			
			const __m128d var1 = calc_variance<N_BITS>(mem_addr1, nBytes, data_array1);
			const __m128d var2 = calc_variance<N_BITS>(mem_addr2, nBytes, data_array2);
			const __m128d var1_2 = _mm_mul_pd(_mm_sqrt_pd(var1), _mm_sqrt_pd(var2));
			const __m128d corr = _mm_corr_dp_method3(data_array1, data_array2, nBytes, var1_2);

			_mm_free(data_array1);
			_mm_free(data_array2);
			return corr;
		}

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
			const size_t dataDoublesNBytes = resizeNBytes(8 * nBytes, 16);
			__m128d * const data_array1 = static_cast<__m128d * const>(_mm_malloc(dataDoublesNBytes, 16));
			__m128d * const data_array2 = static_cast<__m128d * const>(_mm_malloc(dataDoublesNBytes, 16));

			const size_t nElements = nBytes;
			const size_t swap_array_nBytes = nElements << 1;
			__m128i * const swap_array = static_cast<__m128i * const>(_mm_malloc(swap_array_nBytes, 16));

			const __m128d var1 = calc_variance<N_BITS>(mem_addr1, nBytes, data_array1);
			const __m128d var2 = calc_variance<N_BITS>(mem_addr2, nBytes, data_array2);
			const __m128d var1_2 = _mm_sqrt_pd(_mm_mul_pd(var1, var2));

			//std::cout << "INFO: _mm_corr_epu8::_mm_corr_perm_epu8_method3: var1=" << var1.m128d_f64[0] << "; var2=" << var2.m128d_f64[0] << std::endl;

			double * const results_double = reinterpret_cast<double * const>(results);
			for (size_t permutation = 0; permutation < nPermutations; ++permutation) {
				_mm_permute_dp_array(data_array2, dataDoublesNBytes, swap_array, swap_array_nBytes, randInts);
				const __m128d corr = _mm_corr_dp_method3(data_array1, data_array2, dataDoublesNBytes, var1_2);
				//std::cout << "INFO: _mm_corr_epu8::_mm_corr_perm_epu8_method3: corr=" << corr.m128d_f64[0] << std::endl;
				results_double[permutation] = corr.m128d_f64[0];
			}
			_mm_free(data_array1);
			_mm_free(data_array2);
			_mm_free(swap_array);
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
	inline __m128d _mm_corr_epu8(
		const __m128i * const mem_addr1,
		const __m128i * const mem_addr2,
		const size_t nBytes,
		const __m128d average1,
		const __m128d average2)
	{
		return priv::_mm_corr_epu8_method1<N_BITS>(mem_addr1, mem_addr2, nBytes, average1, average2);
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
		return priv::_mm_corr_perm_epu8_method1(mem_addr1, mem_addr2, nBytes, results, nPermutations, randInts);
	}

}