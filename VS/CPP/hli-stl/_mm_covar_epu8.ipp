#pragma once

#include <algorithm>	// std::min
#include <limits>		// std::numeric_limits
#include <iostream>		// std::cout


//#include "mmintrin.h"  // mmx
#include "emmintrin.h"  // sse
#include "pmmintrin.h"  // sse3
#include "tmmintrin.h"  // ssse3
#include "smmintrin.h"  // sse4.1
#include "nmmintrin.h"  // sse4.2
//#include "immintrin.h"  // avx, avx2, avx512, FP16C, KNCNI, FMA
//#include "ammintrin.h"  // AMD-specific intrinsics

#include "_mm_hadd_epu8.ipp"

namespace hli
{

	namespace priv
	{

		// Variance population SSE: return 2x double var
		template <bool HAS_MV, U8 MV>
		inline __m128d _mm_covar_epu8_method0(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const std::tuple<const __m128i * const, const size_t>& data2,
			const size_t nElements,
			const __m128d average1,
			const __m128d average2)
		{
			const double average1_D = average1.m128d_f64[0];
			const double average2_D = average2.m128d_f64[0];

			auto ptr1 = reinterpret_cast<const U8 * const>(std::get<0>(data1));
			auto ptr2 = reinterpret_cast<const U8 * const>(std::get<0>(data2));

			double sum = 0;

			if constexpr (HAS_MV)
			{
				size_t nElements_No_MV = 0;
				for (size_t i = 0; i < nElements; ++i)
				{
					const U8 d1 = ptr1[i];
					const U8 d2 = ptr2[i];

					if ((d1 != MV) && (d2 != MV))
					{
						nElements_No_MV++;
						sum += ((static_cast<double>(d1) - average1_D) * (static_cast<double>(d2) - average2_D));
					}
				}
				return _mm_set1_pd(sum / nElements_No_MV);
			}
			else
			{
				for (size_t i = 0; i < nElements; ++i)
				{
					const double d1 = ptr1[i];
					const double d2 = ptr2[i];
					sum += ((static_cast<double>(d1) - average1_D) * (static_cast<double>(d2) - average2_D));
				}
				return _mm_set1_pd(sum / nElements);
			}
		}

		// Variance population reference
		template <int N_BITS1, int N_BITS2, bool HAS_MV, U8 MV>
		inline __m128d _mm_covar_epu8_method0(
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

				// copy the missing value to both data columns, this to make the separate entropy calculation correct.
				for (int i = 0; i < nElements; ++i)
				{
					if ((data1b_ptr[i] == MV) || (data2b_ptr[i] == MV))
					{
						data1b_ptr[i] = MV;
						data2b_ptr[i] = MV;
					}
				}

				const auto tup1 = _mm_hadd_epu8<N_BITS1, HAS_MV, MV>(data1b, nElements);
				const __m128d sum1 = _mm_cvtepi32_pd(std::get<0>(tup1));
				const __m128d nElements1 = _mm_cvtepi32_pd(std::get<1>(tup1));
				const __m128d average1 = _mm_div_pd(sum1, nElements1);

				const auto tup2 = _mm_hadd_epu8<N_BITS2, HAS_MV, MV>(data2b, nElements);
				const __m128d sum2 = _mm_cvtepi32_pd(std::get<0>(tup2));
				const __m128d nElements2 = _mm_cvtepi32_pd(std::get<1>(tup2));
				const __m128d average2 = _mm_div_pd(sum2, nElements2);

				const __m128d covar = priv::_mm_covar_epu8_method0<HAS_MV, MV>(data1, data2, nElements, average1, average2);

				_mm_free2(data1b);
				_mm_free2(data2b);

				return covar;
			}
			else
			{
				const auto tup1 = _mm_hadd_epu8<N_BITS1, HAS_MV, MV>(data1, nElements);
				const __m128d sum1 = _mm_cvtepi32_pd(std::get<0>(tup1));
				const __m128d nElements1 = _mm_cvtepi32_pd(std::get<1>(tup1));
				const __m128d average1 = _mm_div_pd(sum1, nElements1);

				const auto tup2 = _mm_hadd_epu8<N_BITS2, HAS_MV, MV>(data2, nElements);
				const __m128d sum2 = _mm_cvtepi32_pd(std::get<0>(tup2));
				const __m128d nElements2 = _mm_cvtepi32_pd(std::get<1>(tup2));
				const __m128d average2 = _mm_div_pd(sum2, nElements2);

				return priv::_mm_covar_epu8_method0<HAS_MV, MV>(data1, data2, nElements, average1, average2);
			}
		}

		// Variance population SSE: return 2x double var
		template <bool HAS_MV, U8 MV>
		inline __m128d _mm_covar_epu8_method1(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const std::tuple<const __m128i * const, const size_t>& data2,
			const size_t nElements,
			const __m128d average1,
			const __m128d average2)
		{
			if constexpr (HAS_MV) //TODO
			{
				std::cout << "WARNING: _mm_covar_epu8_method1: Not implemented yet" << std::endl;
				return _mm_setzero_pd();
			}
			else
			{
				__m128d result_a = _mm_setzero_pd();
				__m128d result_b = _mm_setzero_pd();

				const size_t nBytes = std::get<1>(data1);
				const size_t nBlocks = nBytes >> 4;
				for (size_t block = 0; block < nBlocks; ++block)
				{
					const __m128i data1_Block = std::get<0>(data1)[block];
					const __m128i data2_Block = std::get<0>(data2)[block];
					{
						const __m128i d1 = _mm_cvtepu8_epi32(data1_Block);
						const __m128i d2 = _mm_cvtepu8_epi32(data2_Block);
						const __m128d d1a = _mm_sub_pd(_mm_cvtepi32_pd(d1), average1);
						const __m128d d2a = _mm_sub_pd(_mm_cvtepi32_pd(d2), average2);
						const __m128d d1b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d1, 0b01011011)), average1);
						const __m128d d2b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d2, 0b01011011)), average2);
						result_a = _mm_add_pd(result_a, _mm_mul_pd(d1a, d2a));
						result_b = _mm_add_pd(result_b, _mm_mul_pd(d1b, d2b));
					}
					{
						const __m128i d1 = _mm_cvtepu8_epi32(_mm_shuffle_epi32(data1_Block, 0b01010101));
						const __m128i d2 = _mm_cvtepu8_epi32(_mm_shuffle_epi32(data2_Block, 0b01010101));
						const __m128d d1a = _mm_sub_pd(_mm_cvtepi32_pd(d1), average1);
						const __m128d d2a = _mm_sub_pd(_mm_cvtepi32_pd(d2), average2);
						const __m128d d1b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d1, 0b01011011)), average1);
						const __m128d d2b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d2, 0b01011011)), average2);
						result_a = _mm_add_pd(result_a, _mm_mul_pd(d1a, d2a));
						result_b = _mm_add_pd(result_b, _mm_mul_pd(d1b, d2b));
					}
					{
						const __m128i d1 = _mm_cvtepu8_epi32(_mm_shuffle_epi32(data1_Block, 0b10101010));
						const __m128i d2 = _mm_cvtepu8_epi32(_mm_shuffle_epi32(data2_Block, 0b10101010));
						const __m128d d1a = _mm_sub_pd(_mm_cvtepi32_pd(d1), average1);
						const __m128d d2a = _mm_sub_pd(_mm_cvtepi32_pd(d2), average2);
						const __m128d d1b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d1, 0b01011011)), average1);
						const __m128d d2b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d2, 0b01011011)), average2);
						result_a = _mm_add_pd(result_a, _mm_mul_pd(d1a, d2a));
						result_b = _mm_add_pd(result_b, _mm_mul_pd(d1b, d2b));
					}
					{
						const __m128i d1 = _mm_cvtepu8_epi32(_mm_shuffle_epi32(data1_Block, 0b11111111));
						const __m128i d2 = _mm_cvtepu8_epi32(_mm_shuffle_epi32(data2_Block, 0b11111111));
						const __m128d d1a = _mm_sub_pd(_mm_cvtepi32_pd(d1), average1);
						const __m128d d2a = _mm_sub_pd(_mm_cvtepi32_pd(d2), average2);
						const __m128d d1b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d1, 0b01011011)), average1);
						const __m128d d2b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d2, 0b01011011)), average2);
						result_a = _mm_add_pd(result_a, _mm_mul_pd(d1a, d2a));
						result_b = _mm_add_pd(result_b, _mm_mul_pd(d1b, d2b));
					}
					//std::cout << "INFO: hli:::_mm_variance_epu8: data=" << toString_u8(data) << "; d1=" << toString_u32(d1) << "; d2=" << toString_u32(d2) << "; d3=" << toString_u32(d3) << "; d4=" << toString_u32(d4) << std::endl;
				}
				result_a = _mm_add_pd(result_a, result_b);
				const __m128d nElementsD = _mm_set1_pd(static_cast<double>(nElements));
				return _mm_div_pd(_mm_hadd_pd(result_a, result_a), nElementsD);
			}
		}

		template <int N_BITS1, int N_BITS2, bool HAS_MV, U8 MV>
		inline __m128d _mm_covar_epu8_method1(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const std::tuple<const __m128i * const, const size_t>& data2,
			const size_t nElements)
		{
			if constexpr (HAS_MV)
			{
				//TODO
				std::cout << "WARNING: _mm_covar_epu8_method1: Not implemented yet" << std::endl;
				return _mm_setzero_pd();
			}
			else
			{
				const auto tup1 = _mm_hadd_epu8<N_BITS1, HAS_MV, MV>(data1, nElements);
				const __m128d sum1 = _mm_cvtepi32_pd(std::get<0>(tup1));
				const __m128d nElements1 = _mm_cvtepi32_pd(std::get<1>(tup1));
				const __m128d average1 = _mm_div_pd(sum1, nElements1);

				const auto tup2 = _mm_hadd_epu8<N_BITS2, HAS_MV, MV>(data2, nElements);
				const __m128d sum2 = _mm_cvtepi32_pd(std::get<0>(tup2));
				const __m128d nElements2 = _mm_cvtepi32_pd(std::get<1>(tup2));
				const __m128d average2 = _mm_div_pd(sum2, nElements2);

				return priv::_mm_covar_epu8_method1<HAS_MV, MV>(data1, data2, nBytes, average1, average2);
			}
		}
	}

	// Variance population SSE: return 2x double var
	template <bool HAS_MV, U8 MV>
	inline __m128d _mm_covar_epu8(
		const std::tuple<const __m128i * const, const size_t>& data1,
		const std::tuple<const __m128i * const, const size_t>& data2,
		const size_t nElements,
		const __m128d average1,
		const __m128d average2)
	{
		//return priv::_mm_covar_epu8_method0<HAS_MV, MV>(data1, data2, nElements, average1, average2);
		return priv::_mm_covar_epu8_method1<HAS_MV, MV>(data1, data2, nElements, average1, average2);
	}

	template <int N_BITS1, int N_BITS2, bool HAS_MV, U8 MV>
	inline __m128d _mm_covar_epu8(
		const std::tuple<const __m128i * const, const size_t>& data1,
		const std::tuple<const __m128i * const, const size_t>& data2,
		const size_t nElements)
	{
		//return priv::_mm_covar_epu8_method0<N_BITS1, N_BITS2, HAS_MV, MV>(data1, data2, nElements);
		return priv::_mm_covar_epu8_method1<N_BITS1, N_BITS2, HAS_MV, MV>(data1, data2, nElements);
	}
}