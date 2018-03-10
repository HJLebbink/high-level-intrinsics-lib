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
		// Variance population reference
		template <bool HAS_MV, U8 MV>
		inline __m128d _mm_variance_epu8_method0(
			const std::tuple<const __m128i * const, const int>& data,
			const int nElements)
		{
			const auto tup = _mm_hadd_epu8_method0<HAS_MV, MV>(data, nElements);
			unsigned int nElements_No_MV = static_cast<unsigned int>(_mm_cvtsi128_si32(std::get<1>(tup)));

			const double average = static_cast<double>(std::get<0>(tup).m128i_u32[0]) / nElements_No_MV;

			auto ptr = reinterpret_cast<const U8 * const>(std::get<0>(data));
			double sum = 0;

			if (HAS_MV)
			{
				for (int i = 0; i < nElements; ++i)
				{
					const U8 d = ptr[i];
					if (d != MV)
					{
						double tmp = static_cast<double>(d) - average;
						sum += (tmp * tmp);
					}
				}
			}
			else
			{
				for (int i = 0; i < nElements; ++i)
				{
					double tmp = static_cast<double>(ptr[i]) - average;
					sum += (tmp * tmp);
				}
			}
			return _mm_set1_pd(sum / nElements_No_MV);
		}

		// Variance population SSE: return 2x double var
		template <int N_BITS, bool HAS_MV, U8 MV>
		inline __m128d _mm_variance_epu8_method1(
			const std::tuple<const __m128i * const, const int>& data,
			const int nElements)
		{
			if (HAS_MV) //TODO
			{ 
				std::cout << "WARNING: _mm_variance_epu8_method1: Not implemented yet" << std::endl;
				return _mm_setzero_pd();
			}

			const auto tup = _mm_hadd_epu8<N_BITS, HAS_MV, MV>(data, nElements);
			const __m128d sum = _mm_cvtepi32_pd(std::get<0>(tup));
			const __m128d nTrueElements = _mm_cvtepi32_pd(std::get<1>(tup));
			const __m128d average = _mm_div_pd(sum, nTrueElements);
			return _mm_variance_epu8_method1<HAS_MV, MV>(data, average);
		}

		// Variance population SSE: return 2x double var
		template <bool HAS_MV, U8 MV>
		inline __m128d _mm_variance_epu8_method1(
			const std::tuple<const __m128i * const, const int>& data,
			const __m128d average)
		{
			if (HAS_MV)
			{ //TODO
				std::cout << "WARNING: _mm_variance_epu8_method1: Not implemented yet" << std::endl;
				return _mm_setzero_pd();
			}

			const int nBytes = std::get<1>(data);

			__m128d result_a = _mm_setzero_pd();
			__m128d result_b = _mm_setzero_pd();

			const int nBlocks = nBytes >> 4;
			for (int block = 0; block < nBlocks; ++block)
			{
				const __m128i data_Block = std::get<0>(data)[block];
				{
					const __m128i d1 = _mm_cvtepi8_epi32(data_Block);
					const __m128d d1a = _mm_sub_pd(_mm_cvtepi32_pd(d1), average);
					const __m128d d1b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d1, 0b01011011)), average);
					result_a = _mm_add_pd(result_a, _mm_mul_pd(d1a, d1a));
					result_b = _mm_add_pd(result_b, _mm_mul_pd(d1b, d1b));
				}
				{
					const __m128i d1 = _mm_cvtepi8_epi32(_mm_shuffle_epi32(data_Block, 0b01010101));
					const __m128d d1a = _mm_sub_pd(_mm_cvtepi32_pd(d1), average);
					const __m128d d1b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d1, 0b01011011)), average);
					result_a = _mm_add_pd(result_a, _mm_mul_pd(d1a, d1a));
					result_b = _mm_add_pd(result_b, _mm_mul_pd(d1b, d1b));
				}
				{
					const __m128i d1 = _mm_cvtepi8_epi32(_mm_shuffle_epi32(data_Block, 0b10101010));
					const __m128d d1a = _mm_sub_pd(_mm_cvtepi32_pd(d1), average);
					const __m128d d1b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d1, 0b01011011)), average);
					result_a = _mm_add_pd(result_a, _mm_mul_pd(d1a, d1a));
					result_b = _mm_add_pd(result_b, _mm_mul_pd(d1b, d1b));
				}
				{
					const __m128i d1 = _mm_cvtepi8_epi32(_mm_shuffle_epi32(data_Block, 0b11111111));
					const __m128d d1a = _mm_sub_pd(_mm_cvtepi32_pd(d1), average);
					const __m128d d1b = _mm_sub_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(d1, 0b01011011)), average);
					result_a = _mm_add_pd(result_a, _mm_mul_pd(d1a, d1a));
					result_b = _mm_add_pd(result_b, _mm_mul_pd(d1b, d1b));
				}
				//std::cout << "INFO: hli:::_mm_variance_epu8: data=" << toString_u8(data) << "; d1=" << toString_u32(d1) << "; d2=" << toString_u32(d2) << "; d3=" << toString_u32(d3) << "; d4=" << toString_u32(d4) << std::endl;
			}
			result_a = _mm_add_pd(result_a, result_b);
			const __m128d nElements = _mm_set1_pd(static_cast<double>(nBytes));
			return _mm_div_pd(_mm_hadd_pd(result_a, result_a), nElements);
		}

		template <int N_BITS, bool HAS_MV, U8 MV>
		inline __m128d _mm_variance_epu8_method1(
			const std::tuple<const __m128i * const, const int>& data1,
			const int nElements,
			const std::tuple<__m128d * const, const int>& data_double_out)
		{
			if (HAS_MV)
			{ //TODO
				std::cout << "WARNING: _mm_variance_epu8_method1: Not implemented yet" << std::endl;
				return _mm_setzero_pd();
			}

			const auto tup1 = _mm_hadd_epu8<N_BITS, HAS_MV, MV>(data1, nElements);
			const __m128d nTrueElements = _mm_cvtepi32_pd(std::get<1>(tup1));
			const __m128d average = _mm_div_pd(_mm_cvtepi32_pd(std::get<0>(tup1)), nTrueElements);

			const int nBytes = std::get<1>(data1);
			const int nBlocksData = nBytes >> 4;

			__m128d * const data_double = std::get<0>(data_double_out);
			__m128d var = _mm_setzero_pd();

			for (int block = 0; block < nBlocksData; ++block)
			{
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
			//for (int block = 0; block < (nBlocksData * 8); ++block) {
			//	std::cout << "INFO: _mm_corr_epu8::calc_variance: block=" << block << "; d=" << toString_f64(data_double[block]) << std::endl;
			//}
			return _mm_div_pd(_mm_hadd_pd(var, var), nTrueElements);
		}

	}

	namespace test
	{
		using namespace tools::timing;

		void _mm_variance_epu8_speed_test_1(const int nBlocks, const int nExperiments, const bool doTests)
		{
			const double delta = 0.0000001;
			const bool HAS_MV = false;
			const U8 MV = 0xFF;

			const int nElements = nBlocks * 16;
			auto data = _mm_malloc_m128i(nElements);
			fillRand_epu8<5>(data);

			{
				double min_ref = std::numeric_limits<double>::max();
				double min1 = std::numeric_limits<double>::max();
				double min2 = std::numeric_limits<double>::max();
				double min3 = std::numeric_limits<double>::max();
				double min4 = std::numeric_limits<double>::max();

				for (int i = 0; i < nExperiments; ++i)
				{

					reset_and_start_timer();
					const __m128d result_ref = hli::priv::_mm_variance_epu8_method0<HAS_MV, MV>(data, nElements);
					min_ref = std::min(min_ref, get_elapsed_kcycles());

					{
						reset_and_start_timer();
						const __m128d result = hli::priv::_mm_variance_epu8_method1<8, HAS_MV, MV>(data, nElements);
						min1 = std::min(min1, get_elapsed_kcycles());

						if (doTests)
						{
							if (std::abs(result_ref.m128d_f64[0] - result.m128d_f64[0]) > delta)
							{
								std::cout << "WARNING: test _mm_variance_epu8<8>: result-ref=" << hli::toString_f64(result_ref) << "; result=" << hli::toString_f64(result) << std::endl;
								return;
							}
						}
					}
					{
						reset_and_start_timer();
						const __m128d result = hli::priv::_mm_variance_epu8_method1<7, HAS_MV, MV>(data, nElements);
						min2 = std::min(min2, get_elapsed_kcycles());

						if (doTests)
						{
							if (std::abs(result_ref.m128d_f64[0] - result.m128d_f64[0]) > delta)
							{
								std::cout << "WARNING: test _mm_variance_epu8<7>: result-ref=" << hli::toString_f64(result_ref) << "; result=" << hli::toString_f64(result) << std::endl;
								return;
							}
						}
					}
					{
						reset_and_start_timer();
						const __m128d result = hli::priv::_mm_variance_epu8_method1<6, HAS_MV, MV>(data, nElements);
						min3 = std::min(min3, get_elapsed_kcycles());

						if (doTests)
						{
							if (std::abs(result_ref.m128d_f64[0] - result.m128d_f64[0]) > delta)
							{
								std::cout << "WARNING: test _mm_variance_epu8<6>: result-ref=" << hli::toString_f64(result_ref) << "; result=" << hli::toString_f64(result) << std::endl;
								return;
							}
						}
					}
					{
						reset_and_start_timer();
						const __m128d result = hli::priv::_mm_variance_epu8_method1<5, HAS_MV, MV>(data, nElements);
						min4 = std::min(min4, get_elapsed_kcycles());

						if (doTests)
						{
							if (std::abs(result_ref.m128d_f64[0] - result.m128d_f64[0]) > delta)
							{
								std::cout << "WARNING: test _mm_variance_epu8<5>: result-ref=" << hli::toString_f64(result_ref) << "; result=" << hli::toString_f64(result) << std::endl;
								return;
							}
						}
					}
				}
				printf("[_mm_variance_epu8_method0]   : %2.5f Kcycles\n", min_ref);
				printf("[_mm_variance_epu8_method1<8>]: %2.5f Kcycles; %2.3f times faster than ref\n", min1, min_ref / min1);
				printf("[_mm_variance_epu8_method1<7>]: %2.5f Kcycles; %2.3f times faster than ref\n", min2, min_ref / min2);
				printf("[_mm_variance_epu8_method1<6>]: %2.5f Kcycles; %2.3f times faster than ref\n", min3, min_ref / min3);
				printf("[_mm_variance_epu8_method1<5>]: %2.5f Kcycles; %2.3f times faster than ref\n", min4, min_ref / min4);
			}

			_mm_free2(data);
		}

	}

	// Variance population SSE: return 2x double var
	template <bool HAS_MV, U8 MV>
	inline __m128d _mm_variance_epu8(
		const std::tuple<const __m128i * const, const int>& data,
		const __m128d average)
	{
		return priv::_mm_variance_epu8_method1<HAS_MV, MV>(data, average);
	}

	// Variance population SSE: return 2x double var
	template <int N_BITS, bool HAS_MV, U8 MV>
	inline __m128d _mm_variance_epu8(
		const std::tuple<const __m128i * const, const int>& data)
	{
		return priv::_mm_variance_epu8_method1<N_BITS, HAS_MV, MV>(data);
	}
}