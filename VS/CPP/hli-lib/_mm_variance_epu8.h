#pragma once

//#include "mmintrin.h"  // mmx
#include "emmintrin.h"  // sse
#include "pmmintrin.h"  // sse3
#include "tmmintrin.h"  // ssse3
#include "smmintrin.h"  // sse4.1
#include "nmmintrin.h"  // sse4.2
//#include "immintrin.h"  // avx, avx2, avx512, FP16C, KNCNI, FMA
//#include "ammintrin.h"  // AMD-specific intrinsics

#include "_mm_hadd_epu8.h"

namespace hli {

	namespace priv {

		// Variance population reference
		inline __m128d _mm_variance_epu8_method0(
			const std::tuple<const __m128i * const, const size_t>& data)
		{
			const size_t nBytes = std::get<1>(data);
			const double average = static_cast<double>(_mm_hadd_epu8_method0(data).m128i_u32[0]) / nBytes;
			
			const unsigned __int8 * const ptr = reinterpret_cast<const unsigned __int8 * const>(std::get<0>(data));
			double sum = 0;
			for (size_t i = 0; i < nBytes; ++i) {
				double d = static_cast<double>(ptr[i]) - average;
				sum += (d * d);
			}
			return _mm_set1_pd(sum / nBytes);
		}

		// Variance population SSE: return 2x double var
		template <int N_BITS>
		inline __m128d _mm_variance_epu8_method1(
			const std::tuple<const __m128i * const, const size_t>& data)
		{
			const size_t nBytes = std::get<1>(data);
			const __m128d sum = _mm_cvtepi32_pd(_mm_hadd_epu8<N_BITS>(data));
			const __m128d nElements = _mm_set1_pd(static_cast<double>(nBytes));
			const __m128d average = _mm_div_pd(sum, nElements);
			return _mm_variance_epu8_method1(data, average);
		}

		// Variance population SSE: return 2x double var
		inline __m128d _mm_variance_epu8_method1(
			const std::tuple<const __m128i * const, const size_t>& data,
			const __m128d average)
		{
			const size_t nBytes = std::get<1>(data);

			__m128d result_a = _mm_setzero_pd();
			__m128d result_b = _mm_setzero_pd();

			const size_t nBlocks = nBytes >> 4;
			for (size_t block = 0; block < nBlocks; ++block) {
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
	}

	namespace test {

		void test_mm_variance_epu8(const size_t nBlocks, const size_t nExperiments, const bool doTests)
		{
			const double delta = 0.0000001;
			auto data = _mm_malloc_m128i(16 * nBlocks);
			fillRand_epu8<5>(data);

			{
				double min_ref = std::numeric_limits<double>::max();
				double min1 = std::numeric_limits<double>::max();
				double min2 = std::numeric_limits<double>::max();
				double min3 = std::numeric_limits<double>::max();
				double min4 = std::numeric_limits<double>::max();

				for (size_t i = 0; i < nExperiments; ++i) {

					timer::reset_and_start_timer();
					const __m128d result_ref = hli::priv::_mm_variance_epu8_method0(data);
					min_ref = std::min(min_ref, timer::get_elapsed_kcycles());

					{
						timer::reset_and_start_timer();
						const __m128d result = hli::priv::_mm_variance_epu8_method1<8>(data);
						min1 = std::min(min1, timer::get_elapsed_kcycles());

						if (doTests) {
							if (std::abs(result_ref.m128d_f64[0] - result.m128d_f64[0]) > delta) {
								std::cout << "WARNING: test _mm_variance_epu8<8>: result-ref=" << hli::toString_f64(result_ref) << "; result=" << hli::toString_f64(result) << std::endl;
								return;
							}
						}
					}
					{
						timer::reset_and_start_timer();
						const __m128d result = hli::priv::_mm_variance_epu8_method1<7>(data);
						min2 = std::min(min2, timer::get_elapsed_kcycles());

						if (doTests) {
							if (std::abs(result_ref.m128d_f64[0] - result.m128d_f64[0]) > delta) {
								std::cout << "WARNING: test _mm_variance_epu8<7>: result-ref=" << hli::toString_f64(result_ref) << "; result=" << hli::toString_f64(result) << std::endl;
								return;
							}
						}
					}
					{
						timer::reset_and_start_timer();
						const __m128d result = hli::priv::_mm_variance_epu8_method1<6>(data);
						min3 = std::min(min3, timer::get_elapsed_kcycles());

						if (doTests) {
							if (std::abs(result_ref.m128d_f64[0] - result.m128d_f64[0]) > delta) {
								std::cout << "WARNING: test _mm_variance_epu8<6>: result-ref=" << hli::toString_f64(result_ref) << "; result=" << hli::toString_f64(result) << std::endl;
								return;
							}
						}
					}
					{
						timer::reset_and_start_timer();
						const __m128d result = hli::priv::_mm_variance_epu8_method1<5>(data);
						min4 = std::min(min4, timer::get_elapsed_kcycles());

						if (doTests) {
							if (std::abs(result_ref.m128d_f64[0] - result.m128d_f64[0]) > delta) {
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
	inline __m128d _mm_variance_epu8(
		const std::tuple<const __m128i * const, const size_t>& data,
		const __m128d average)
	{
		return priv::_mm_variance_epu8_method1(data, average);
	}

	// Variance population SSE: return 2x double var
	template <int N_BITS>
	inline __m128d _mm_variance_epu8(
		const std::tuple<const __m128i * const, const size_t>& data)
	{
		return priv::_mm_variance_epu8_method1<N_BITS>(data);
	}
}