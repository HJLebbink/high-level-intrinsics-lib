#pragma once

#include <tuple>
#include <array>
#include <map>

//#include "mmintrin.h"  // mmx
#include "emmintrin.h"  // sse
#include "pmmintrin.h"  // sse3
#include "tmmintrin.h"  // ssse3
#include "smmintrin.h"  // sse4.1
#include "nmmintrin.h"  // sse4.2
#include "immintrin.h"  // for _mm_log2_pd
//#include "immintrin.h"  // avx, avx2, avx512, FP16C, KNCNI, FMA
//#include "ammintrin.h"  // AMD-specific intrinsics

#include "intrin.h"

#include "_mm_rand_si128.h"
#include "tools.h"
#include "timer.h"

namespace hli {

	namespace priv {

		inline __m128d _mm_log2_pd(__m128d a) {
			return _mm_set_pd(log2(a.m128d_f64[1]), log2(a.m128d_f64[0]));
		}

		inline __m128d freq_2bits_to_entropy_ref(
			const __m128i freq)
		{
			const __m128i nValues0 = _mm_shuffle_epi32(freq, 0b00000000);
			const __m128i nValues1 = _mm_shuffle_epi32(freq, 0b01010101);
			const __m128i nValues2 = _mm_shuffle_epi32(freq, 0b10101010);
			const __m128i nValues = _mm_add_epi32(nValues0, _mm_add_epi32(nValues1, nValues2));
			const __m128d nValuesDP = _mm_cvtepi32_pd(nValues);

			const __m128d prob1 = _mm_div_pd(_mm_cvtepi32_pd(freq), nValuesDP);
			const __m128d prob2 = _mm_div_pd(_mm_cvtepi32_pd(_mm_swap_64(freq)), nValuesDP);

			const __m128d sum_a = _mm_mul_pd(prob1, _mm_log2_pd(prob1));
			const __m128d sum_b = _mm_mul_pd(prob2, _mm_log2_pd(prob2));

			const __m128d sum_c = _mm_add_pd(sum_a, _mm_blend_pd(sum_b, _mm_setzero_pd(), 1));
			const __m128d sum_d = _mm_hadd_pd(sum_c, sum_c);
			return sum_d;
			/*
			const unsigned int nRowsNoMissingValues = nRows - nMissingValues;
			double sum = 0;
			for (size_t i = 0; i < 2; ++i) {
				const auto f = freq.get(i);
				if (f > 0) {
					const double probability = static_cast<double>(f) / nRowsNoMissingValues;
					sum += (probability * log2(probability));
				}
			}
			return (sum == 0) ? 0.0 : -sum;
			*/
		}

		// 
		template <int N_BITS>
		inline __m128i _mm_freq_epu8(
			const std::tuple<const __m128i * const, const size_t>& data)
		{
			switch (N_BITS) {
				case 2: return _mm_freq_epu8_nBits2(data)
				default: return _mm_setzero_si128();
			}
		}

		inline __m128i _mm_freq_epu8_nBits2_method1(
			const std::tuple<const __m128i * const, const size_t>& data)
		{
			const __m128i * const ptr = std::get<0>(data);

			const __m128i mask0 = _mm_setzero_si128();
			const __m128i mask1 = _mm_set1_epi8(1);
			const __m128i mask2 = _mm_set1_epi8(2);
			
			__m128i freq = _mm_setzero_si128();

			const size_t nBlocks = std::get<1>(data) >> 4;
			size_t block;

			for (block = 0; block < nBlocks - 4; block += 4) {

				const __m128i d0 = ptr[block + 0];
				const __m128i d1 = ptr[block + 1];
				const __m128i d2 = ptr[block + 2];
				const __m128i d3 = ptr[block + 3];

				const int nBits0 = static_cast<int>(__popcnt64(
					(_mm_movemask_epi8(_mm_cmpeq_epi8(d0, mask0))) ||
					(_mm_movemask_epi8(_mm_cmpeq_epi8(d1, mask0)) << 2) ||
					(_mm_movemask_epi8(_mm_cmpeq_epi8(d2, mask0)) << 4) ||
					(_mm_movemask_epi8(_mm_cmpeq_epi8(d3, mask0)) << 5)));

				const int nBits1 = static_cast<int>(__popcnt64(
					(_mm_movemask_epi8(_mm_cmpeq_epi8(d0, mask1))) ||
					(_mm_movemask_epi8(_mm_cmpeq_epi8(d1, mask1)) << 2) ||
					(_mm_movemask_epi8(_mm_cmpeq_epi8(d2, mask1)) << 4) ||
					(_mm_movemask_epi8(_mm_cmpeq_epi8(d3, mask1)) << 5)));

				const int nBits2 = static_cast<int>(__popcnt64(
					(_mm_movemask_epi8(_mm_cmpeq_epi8(d0, mask2))) ||
					(_mm_movemask_epi8(_mm_cmpeq_epi8(d1, mask2)) << 2) ||
					(_mm_movemask_epi8(_mm_cmpeq_epi8(d2, mask2)) << 4) ||
					(_mm_movemask_epi8(_mm_cmpeq_epi8(d3, mask2)) << 6)));

				freq = _mm_add_epi32(freq, _mm_set_epi32(nBits0, nBits1, nBits2, 0));
			}

			for (; block < nBlocks; ++block) {
				const __m128i d0 = ptr[block];
				const int nBits0 = static_cast<int>(__popcnt64(_mm_movemask_epi8(_mm_cmpeq_epi8(d0, mask0))));
				const int nBits1 = static_cast<int>(__popcnt64(_mm_movemask_epi8(_mm_cmpeq_epi8(d0, mask1))));
				const int nBits2 = static_cast<int>(__popcnt64(_mm_movemask_epi8(_mm_cmpeq_epi8(d0, mask2))));
				freq = _mm_add_epi32(freq, _mm_set_epi32(nBits0, nBits1, nBits2, 0));
			}

			return freq;
		}

		inline __m128i _mm_freq_epu8_nBits2_method2(
			const std::tuple<const __m128i * const, const size_t>& data)
		{
			const __m128i * const ptr = std::get<0>(data);
			
			const __m128i mask_0_epu8 = _mm_setzero_si128();
			const __m128i mask_1_epu8 = _mm_set1_epi8(1);
			const __m128i mask_2_epu8 = _mm_set1_epi8(2);

			__m128i freq = _mm_setzero_si128();
			__m128i freq0 = _mm_setzero_si128();
			__m128i freq1 = _mm_setzero_si128();
			__m128i freq2 = _mm_setzero_si128();

			const size_t nBlocks = std::get<1>(data) >> 4;
			const size_t nLoops = nBlocks >> 6; //divide by 2^6=64

			for (size_t block = 0; block < nBlocks; ++block) {
				const __m128i d0 = ptr[block + 0];
				freq0 = _mm_add_epi8(freq0, _mm_and_si128(_mm_cmpeq_epi8(d0, mask_0_epu8), mask_1_epu8));
				freq1 = _mm_add_epi8(freq1, _mm_and_si128(_mm_cmpeq_epi8(d0, mask_1_epu8), mask_1_epu8));
				freq2 = _mm_add_epi8(freq2, _mm_and_si128(_mm_cmpeq_epi8(d0, mask_2_epu8), mask_1_epu8));

				if ((block & 0x80) == 0) {
					freq0 = _mm_sad_epu8(freq0, mask_0_epu8);
					freq = _mm_add_epi32(freq, _mm_shuffle_epi32(freq0, _MM_SHUFFLE_EPI32_INT(2,3,3,3)));
					freq = _mm_add_epi32(freq, _mm_shuffle_epi32(freq0, _MM_SHUFFLE_EPI32_INT(0,3,3,3)));
					freq0 = _mm_setzero_si128();
				}
			}
			return freq;
		}

		inline __m128i _mm_freq_epu8_nBits2_method3(
			const std::tuple<const __m128i * const, const size_t>& data)
		{
			const __m128i * const ptr = std::get<0>(data);
			

			__m128i freq = _mm_setzero_si128();

			const size_t nBlocks = std::get<1>(data) >> 4;
			size_t block;

			for (block = 0; block < nBlocks - 4; block += 4) 
			{
				const __m128i d0 = ptr[block + 0];
				const __m128i d1 = ptr[block + 1];
				const __m128i d2 = ptr[block + 2];
				const __m128i d3 = ptr[block + 3];


				const __int64 bit0 =
					(_mm_movemask_epi8(_mm_slli_si128(d0, 6)) ||
					(_mm_movemask_epi8(_mm_slli_si128(d1, 6)) << 2) ||
					(_mm_movemask_epi8(_mm_slli_si128(d2, 6)) << 4) ||
					(_mm_movemask_epi8(_mm_slli_si128(d3, 6)) << 6));
				
				const __int64 bit1 =
					(_mm_movemask_epi8(_mm_slli_si128(d0, 7)) ||
					(_mm_movemask_epi8(_mm_slli_si128(d1, 7)) << 2) ||
					(_mm_movemask_epi8(_mm_slli_si128(d2, 7)) << 4) ||
					(_mm_movemask_epi8(_mm_slli_si128(d3, 7)) << 6));
				
				/*
				const __m256i de0, de1;
				const __int64 bit0e =
					(_mm256_movemask_epi8(_mm256_slli_si256(de0, 7)) ||
					(_mm256_movemask_epi8(_mm256_slli_si256(de1, 7)) << (1 * 32)));
				*/



			}
			return _mm_setzero_si128();
		}


		template <int N_BITS>
		inline __m128d _mm_entropy_epu8_ref(
			const std::tuple<const __m128i * const, const size_t>& data,
			const size_t nElements)
		{
			const __int8 * const ptr1 = reinterpret_cast<const __int8 * const>(std::get<0>(data));

			std::map<__int8, int> freq;
			size_t nMissingValues = 0;

			for (size_t element = 0; element < nElements; ++element) {
				if (ptr1[element] == 0xFF) {
					nMissingValues++;
				} else {
					const __int8 mergedData = ptr1[element];
					if (freq.count(mergedData) == 1) {
						freq[mergedData]++;
					} else {
						freq[mergedData] = 1;
					}
				}
			}
			size_t nValues_noMissing = nElements - nMissingValues;

			double h = 0;

			for (const auto &iter : freq) {
				double prob = (static_cast<double>(iter.second) / nValues_noMissing);
				h += prob * log2(prob);
			}

			return _mm_set1_pd(-h);
		}

		inline __m128d _mm_entropy_epu8_ref(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const std::tuple<const __m128i * const, const size_t>& data2,
			const size_t nElements)
		{
			const __int8 * const ptr1 = reinterpret_cast<const __int8 * const>(std::get<0>(data1));
			const __int8 * const ptr2 = reinterpret_cast<const __int8 * const>(std::get<0>(data2));

			std::map<__int16, int> freq;
			size_t nMissingValues = 0;

			for (size_t element = 0; element < nElements; ++element) {
				if ((ptr1[element] == 0xFF) || (ptr2[element] == 0xFF)) {
					nMissingValues++;
				} else {
					const __int16 mergedData = ptr1[element] | (static_cast<__int16>(ptr2[element]) << 8);
					if (freq.count(mergedData) == 1) {
						freq[mergedData]++;
					} else {
						freq[mergedData] = 1;
					}
				}
			}
			size_t nValues_noMissing = nElements - nMissingValues;

			double h = 0;

			for (const auto &iter : freq) {
				double prob = (static_cast<double>(iter.second) / nValues_noMissing);
				h += prob * log2(prob);
			}

			return _mm_set1_pd(-h);
		}

		template <int N_BITS>
		inline __m128d _mm_entropy_epu8_method0(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const std::tuple<const __m128i * const, const size_t>& data2,
			const size_t nElements)
		{
			return _mm_entropy_epu8_ref(data1, data2, nElements);
		}

		template <int N_BITS>
		inline __m128d _mm_entropy_epu8_method1(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const std::tuple<const __m128i * const, const size_t>& data2,
			const size_t nElements)
		{
			return _mm_entropy_epu8_ref(data1, data2, nElements);
		}
	}

	namespace test
	{
		void test_mm_entropy_epu8(const size_t nBlocks, const size_t nExperiments, const bool doTests)
		{
			const double delta = 0.0000001;

			const size_t nElements = 16 * nBlocks;


			auto data1 = _mm_malloc_m128i(nElements);
			auto data2 = _mm_malloc_m128i(nElements);
			fillRand_epu8<2>(data1);
			fillRand_epu8<2>(data2);

			double min_ref = std::numeric_limits<double>::max();
			double min1 = std::numeric_limits<double>::max();
			double min2 = std::numeric_limits<double>::max();
			double min3 = std::numeric_limits<double>::max();
			double min4 = std::numeric_limits<double>::max();

			__m128d result_ref, result1, result2, result3, result4;

			for (size_t i = 0; i < nExperiments; ++i) {

				timer::reset_and_start_timer();
				result_ref = hli::priv::_mm_entropy_epu8_ref(data1, data2, nElements);
				min_ref = std::min(min_ref, timer::get_elapsed_kcycles());

				{
					timer::reset_and_start_timer();
					result1 = hli::priv::_mm_entropy_epu8_method0<2>(data1, data2, nElements);
					min1 = std::min(min1, timer::get_elapsed_kcycles());

					if (doTests) {
						if (std::abs(result_ref.m128d_f64[0] - result1.m128d_f64[0]) > delta) {
							std::cout << "WARNING: test _mm_entropy_epu8_method0<2>: result-ref=" << hli::toString_f64(result_ref) << "; result=" << hli::toString_f64(result1) << std::endl;
							return;
						}
					}
				}
				{
					timer::reset_and_start_timer();
					result2 = hli::priv::_mm_entropy_epu8_method0<3>(data1, data2, nElements);
					min2 = std::min(min2, timer::get_elapsed_kcycles());

					if (doTests) {
						if (std::abs(result_ref.m128d_f64[0] - result2.m128d_f64[0]) > delta) {
							std::cout << "WARNING: test _mm_entropy_epu8_method0<3>: result-ref=" << hli::toString_f64(result_ref) << "; result=" << hli::toString_f64(result2) << std::endl;
							return;
						}
					}
				}

				{
					timer::reset_and_start_timer();
					result3 = hli::priv::_mm_entropy_epu8_method1<2>(data1, data2, nElements);
					min3 = std::min(min3, timer::get_elapsed_kcycles());

					if (doTests) {
						if (std::abs(result_ref.m128d_f64[0] - result3.m128d_f64[0]) > delta) {
							std::cout << "WARNING: test _mm_entropy_epu8_method1<2>: result-ref=" << hli::toString_f64(result_ref) << "; result=" << hli::toString_f64(result3) << std::endl;
							return;
						}
					}
				}
				{
					timer::reset_and_start_timer();
					result4 = hli::priv::_mm_entropy_epu8_method1<3>(data1, data2, nElements);
					min4 = std::min(min4, timer::get_elapsed_kcycles());

					if (doTests) {
						if (std::abs(result_ref.m128d_f64[0] - result4.m128d_f64[0]) > delta) {
							std::cout << "WARNING: test _mm_entropy_epu8_method1<3>: result-ref=" << hli::toString_f64(result_ref) << "; result=" << hli::toString_f64(result4) << std::endl;
							return;
						}
					}
				}

			}
			printf("[_mm_entropy_epu8_ref<2>]    : %2.5f Kcycles; %0.14f\n", min_ref, result_ref.m128d_f64[0]);
			printf("[_mm_entropy_epu8_method0<2>]: %2.5f Kcycles; %0.14f; %2.3f times faster than ref\n", min1, result1.m128d_f64[0], min_ref / min1);
			printf("[_mm_entropy_epu8_method0<3>]: %2.5f Kcycles; %0.14f; %2.3f times faster than ref\n", min2, result2.m128d_f64[0], min_ref / min2);
			printf("[_mm_entropy_epu8_method1<2>]: %2.5f Kcycles; %0.14f; %2.3f times faster than ref\n", min3, result3.m128d_f64[0], min_ref / min3);
			printf("[_mm_entropy_epu8_method1<3>]: %2.5f Kcycles; %0.14f; %2.3f times faster than ref\n", min4, result4.m128d_f64[0], min_ref / min4);
		
			_mm_free2(data1);
			_mm_free2(data2);
		}
	}
	
	template <int N_BITS>
	inline __m128d _mm_entropy_epu8(
		const std::tuple<const __m128i * const, const size_t>& data,
		const size_t nElements)
	{
		return priv::_mm_entropy_epu8_ref<N_BITS>(data, nElements);
	}

	template <int N_BITS>
	inline __m128d _mm_entropy_epu8(
		const std::tuple<const __m128i * const, const size_t>& data1,
		const std::tuple<const __m128i * const, const size_t>& data2,
		const size_t nElements)
	{
		return priv::_mm_entropy_epu8_ref(data1, data2, nElements);
	}

}