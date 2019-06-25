#pragma once

#include <algorithm>	// std::min
#include <limits>		// std::numeric_limits
#include <iostream>		// std::cout
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

#include "_mm_rand_si128.ipp"
#include "tools.ipp"
#include "timing.ipp"

namespace hli
{
	namespace priv
	{

		#ifdef _MSC_VER
		inline __m128d _mm_log2_pd(__m128d a)
		{
			return _mm_set_pd(log2(a.m128d_f64[1]), log2(a.m128d_f64[0]));
		}
		#endif


		template <int N_BITS1, int N_BITS2, bool HAS_MV, U8 MV>
		inline void merge_U8xU8_to_U8(
			const std::tuple<const __m128i * const, const int>& data1,
			const std::tuple<const __m128i * const, const int>& data2,
			const std::tuple<__m128i * const, const int>& data3)
		{
			#pragma region Tests
			static_assert(N_BITS1 > 0, "NBITS_1 has to be larger than zero");
			static_assert(N_BITS2 > 0, "NBITS_2 has to be larger than zero");
			static_assert((N_BITS1 + N_BITS2) <= 8, "NBITS_1 + N_BITS_2 has to be smaller than 9");

			#if _DEBUG
			if (std::get<1>(data1) != std::get<1>(data2)) std::cout << "WARNING: merge_U8xU8_to_U8: unequal length data1 and data2";
			if (std::get<1>(data1) != std::get<1>(data3)) std::cout << "WARNING: merge_U8xU8_to_U8: unequal length data1 and data3";
			#endif
			#pragma endregion

			const int nBlocks = std::get<1>(data1) >> 4;
			if (HAS_MV)
			{
				if (MV == 0xFF)
				{
					//TODO
				}
				else
				{
					for (int block = 0; block < nBlocks; ++block)
					{
						const __m128i d1 = std::get<0>(data1)[block];
						const __m128i d2 = std::get<0>(data2)[block];
						//TODO

						std::get<0>(data3)[block] = _mm_or_si128(d1, _mm_slli_epi16(d2, N_BITS1));
					}
				}
			}
			else
			{
				for (int block = 0; block < nBlocks; ++block)
				{
					std::get<0>(data3)[block] = _mm_or_si128(std::get<0>(data1)[block], _mm_slli_epi16(std::get<0>(data2)[block], N_BITS1));
				}
			}
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

			const __m128d sum_a = _mm_set1_pd(0); //_mm_mul_pd(prob1, _mm_log2_pd(prob1));
			const __m128d sum_b = _mm_set1_pd(0); //_mm_mul_pd(prob2, _mm_log2_pd(prob2));

			const __m128d sum_c = _mm_add_pd(sum_a, _mm_blend_pd(sum_b, _mm_setzero_pd(), 1));
			const __m128d sum_d = _mm_hadd_pd(sum_c, sum_c);
			return sum_d;
			/*
			const unsigned int nRowsNoMissingValues = nRows - nMissingValues;
			double sum = 0;
			for (int i = 0; i < 2; ++i) {
				const auto f = freq.get(i);
				if (f > 0) {
					const double probability = static_cast<double>(f) / nRowsNoMissingValues;
					sum += (probability * log2(probability));
				}
			}
			return (sum == 0) ? 0.0 : -sum;
			*/
		}

		template <int N_BITS, bool HAS_MV, U8 MV>
		inline __m128d _mm_entropy_epu8_method0(
			const std::tuple<const __m128i * const, const int>& data,
			const int nElements)
		{
			static_assert(N_BITS > 0, "_mm_entropy_epu8_method0: N_BITS should be larger than 0");
			static_assert(N_BITS <= 8, "_mm_entropy_epu8_method0: N_BITS should be smaller or equal than 8");

			const U8 * const ptr1 = reinterpret_cast<const U8 * const>(std::get<0>(data));

			const int N_DISTINCT_VALUES = (1 << N_BITS);
			std::array<int, N_DISTINCT_VALUES> freq;
			freq.fill(0);

			double h = 0;
			if (HAS_MV)
			{
				int nElements_No_MV = 0;
				for (int element = 0; element < nElements; ++element)
				{
					const U8 mergedData = ptr1[element];
					if (mergedData != MV)
					{
						freq[mergedData]++;
						nElements_No_MV++;
					}
				}
				if (nElements_No_MV > 0)
				{
					for (int i = 0; i < N_DISTINCT_VALUES; ++i)
					{
						//std::cout << "INFO: _mm_entropy_epu8_method0: freq[" << i <<"]=" << freq[i] << std::endl;
						if (freq[i] > 0)
						{
							double prob = (static_cast<double>(freq[i]) / nElements_No_MV);
							h += prob * log2(prob);
						}
					}
				}
			}
			else
			{
				for (int element = 0; element < nElements; ++element)
				{
					const U8 mergedData = ptr1[element];
					freq[mergedData]++;
				}
				for (int i = 0; i < N_DISTINCT_VALUES; ++i)
				{
					//std::cout << "INFO: _mm_entropy_epu8_method0: freq[" << i <<"]=" << freq[i] << std::endl;
					if (freq[i] > 0)
					{
						double prob = (static_cast<double>(freq[i]) / nElements);
						h += prob * log2(prob);
					}
				}
			}
			return _mm_set1_pd(-h);
		}

		template <int N_BITS, bool HAS_MV, U8 MV>
		inline __m128d _mm_entropy_epu8_method1(
			const std::tuple<const __m128i * const, const int>& data,
			const int nElements)
		{
			static_assert(N_BITS > 0, "");
			static_assert(N_BITS <= 8, "");

			const int N_DISTINCT_VALUES = (1 << N_BITS);
			std::array<int, N_DISTINCT_VALUES> freq;
			freq.fill(0);

			const int nBlocks = std::get<1>(data) >> 4;

			for (int block = 0; block < nBlocks; ++block)
			{
				const __m128i d = std::get<0>(data)[block];
				for (int i = 0; i < N_DISTINCT_VALUES; ++i)
				{
					freq[i] += _mm_popcnt_u64(_mm_movemask_epi8(_mm_cmpeq_epi8(d, _mm_set1_epi8(static_cast<char>(i)))));
				}
			}

			//TODO: ERROR here: some training data is not counted

			// ASSUME THAT NO MISSING VALUES EXIST
			double h = 0;
			for (int i = 0; i < N_DISTINCT_VALUES; ++i)
			{
				//std::cout << "INFO: _mm_entropy_epu8_method1: freq[" << i <<"]=" << freq[i] << std::endl;
				if (freq[i] > 0)
				{
					double prob = (static_cast<double>(freq[i]) / nElements);
					h += prob * log2(prob);
				}
			}
			return _mm_set1_pd(-h);
		}

		template <int N_BITS1, int N_BITS2, bool HAS_MV, U8 MV>
		inline __m128d _mm_entropy_epu8_method0(
			const std::tuple<const __m128i * const, const int>& data1,
			const std::tuple<const __m128i * const, const int>& data2,
			const int nElements)
		{
			static_assert(N_BITS1 > 0, "NBITS_1 has to be larger than zero");
			static_assert(N_BITS2 > 0, "NBITS_2 has to be larger than zero");
			const int N_BITS = N_BITS1 + N_BITS2;
			static_assert(N_BITS <= 8, "NBITS_1 + N_BITS_2 has to be smaller or equal than 8");

			const U8 * const ptr1 = reinterpret_cast<const U8 * const>(std::get<0>(data1));
			const U8 * const ptr2 = reinterpret_cast<const U8 * const>(std::get<0>(data2));

			const int N_DISTINCT_VALUES = (1 << N_BITS);
			std::array<int, N_DISTINCT_VALUES> freq;
			freq.fill(0);

			int nEffectiveElements;

			if (HAS_MV)
			{
				nEffectiveElements = 0;
				for (int element = 0; element < nElements; ++element)
				{
					const U8 d1 = ptr1[element];
					const U8 d2 = ptr2[element];
					if ((d1 != MV) && (d2 != MV))
					{
						const U8 mergedData = d1 | (d2 << N_BITS1);
						freq[mergedData]++;
						nEffectiveElements++;
					}
				}
			}
			else
			{
				nEffectiveElements = nElements;
				for (int element = 0; element < nElements; ++element)
				{
					const U8 d1 = ptr1[element];
					const U8 d2 = ptr2[element];
					const U8 mergedData = d1 | (d2 << N_BITS1);
					freq[mergedData]++;
				}
			}

			double h = 0;
			for (int i = 0; i < N_DISTINCT_VALUES; ++i)
			{
				//std::cout << "INFO: _mm_entropy_epu8_method1: freq[" << i <<"]=" << freq[i] << std::endl;
				if (freq[i] > 0)
				{
					double prob = (static_cast<double>(freq[i]) / nEffectiveElements);
					h += prob * log2(prob);
				}
			}
			return _mm_set1_pd(-h);
		}

		template <int N_BITS1, int N_BITS2, bool HAS_MV, U8 MV>
		inline __m128d _mm_entropy_epu8_method1(
			const std::tuple<const __m128i * const, const int>& data1,
			const std::tuple<const __m128i * const, const int>& data2,
			const int nElements)
		{
			const std::tuple<__m128i * const, const int> data3 = _mm_malloc_m128i(std::get<1>(data1));
			merge_U8xU8_to_U8<N_BITS1, N_BITS2, HAS_MV, MV>(data1, data2, data3);
			const int N_BITS3 = N_BITS1 + N_BITS2;
			const __m128d result = _mm_entropy_epu8<N_BITS3, HAS_MV, MV>(data3, nElements);
			_mm_free2(data3);
			return result;
		}

		template <int N_BITS1, int N_BITS2, bool HAS_MV, U8 MV>
		inline __m128d _mm_entropy_epu8_method2(
			const std::tuple<const __m128i * const, const int>& data1,
			const std::tuple<const __m128i * const, const int>& data2,
			const int nElements)
		{
			//TODO
			return _mm_entropy_epu8_method0<N_BITS1, N_BITS2, HAS_MV, MV>(data1, data2, nElements);
		}

		template <int N_BITS1, int N_BITS2, bool HAS_MV, U8 MV>
		inline __m128d _mm_entropy_epu8_method3(
			const std::tuple<const __m128i * const, const int>& data1,
			const std::tuple<const __m128i * const, const int>& data2,
			const int nElements)
		{
			//TODO
			return _mm_entropy_epu8_method0<N_BITS1, N_BITS2, HAS_MV, MV>(data1, data2, nElements);
		}
	}

	namespace test
	{
		void _mm_entropy_epu8_speed_test_1(const int nBlocks, const int nExperiments, const bool doTests)
		{
			const double delta = 0.0000001;
			const bool HAS_MV = false;
			const U8 MV = 0xFF;

			const int nElements = 16 * nBlocks;
			const int N_BITS1 = 3;
			const int N_BITS2 = 2;

			auto data1 = _mm_malloc_m128i(nElements);
			auto data2 = _mm_malloc_m128i(nElements);
			fillRand_epu8<N_BITS1>(data1);
			fillRand_epu8<N_BITS2>(data2);

			double min0 = std::numeric_limits<double>::max();
			double min1 = std::numeric_limits<double>::max();
			double min2 = std::numeric_limits<double>::max();
			double min3 = std::numeric_limits<double>::max();
			//			double min4 = std::numeric_limits<double>::max();

			__m128d result0, result1, result2, result3;

			for (int i = 0; i < nExperiments; ++i)
			{
				reset_and_start_timer();
				result0 = hli::priv::_mm_entropy_epu8_method0<N_BITS1, N_BITS2, HAS_MV, MV>(data1, data2, nElements);
				min0 = std::min(min0, get_elapsed_kcycles());

				{
					reset_and_start_timer();
					result1 = hli::priv::_mm_entropy_epu8_method1<N_BITS1, N_BITS2, HAS_MV, MV>(data1, data2, nElements);
					min1 = std::min(min1, get_elapsed_kcycles());

					if (doTests)
					{
						if (std::abs(result0.m128d_f64[0] - result1.m128d_f64[0]) > delta)
						{
							std::cout << "WARNING: test _mm_entropy_epu8_method0<" << N_BITS1 << "," << N_BITS2 << ">: result-ref=" << hli::toString_f64(result0) << "; result=" << hli::toString_f64(result1) << std::endl;
							return;
						}
					}
				}
				{
					reset_and_start_timer();
					result2 = hli::priv::_mm_entropy_epu8_method2<N_BITS1, N_BITS2, HAS_MV, MV>(data1, data2, nElements);
					min2 = std::min(min2, get_elapsed_kcycles());

					if (doTests)
					{
						if (std::abs(result0.m128d_f64[0] - result2.m128d_f64[0]) > delta)
						{
							std::cout << "WARNING: test _mm_entropy_epu8_method1<" << N_BITS1 << "," << N_BITS2 << ">: result-ref=" << hli::toString_f64(result0) << "; result=" << hli::toString_f64(result2) << std::endl;
							return;
						}
					}
				}
				{
					reset_and_start_timer();
					result3 = hli::priv::_mm_entropy_epu8_method3<N_BITS1, N_BITS2, HAS_MV, MV>(data1, data2, nElements);
					min3 = std::min(min3, get_elapsed_kcycles());

					if (doTests)
					{
						if (std::abs(result0.m128d_f64[0] - result3.m128d_f64[0]) > delta)
						{
							std::cout << "WARNING: test _mm_entropy_epu8_method1<" << N_BITS1 << "," << N_BITS2 << ">: result-ref=" << hli::toString_f64(result0) << "; result=" << hli::toString_f64(result3) << std::endl;
							return;
						}
					}
				}

			}
			printf("[_mm_entropy_epu8_method0<%i,%i>]: %2.5f Kcycles; %0.14f\n", N_BITS1, N_BITS2, min0, result0.m128d_f64[0]);
			printf("[_mm_entropy_epu8_method1<%i,%i>]: %2.5f Kcycles; %0.14f; %2.3f times faster than ref\n", N_BITS1, N_BITS2, min1, result1.m128d_f64[0], min0 / min1);
			printf("[_mm_entropy_epu8_method2<%i,%i>]: %2.5f Kcycles; %0.14f; %2.3f times faster than ref\n", N_BITS1, N_BITS2, min2, result2.m128d_f64[0], min0 / min2);
			printf("[_mm_entropy_epu8_method3<%i,%i>]: %2.5f Kcycles; %0.14f; %2.3f times faster than ref\n", N_BITS1, N_BITS2, min3, result3.m128d_f64[0], min0 / min3);

			_mm_free2(data1);
			_mm_free2(data2);
		}
	}

	template <int N_BITS, bool HAS_MV, U8 MV>
	inline __m128d _mm_entropy_epu8(
		const std::tuple<const __m128i * const, const int>& data,
		const int nElements)
	{
		const __m128d result = priv::_mm_entropy_epu8_method0<N_BITS, HAS_MV, MV>(data, nElements);

		#		if _DEBUG
		if (isnan(result.m128d_f64[0])) std::cout << "WARNING: _mm_entropy_epu8: result is NAN" << std::endl;
		if (result.m128d_f64[0] < 0)    std::cout << "WARNING: _mm_entropy_epu8: result is smaller than 0. result=" << result.m128d_f64[0] << std::endl;
		if (result.m128d_f64[0] > N_BITS) std::cout << "WARNING: _mm_entropy_epu8: result is larger than N_BITS=" << N_BITS << ". result=" << result.m128d_f64[0] << std::endl;
		#		endif
		return result;
	}

	template <bool HAS_MV, U8 MV>
	inline __m128d _mm_entropy_epu8(
		const std::tuple<const __m128i * const, const int>& data,
		const int nBits,
		const int nElements)
	{
		#if _DEBUG
		if ((nBits > 8) || (nBits < 1)) std::cout << "WARNING: _mm_entropy_epu8: nBits=" << nBits << " has to be in range[1..8]" << std::endl;
		#endif

		switch (nBits)
		{
			case 1: return _mm_entropy_epu8<1, HAS_MV, MV>(data, nElements);
			case 2: return _mm_entropy_epu8<2, HAS_MV, MV>(data, nElements);
			case 3: return _mm_entropy_epu8<3, HAS_MV, MV>(data, nElements);
			case 4: return _mm_entropy_epu8<4, HAS_MV, MV>(data, nElements);
			case 5: return _mm_entropy_epu8<5, HAS_MV, MV>(data, nElements);
			case 6: return _mm_entropy_epu8<6, HAS_MV, MV>(data, nElements);
			case 7: return _mm_entropy_epu8<7, HAS_MV, MV>(data, nElements);
			case 8: return _mm_entropy_epu8<8, HAS_MV, MV>(data, nElements);
			default: return _mm_setzero_pd();
		}
	}

	template <int N_BITS1, int N_BITS2, bool HAS_MV, U8 MV>
	inline __m128d _mm_entropy_epu8(
		const std::tuple<const __m128i * const, const int>& data1,
		const std::tuple<const __m128i * const, const int>& data2,
		const int nElements)
	{
		const __m128d result = priv::_mm_entropy_epu8_method0<N_BITS1, N_BITS2, HAS_MV, MV>(data1, data2, nElements);
		//const __m128d result = priv::_mm_entropy_epu8_method1<N_BITS1, N_BITS2, HAS_MV, MV>(data1, data2, nElements);

		#		if	_DEBUG
		if (isnan(result.m128d_f64[0])) std::cout << "WARNING: _mm_entropy_epu8: result is NAN" << std::endl;
		if (result.m128d_f64[0] < 0) std::cout << "WARNING: _mm_entropy_epu8: result is smaller than 0. " << result.m128d_f64[0] << std::endl;
		#		endif
		return result;
	}

	template <bool HAS_MV, U8 MV>
	inline __m128d _mm_entropy_epu8(
		const std::tuple<const __m128i * const, const int>& data1,
		const int nBits1,
		const std::tuple<const __m128i * const, const int>& data2,
		const int nBits2,
		const int nElements)
	{
		#if _DEBUG
		if ((nBits1 > 8) || (nBits1 < 1)) std::cout << "WARNING: _mm_entropy_epu8: nBits1=" << nBits1 << " has to be in range[1..8]" << std::endl;
		if ((nBits2 > 8) || (nBits2 < 1)) std::cout << "WARNING: _mm_entropy_epu8: nBits2=" << nBits2 << " has to be in range[1..8]" << std::endl;
		#endif

		switch (nBits1)
		{
			case 1:
				switch (nBits2)
				{
					case 1: return _mm_entropy_epu8<1, 1, HAS_MV, MV>(data1, data2, nElements);
					case 2: return _mm_entropy_epu8<1, 2, HAS_MV, MV>(data1, data2, nElements);
					case 3: return _mm_entropy_epu8<1, 3, HAS_MV, MV>(data1, data2, nElements);
					case 4: return _mm_entropy_epu8<1, 4, HAS_MV, MV>(data1, data2, nElements);
					case 5: return _mm_entropy_epu8<1, 5, HAS_MV, MV>(data1, data2, nElements);
					case 6: return _mm_entropy_epu8<1, 6, HAS_MV, MV>(data1, data2, nElements);
					case 7: return _mm_entropy_epu8<1, 7, HAS_MV, MV>(data1, data2, nElements);
					default: return _mm_setzero_pd();
				}
			case 2:
				switch (nBits2)
				{
					case 1: return _mm_entropy_epu8<2, 1, HAS_MV, MV>(data1, data2, nElements);
					case 2: return _mm_entropy_epu8<2, 2, HAS_MV, MV>(data1, data2, nElements);
					case 3: return _mm_entropy_epu8<2, 3, HAS_MV, MV>(data1, data2, nElements);
					case 4: return _mm_entropy_epu8<2, 4, HAS_MV, MV>(data1, data2, nElements);
					case 5: return _mm_entropy_epu8<2, 5, HAS_MV, MV>(data1, data2, nElements);
					case 6: return _mm_entropy_epu8<2, 6, HAS_MV, MV>(data1, data2, nElements);
					default: return _mm_setzero_pd();
				}
			case 3:
				switch (nBits2)
				{
					case 1: return _mm_entropy_epu8<3, 1, HAS_MV, MV>(data1, data2, nElements);
					case 2: return _mm_entropy_epu8<3, 2, HAS_MV, MV>(data1, data2, nElements);
					case 3: return _mm_entropy_epu8<3, 3, HAS_MV, MV>(data1, data2, nElements);
					case 4: return _mm_entropy_epu8<3, 4, HAS_MV, MV>(data1, data2, nElements);
					case 5: return _mm_entropy_epu8<3, 5, HAS_MV, MV>(data1, data2, nElements);
					default: return _mm_setzero_pd();
				}
			case 4:
				switch (nBits2)
				{
					case 1: return _mm_entropy_epu8<4, 1, HAS_MV, MV>(data1, data2, nElements);
					case 2: return _mm_entropy_epu8<4, 2, HAS_MV, MV>(data1, data2, nElements);
					case 3: return _mm_entropy_epu8<4, 3, HAS_MV, MV>(data1, data2, nElements);
					case 4: return _mm_entropy_epu8<4, 4, HAS_MV, MV>(data1, data2, nElements);
					default: return _mm_setzero_pd();
				}
			case 5:
				switch (nBits2)
				{
					case 1: return _mm_entropy_epu8<5, 1, HAS_MV, MV>(data1, data2, nElements);
					case 2: return _mm_entropy_epu8<5, 2, HAS_MV, MV>(data1, data2, nElements);
					case 3: return _mm_entropy_epu8<5, 3, HAS_MV, MV>(data1, data2, nElements);
					default: return _mm_setzero_pd();
				}
			case 6:
				switch (nBits2)
				{
					case 1: return _mm_entropy_epu8<6, 1, HAS_MV, MV>(data1, data2, nElements);
					case 2: return _mm_entropy_epu8<6, 2, HAS_MV, MV>(data1, data2, nElements);
					default: return _mm_setzero_pd();
				}
			case 7:
				switch (nBits2)
				{
					case 1: return _mm_entropy_epu8<7, 1, HAS_MV, MV>(data1, data2, nElements);
					default: return _mm_setzero_pd();
				}
			default: return _mm_setzero_pd();
		}
	}
}