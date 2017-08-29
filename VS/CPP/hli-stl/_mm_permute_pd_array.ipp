#pragma once

#include <iostream>		// std::cout
#include <math.h>
#include <tuple>

//#include "mmintrin.h"  // mmx
#include "emmintrin.h"  // sse
#include "pmmintrin.h"  // sse3
#include "tmmintrin.h"  // ssse3
#include "smmintrin.h"  // sse4.1
#include "nmmintrin.h"  // sse4.2
#include "immintrin.h"  // avx, avx2, avx512, FP16C, KNCNI, FMA
//#include "ammintrin.h"  // AMD-specific intrinsics

#include "tools.ipp"
#include "equal.ipp"
#include "swap.ipp"

#include "_mm_rand_si128.ipp"
#include "_mm_rescale_epu16.ipp"

namespace hli
{

	namespace priv
	{

		inline void _mm_permute_pd_array_method0(
			const std::tuple<__m128d * const, const size_t>& data,
			const size_t nElements,
			const std::tuple<__m128i * const, const size_t>& swap,
			__m128i& randInts)
		{
			_mm_lfsr32_epu32(swap, randInts);
			_mm_rescale_epu16_method0(swap);

			{	// perform the swapping, cannot be done in a vectorized manner
				__m128d * const tmp = std::get<0>(data);
				double * const data2 = reinterpret_cast<double * const>(tmp);
				U16 * const swap_array_int = reinterpret_cast<U16 * const>(std::get<0>(swap));
				swapArray(data2, swap_array_int, nElements);
			}
		}

		inline void _mm_permute_pd_array_method2(
			const std::tuple<__m128d * const, const size_t>& data,
			const size_t nElements,
			const std::tuple<__m128i * const, const size_t>& swap,
			__m128i& randInts)
		{
			_mm_lfsr32_epu32(swap, randInts);
			_mm_rescale_epu16(swap);

			{	// perform the swapping, cannot be done in a vectorized manner
				__m128d * const tmp = std::get<0>(data);
				double * const data2 = reinterpret_cast<double * const>(tmp);
				U16 * const swap_array_int = reinterpret_cast<U16 * const>(std::get<0>(swap));
				swapArray(data2, swap_array_int, nElements);
			}
		}
	}

	namespace test
	{

		void _mm_permute_pd_array_speed_test_1(
			const size_t nBlocks,
			const size_t nExperiments,
			const bool doTests)
		{
			if ((nBlocks * 8) > 0xFFFF)
			{
				std::cout << "WARNING: t test_mm_permute_epu8: too many blocks=" << nBlocks << std::endl;
				return;
			}

			const size_t nBytes = 16 * nBlocks;
			auto data = _mm_malloc_m128d(nBytes);
			auto data_1 = _mm_malloc_m128d(nBytes);
			auto data_2 = _mm_malloc_m128d(nBytes);

			const size_t nElements = nBytes >> 3;
			auto swap = _mm_malloc_m128i(nElements << 1);

			const __m128i seed = _mm_set_epi32(rand() || rand() << 16, rand() || rand() << 16, rand() || rand() << 16, rand() || rand() << 16);
			__m128i randInt = seed;
			__m128i randInt1 = seed;
			const int N_BITS = 5;

			fillRand_pd(data);

			double min_ref = std::numeric_limits<double>::max();
			double min1 = std::numeric_limits<double>::max();

			for (size_t i = 0; i < nExperiments; ++i)
			{
				memcpy(std::get<0>(data_1), std::get<0>(data), nBytes);
				timer::reset_and_start_timer();
				hli::priv::_mm_permute_pd_array_method0(data_1, nElements, swap, randInt);
				min_ref = std::min(min_ref, timer::get_elapsed_kcycles());

				{
					memcpy(std::get<0>(data_2), std::get<0>(data), nBytes);
					timer::reset_and_start_timer();
					hli::priv::_mm_permute_pd_array_method2(data_2, nElements, swap, randInt1);
					min1 = std::min(min1, timer::get_elapsed_kcycles());

					if (doTests)
					{
						for (size_t block = 0; block < nBlocks; ++block)
						{
							if (!equal(std::get<0>(data_1)[block], std::get<0>(data_2)[block]))
							{
								std::cout << "WARNING: test_mm_permute_epu8: result-ref=" << hli::toString_f64(std::get<0>(data_1)[block]) << "; result1=" << hli::toString_f64(std::get<0>(data_2)[block]) << std::endl;
								return;
							}
						}
						if (!equal(randInt, randInt1))
						{
							std::cout << "WARNING: test_mm_permute_epu8: randInt=" << hli::toString_u32(randInt) << "; randInt1=" << hli::toString_u32(randInt1) << std::endl;
							return;
						}
					}
				}
			}
			if (doTests)
			{
				//	const __m128i sum1 = hli::_mm_hadd_epu8<N_BITS>(mem_addr1, nBytes);
				//	const __m128i sum2 = hli::_mm_hadd_epu8<N_BITS>(mem_addr2, nBytes);
				//	if (sum1.m128i_u32[0] != sum2.m128i_u32[0]) {
				//		std::cout << "WARNING: test_mm_permute_epu8: sums are unequal: sum1=" << sum1.m128i_u32[0] << "; sum2=" << sum2.m128i_u32[0] << std::endl;
				//	}
			}

			printf("[_mm_permute_pd_array_method0]: %2.5f Kcycles\n", min_ref);
			printf("[_mm_permute_pd_array_method1]: %2.5f Kcycles; %2.3f times faster than ref\n", min1, min_ref / min1);

			_mm_free2(data);
			_mm_free2(data_1);
			_mm_free2(data_2);
			_mm_free2(swap);
		}

	}

	inline void _mm_permute_pd_array(
		const std::tuple<__m128d * const, const size_t> data,
		const size_t nElements,
		const std::tuple<__m128i * const, const size_t> swap,
		__m128i& randInts)
	{
		//std::cout << "INFO: _mm_permute_array::_mm_permute_dp_array: nBytes=" << nBytes << std::endl;
		priv::_mm_permute_pd_array_method2(data, nElements, swap, randInts);
	}
}