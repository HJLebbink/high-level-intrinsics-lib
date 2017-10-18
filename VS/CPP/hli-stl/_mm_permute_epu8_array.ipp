#pragma once

#include <algorithm>	// std::min
#include <limits>		// std::numeric_limits
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

#include "toString.ipp"
#include "timing.ipp"
#include "equal.ipp"
#include "tools.ipp"
#include "swap.ipp"

#include "_mm_rand_si128.ipp"
#include "_mm_rescale_epu16.ipp"

namespace hli
{

	namespace priv
	{

		inline void _mm_permute_epu8_array_method0(
			const std::tuple<__m128i * const, const int>& data,
			const int nElements,
			const std::tuple<__m128i * const, const int>& swap,
			__m128i& randInts)
		{
			_mm_rand_si128_ref(swap, randInts);
			_mm_rescale_epu16_method0(swap);
			U8 * const data_ptr = reinterpret_cast<U8 * const>(std::get<0>(data));
			U16 * const swap_array_int = reinterpret_cast<U16 * const>(std::get<0>(swap));
			swapArray(data_ptr, swap_array_int, nElements);
		}

		inline void _mm_permute_epu8_array_method1(
			const std::tuple<__m128i * const, const int>& data,
			const int nElements,
			const std::tuple<__m128i * const, const int>& swap,
			__m128i& randInts)
		{
			_mm_lfsr32_epu32(swap, randInts);
			_mm_rescale_epu16_method1(swap);
			U8 * const data_ptr = reinterpret_cast<U8 * const>(std::get<0>(data));
			U16 * const swap_array_int = reinterpret_cast<U16 * const>(std::get<0>(swap));
			swapArray(data_ptr, swap_array_int, nElements);
		}

		inline void _mm_permute_epu8_array_method2(
			const std::tuple<__m128i * const, const int>& data,
			const int nElements,
			const std::tuple<__m128i * const, const int>& swap,
			__m128i& randInts)
		{
			_mm_lfsr32_epu32(swap, randInts);
			_mm_rescale_epu16_method2(swap);
			U8 * const data_ptr = reinterpret_cast<U8 * const>(std::get<0>(data));
			U16 * const swap_array_int = reinterpret_cast<U16 * const>(std::get<0>(swap));
			swapArray(data_ptr, swap_array_int, nElements);
		}

		inline void _mm_permute_epu8_array_method3(
			const std::tuple<__m128i * const, const int>& data,
			const int nElements,
			const std::tuple<__m128i * const, const int>& swap,
			__m128i& randInts)
		{
			if (true)
			{
				_mm_lfsr32_epu32(swap, randInts);
				_mm_rescale_epu16_method2(swap);
			}
			else
			{

				//following code seems broken

				__m128i * const ptr = std::get<0>(swap);
				const int nBytes = std::get<1>(swap);
				const int nBlocks = nBytes >> 4;

				__m128i m = _mm_set_epi16(8, 7, 6, 5, 4, 3, 2, 1);
				const __m128i increment = _mm_set1_epi16(8);

				for (int block = 0; block < nBlocks; ++block)
				{
					__m128i randInts2 = priv::lfsr32_galois(randInts);
					randInts = priv::lfsr32_galois(randInts);
					__m128i rand16Bits = _mm_xor_si128(randInts, _mm_swap_64(randInts2));
					ptr[block] = _mm_mulhi_epu16(rand16Bits, m);
					m = _mm_add_epi16(m, increment);
				}
			}
			U8 * const data_ptr = reinterpret_cast<U8 * const>(std::get<0>(data));
			U16 * const swap_array_int = reinterpret_cast<U16 * const>(std::get<0>(swap));
			swapArray(data_ptr, swap_array_int, nElements);
		}
	}

	namespace test
	{
		using namespace tools::timing;

		void _mm_permute_epu8_array_speed_test_1(
			const int nBlocks,
			const int nExperiments,
			const bool doTests)
		{
			if ((nBlocks * 8) > 0xFFFF)
			{
				std::cout << "WARNING: t test_mm_permute_epu8: too many blocks=" << nBlocks << std::endl;
				return;
			}

			const bool HAS_MV = true;
			const U8 MV = 0xFF;

			const int nElements = 8 * nBlocks;
			const int nBytes = nElements * 2;
			const int N_BITS = 5;

			auto data_source_r = _mm_malloc_m128i(nBytes);
			auto data0 = _mm_malloc_m128i(nBytes);
			auto data1 = _mm_malloc_m128i(nBytes);
			auto data2 = _mm_malloc_m128i(nBytes);
			auto data3 = _mm_malloc_m128i(nBytes);


			fillRand_epu8<N_BITS>(data_source_r);
			const std::tuple<const __m128i * const, const int> data_source = data_source_r;

			auto swap = _mm_malloc_m128i(nElements << 1);

			const __m128i seed = _mm_set_epi32(rand() || rand() << 16, rand() || rand() << 16, rand() || rand() << 16, rand() || rand() << 16);
			__m128i randInt0 = seed;
			__m128i randInt1 = seed;
			__m128i randInt2 = seed;
			__m128i randInt3 = seed;


			double min0 = std::numeric_limits<double>::max();
			double min1 = std::numeric_limits<double>::max();
			double min2 = std::numeric_limits<double>::max();
			double min3 = std::numeric_limits<double>::max();

			for (int i = 0; i < nExperiments; ++i)
			{
				copy(data_source, data0);
				reset_and_start_timer();
				hli::priv::_mm_permute_epu8_array_method0(data0, nElements, swap, randInt0);
				min0 = std::min(min0, get_elapsed_kcycles());

				{
					copy(data_source, data1);
					reset_and_start_timer();
					hli::priv::_mm_permute_epu8_array_method1(data1, nElements, swap, randInt1);
					min1 = std::min(min1, get_elapsed_kcycles());

					if (doTests)
					{
						for (int block = 0; block < nBlocks; ++block)
						{
							if (!equal(std::get<0>(data0)[block], std::get<0>(data1)[block]))
							{
								std::cout << "WARNING: test_mm_permute_epu8: block=" << block << ": result0=" << hli::toString_u16(std::get<0>(data0)[block]) << "; result1=" << hli::toString_u16(std::get<0>(data1)[block]) << std::endl;
								return;
							}
						}
						if (!equal(randInt0, randInt1))
						{
							std::cout << "WARNING: test_mm_permute_epu8: randInt0=" << hli::toString_u32(randInt0) << "; randInt1=" << hli::toString_u32(randInt1) << std::endl;
							return;
						}
					}
				}
				{
					copy(data_source, data2);
					reset_and_start_timer();
					hli::priv::_mm_permute_epu8_array_method2(data2, nElements, swap, randInt2);
					min2 = std::min(min2, get_elapsed_kcycles());

					if (doTests)
					{
						for (int block = 0; block < nBlocks; ++block)
						{
							if (!equal(std::get<0>(data0)[block], std::get<0>(data2)[block]))
							{
								std::cout << "WARNING: test_mm_permute_epu8: block=" << block << ": result0=" << hli::toString_u16(std::get<0>(data0)[block]) << "; result2=" << hli::toString_u16(std::get<0>(data2)[block]) << std::endl;
								return;
							}
						}
						if (!equal(randInt0, randInt2))
						{
							std::cout << "WARNING: test_mm_permute_epu8: randInt0=" << hli::toString_u32(randInt0) << "; randInt2=" << hli::toString_u32(randInt2) << std::endl;
							return;
						}
					}
				}
				{
					copy(data_source, data3);
					reset_and_start_timer();
					hli::priv::_mm_permute_epu8_array_method3(data3, nElements, swap, randInt3);
					min3 = std::min(min3, get_elapsed_kcycles());

					if (doTests)
					{
						for (int block = 0; block < nBlocks; ++block)
						{
							if (!equal(std::get<0>(data0)[block], std::get<0>(data3)[block]))
							{
								std::cout << "WARNING: test_mm_permute_epu8: block=" << block << ": result0=" << hli::toString_u16(std::get<0>(data0)[block]) << "; result3=" << hli::toString_u16(std::get<0>(data3)[block]) << std::endl;
								return;
							}
						}
						if (!equal(randInt0, randInt3))
						{
							std::cout << "WARNING: test_mm_permute_epu8: randInt0=" << hli::toString_u32(randInt0) << "; randInt3=" << hli::toString_u32(randInt3) << std::endl;
							return;
						}
					}
				}
			}
			if (doTests)
			{
				const __m128i sum0 = std::get<0>(hli::_mm_hadd_epu8<N_BITS, HAS_MV, MV>(data0, nElements));
				const __m128i sum1 = std::get<0>(hli::_mm_hadd_epu8<N_BITS, HAS_MV, MV>(data1, nElements));
				const __m128i sum2 = std::get<0>(hli::_mm_hadd_epu8<N_BITS, HAS_MV, MV>(data2, nElements));
				const __m128i sum3 = std::get<0>(hli::_mm_hadd_epu8<N_BITS, HAS_MV, MV>(data3, nElements));
				if (sum0.m128i_u32[0] != sum1.m128i_u32[0])
				{
					std::cout << "WARNING: test_mm_permute_epu8: sums are unequal: sum0=" << sum0.m128i_u32[0] << "; sum1=" << sum1.m128i_u32[0] << std::endl;
				}
				if (sum0.m128i_u32[0] != sum2.m128i_u32[0])
				{
					std::cout << "WARNING: test_mm_permute_epu8: sums are unequal: sum0=" << sum1.m128i_u32[0] << "; sum2=" << sum2.m128i_u32[0] << std::endl;
				}
				if (sum0.m128i_u32[0] != sum3.m128i_u32[0])
				{
					std::cout << "WARNING: test_mm_permute_epu8: sums are unequal: sum0=" << sum1.m128i_u32[0] << "; sum3=" << sum3.m128i_u32[0] << std::endl;
				}
			}

			printf("[_mm_permute_epu8_method0]: %2.5f Kcycles\n", min0);
			printf("[_mm_permute_epu8_method1]: %2.5f Kcycles; %2.3f times faster than ref\n", min1, min0 / min1);
			printf("[_mm_permute_epu8_method2]: %2.5f Kcycles; %2.3f times faster than ref\n", min2, min0 / min2);
			printf("[_mm_permute_epu8_method3]: %2.5f Kcycles; %2.3f times faster than ref\n", min3, min0 / min3);

			_mm_free2(data_source);
			_mm_free2(data0);
			_mm_free2(data1);
			_mm_free2(data2);
			_mm_free2(data3);
			_mm_free2(swap);
		}
	}

	inline void _mm_permute_epu8_array(
		const std::tuple<__m128i * const, const int>& data,
		const int nElements,
		const std::tuple<__m128i * const, const int>& swap,
		__m128i& randInts)
	{
		#if _DEBUG
		int nBytes_Data = std::get<1>(data);
		if (nBytes_Data < nElements) std::cout << "ERROR: _mm_permute_epu8_array: nBytes_Data=" << nBytes_Data << " is smaller than nElements=" << nElements << ".";
		int nBytes_Swap = std::get<1>(swap);
		if (nBytes_Swap < nElements * 2) std::cout << "ERROR: _mm_permute_epu8_array: nBytes_Swap=" << nBytes_Swap << " is smaller than nElements*2=" << nElements * 2 << ".";
		//printf("INFO: _mm_permute_epu8_array: nElements=%d; data.nBytes=%d; swap.nBytes=%d\n", nElements, std::get<1>(data), std::get<1>(swap));
		#endif

		priv::_mm_permute_epu8_array_method0(data, nElements, swap, randInts);
		// there seems to be a bug with the other methods
		//priv::_mm_permute_epu8_array_method1(data, nElements, swap, randInts);
		//priv::_mm_permute_epu8_array_method2(data, nElements, swap, randInts);
		//priv::_mm_permute_epu8_array_method3(data, nElements, swap, randInts);
	}
}