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
#include "immintrin.h"  // avx, avx2, avx512, FP16C, KNCNI, FMA
//#include "ammintrin.h"  // AMD-specific intrinsics

#include "toString.ipp"



namespace hli {

	namespace priv {

		inline void _mm_rescale_epu16_method0(
			const std::tuple<__m128i * const, const size_t>& data)
		{
			const size_t nBytes = std::get<1>(data);
			const size_t nElements = nBytes >> 1;
			const bool showInfo = false;

			if (showInfo) std::cout << "_mm_rescale_epu16_method0: nElements=" << nElements << std::endl;
			unsigned __int16 * const ptr2 = reinterpret_cast<unsigned __int16 * const>(std::get<0>(data));

			for (size_t i = 0; i < nElements; ++i) {
				//const size_t block = nElements >> 3;
				//if (block == 178-1) std::cout << "_mm_rescale_epu16_method0: input: i=" << i << ":" << ptr2[i] << std::endl;

				const unsigned __int16 original = ptr2[i];
				const unsigned int product = original * static_cast<unsigned int>(i + 1);
				const unsigned int result = product >> 16;
				//if (block == 178) std::cout << "_mm_rescale_epu16_method0: i=" << i << "; original " << original << "; product " << product << "; result " << result << std::endl;

				ptr2[i] = static_cast<unsigned __int16>(result);
				//std::cout << "rescaleVector: after: i=" << i << ":" << ptr[i] << std::endl;
			}
		}

		inline void _mm_rescale_epu16_method1(
			const std::tuple<__m128i * const, const size_t>& data)
		{
			__m128i * const ptr = std::get<0>(data);
			const size_t nBytes = std::get<1>(data);
			const size_t nBlocks = nBytes >> 4;
			const bool showInfo = false;

			__m128i m = _mm_set_epi32(4, 3, 2, 1);
			const __m128i increment = _mm_set1_epi32(4);

			if (showInfo) std::cout << "_mm_rescale_epu16_method1: increment=" << toString_i32(increment) << std::endl;

			for (size_t i = 0; i < nBlocks; ++i) {
				const __m128i block = ptr[i];
				if (showInfo) std::cout << "_mm_rescale_epu16_method1: input data=" << toString_u16(block) << std::endl;

				const __m128i numbers1 = _mm_cvtepu16_epi32(block);
				const __m128i numbers2 = _mm_cvtepu16_epi32(_mm_swap_64(block));
				if (showInfo) std::cout << "_mm_rescale_epu16_method1: input numbers1=" << toString_u32(numbers1) << "; input numbers2=" << toString_u32(numbers2) << std::endl;

				if (showInfo) std::cout << "_mm_rescale_epu16_method1 :max " << toString_i32(m) << std::endl;

				const __m128i product1 = _mm_mullo_epi32(numbers1, m);
				if (showInfo) std::cout << "_mm_rescale_epu16_method1: product1 " << toString_i32(product1) << std::endl;
				m = _mm_add_epi32(m, increment);

				const __m128i product2 = _mm_mullo_epi32(numbers2, m);
				if (showInfo) std::cout << "_mm_rescale_epu16_method1: product2 " << toString_i32(product2) << std::endl;
				m = _mm_add_epi32(m, increment);


				const __m128i f3_int1 = _mm_srli_epi32(product1, 16); //Shifts the 4 signed or unsigned 32 - bit integers in a right by count bits while shifting in zeros.
				if (showInfo) std::cout << "_mm_rescale_epu16_method1: output int1 " << toString_i32(f3_int1) << std::endl;
				const __m128i f3_int2 = _mm_srli_epi32(product2, 16); //Shifts the 4 signed or unsigned 32 - bit integers in a right by count bits while shifting in zeros.
				if (showInfo) std::cout << "_mm_rescale_epu16_method1: output int2 " << toString_i32(f3_int2) << std::endl;
	
				const __m128i saturated = _mm_packs_epi32(f3_int1, f3_int2);
				if (showInfo) std::cout << "_mm_rescale_epu16_method1: output short " << toString_u16(saturated) << std::endl;

				if (showInfo) std::cout << std::endl;
				ptr[i] = saturated;
			}
		}

		inline void _mm_rescale_epu16_method2(
			const std::tuple<__m128i * const, const size_t>& data)
		{
			const size_t nBytes = std::get<1>(data);
			const size_t nBlocks = nBytes >> 4;

			__m128i m = _mm_set_epi16(8, 7, 6, 5, 4, 3, 2, 1);
			const __m128i increment = _mm_set1_epi16(8);

			for (size_t block = 0; block < nBlocks; ++block)
			{
				const __m128i input = std::get<0>(data)[block];
				//if (block == 178) std::cout << "_mm_rescale_epu16_method2: input  =" << toString_u16(input) << std::endl;

				const __m128i output = _mm_mulhi_epu16(input, m);
				m = _mm_add_epi16(m, increment);
				//if (block == 178) std::cout << "_mm_rescale_epu16_method2: output =" << toString_u16(output) << std::endl;

				//if (showInfo) std::cout << std::endl;
				std::get<0>(data)[block] = output;
			}
		}
	}

	namespace test {

		void _mm_rescale_epu16_speed_test_1(
			const size_t nBlocks,
			const size_t nExperiments,
			const bool doTests)
		{
			if ((nBlocks * 8) > 0xFFFF) {
				std::cout << "WARNING: test_mm_rescale_epu16: too many blocks=" << nBlocks << std::endl;
				return;
			}

			auto data_source_r = _mm_malloc_m128i(16 * nBlocks);
			auto data0 = _mm_malloc_m128i(16 * nBlocks);
			auto data1 = _mm_malloc_m128i(16 * nBlocks);
			auto data2 = _mm_malloc_m128i(16 * nBlocks);

			const __m128i seed = _mm_set_epi16((short)rand(), (short)rand(), (short)rand(), (short)rand(), (short)rand(), (short)rand(), (short)rand(), (short)rand());
			__m128i randInt = seed;

			hli::_mm_lfsr32_epu32(data_source_r, randInt);
			const std::tuple<const __m128i * const, const size_t> data_source = data_source_r;

			double min0 = std::numeric_limits<double>::max();
			double min1 = std::numeric_limits<double>::max();
			double min2 = std::numeric_limits<double>::max();

			for (size_t i = 0; i < nExperiments; ++i)
			{
				copy(data_source, data0);
				timer::reset_and_start_timer();
				hli::priv::_mm_rescale_epu16_method0(data0);
				min0 = std::min(min0, timer::get_elapsed_kcycles());

				{
					copy(data_source, data1);
					timer::reset_and_start_timer();
					hli::priv::_mm_rescale_epu16_method1(data1);
					min1 = std::min(min1, timer::get_elapsed_kcycles());

					if (doTests) {
						for (size_t block = 0; block < nBlocks; ++block) {
							for (size_t j = 0; j < 8; ++j) {
								if (std::abs(std::get<0>(data0)[block].m128i_u16[j] != std::get<0>(data1)[block].m128i_u16[j])) {
									std::cout << "WARNING: test mm_rescale_epu16: result0=" << hli::toString_u16(std::get<0>(data0)[block]) << "; result1=" << hli::toString_u16(std::get<0>(data1)[block]) << std::endl;
									return;
								}
							}
						}
					}
				}
				{
					copy(data_source, data2);
					timer::reset_and_start_timer();
					hli::priv::_mm_rescale_epu16_method2(data2);
					min2 = std::min(min2, timer::get_elapsed_kcycles());

					if (doTests) {
						for (size_t block = 0; block < nBlocks; ++block) {
							for (size_t j = 0; j < 8; ++j) {
								if (std::abs(std::get<0>(data0)[block].m128i_u16[j] != std::get<0>(data2)[block].m128i_u16[j])) {
									std::cout << "WARNING: test mm_rescale_epu16: result0=" << hli::toString_u16(std::get<0>(data0)[block]) << "; result2=" << hli::toString_u16(std::get<0>(data2)[block]) << std::endl;
									return;
								}
							}
						}
					}
				}
			}
			if (doTests) {
				__int16 k = 0;
				for (size_t block = 0; block < nBlocks; ++block) {
					for (size_t j = 0; j < 8; ++j) {
						if (std::get<0>(data0)[block].m128i_u16[j] > k) {
							std::cout << "WARNING: test mm_rescale_epu16: position " << k << " has value " << std::get<0>(data0)[block].m128i_u16[j] << " which is too large" << std::endl;
							return;
						}
						k++;
					}
					//std::cout << "INFO: test mm_rescale_epu16: block=" << block << "; result=" << hli::toString_u16(mem_addr1[block]) << std::endl;
				}
			}

			printf("[_mm_rescale_epu16_method0]: %2.5f Kcycles\n", min0);
			printf("[_mm_rescale_epu16_method1]: %2.5f Kcycles; %2.3f times faster than ref\n", min1, min0 / min1);
			printf("[_mm_rescale_epu16_method2]: %2.5f Kcycles; %2.3f times faster than ref\n", min2, min0 / min2);

			_mm_free2(data_source);
			_mm_free2(data0);
			_mm_free2(data1);
		}
	}

	inline void _mm_rescale_epu16(
		const std::tuple<__m128i * const, const size_t>& data)
	{
		priv::_mm_rescale_epu16_method2(data);

		//for (size_t block = 0; block < (nBytes >> 4); ++block) {
		//	std::cout << "INFO: _mm_rescale_epu16::_mm_rescale_epu16: block="<<block << ": " << toString_u16(mem_addr[block]) << std::endl;
		//}
	}
}