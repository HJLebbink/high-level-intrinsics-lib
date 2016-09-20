#pragma once

#include <iostream>		// for cout
#include <math.h>
#include <algorithm>	// for std::max
#include <tuple>

//#include "mmintrin.h"  // mmx
#include "emmintrin.h"  // sse
#include "pmmintrin.h"  // sse3
#include "tmmintrin.h"  // ssse3
#include "smmintrin.h"  // sse4.1
#include "nmmintrin.h"  // sse4.2
#include "immintrin.h"  // avx, avx2, avx512, FP16C, KNCNI, FMA
//#include "ammintrin.h"  // AMD-specific intrinsics

#include "toString.h"
#include "timer.h"
#include "Equal.h"
#include "tools.h"

#include "_mm_rand_si128.h"
#include "_mm_rescale_epu16.h"

namespace hli {

	namespace priv {

		// swap the elements of data given pos1 and pos2
		template <class T>
		inline void swapElement(
			T * const data,
			const int pos1, 
			const unsigned __int16 pos2)
		{
			const T temp = data[pos1];
			//std::cout << "INFO: _mm_permute_array::swapElement: swapping pos1=" << pos1 << " with pos2=" << pos2 << "; data=" << temp << std::endl;
			data[pos1] = data[pos2];
			data[pos2] = temp;
		}

		template <class T>
		inline void swapArray(
			T * const data,
			const unsigned __int16 * const swap_array,
			const size_t nElements)
		{
			for (int i = static_cast<int>(nElements) - 1; i > 0; --i) {
#		if	_DEBUG 
				if (i >= static_cast<int>(nElements)) {
					std::cout << "ERROR: _mm_permutate_array:swapArray i=" << i << "; nElements=" << nElements << std::endl;
					return;
				}
				if (swap_array[i] >= nElements) {
					std::cout << "ERROR: _mm_permutate_array:swapArray i=" << i << "; swap_array[i]=" << swap_array[i] << "; nElements = " << nElements << std::endl;
					return;
				}
#		endif
				swapElement(data, i, swap_array[i]);
			}
		}

		inline void _mm_permute_epu8_array_ref(
			const std::tuple<__m128i * const, const size_t>& data,
			const size_t nElements,
			const std::tuple<__m128i * const, const size_t>& swap,
			__m128i& randInts)
		{
			_mm_rand_si128_ref(swap, randInts);
			_mm_rescale_epu16_ref(swap);

			unsigned __int8 * const data_ptr = reinterpret_cast<unsigned __int8 * const>(std::get<0>(data));
			unsigned __int16 * const swap_array_int = reinterpret_cast<unsigned __int16 * const>(std::get<0>(swap));

			swapArray(data_ptr, swap_array_int, nElements);
		}

		inline void _mm_permute_epu8_array_method1(
			const std::tuple<__m128i * const, const size_t>& data,
			const size_t nElements,
			const std::tuple<__m128i * const, const size_t>& swap,
			__m128i& randInts)
		{
			_mm_lfsr32_epu32(swap, randInts);
			_mm_rescale_epu16(swap);

			unsigned __int8 * const data_ptr = reinterpret_cast<unsigned __int8 * const>(std::get<0>(data));
			unsigned __int16 * const swap_array_int = reinterpret_cast<unsigned __int16 * const>(std::get<0>(swap));

			swapArray(data_ptr, swap_array_int, nElements);
		}

		inline void _mm_permute_epu8_array_method2(
			const std::tuple<__m128i * const, const size_t>& data,
			const size_t nElements,
			const std::tuple<__m128i * const, const size_t>& swap,
			__m128i& randInts)
		{
			const size_t nBytes = std::get<1>(data);
			//std::cout << "INFO: _mm_permute_array::_mm_permute_epu8_array_method2: nElements=" << nElements << std::endl;

			const size_t nBlocks = nBytes >> 4;

			if (false) {
				for (size_t block = 0; block < nBlocks; ++block) {
					randInts = priv::lfsr32_galois(randInts);
					std::get<0>(swap)[block] = randInts;
				}
			} else {
				_mm_lfsr32_epu32(swap, randInts);
			}
			if (false) {
				const __m128i increment = _mm_set1_epi32(4);
				__m128i m = _mm_set_epi32(4, 3, 2, 1);
				for (size_t block = 0; block < nBlocks; ++block) {
					const __m128i data = std::get<0>(swap)[block];
					const __m128i numbers1 = _mm_cvtepu16_epi32(data);
					const __m128i numbers2 = _mm_cvtepu16_epi32(_mm_shuffle_epi32(data, 0b11101110));
					const __m128i product1 = _mm_mullo_epi32(numbers1, m);
					m = _mm_add_epi32(m, increment);
					const __m128i product2 = _mm_mullo_epi32(numbers2, m);
					m = _mm_add_epi32(m, increment);
					const __m128i f3_int1 = _mm_srli_epi32(product1, 16); //Shifts the 4 signed or unsigned 32 - bit integers in a right by count bits while shifting in zeros.
					const __m128i f3_int2 = _mm_srli_epi32(product2, 16); //Shifts the 4 signed or unsigned 32 - bit integers in a right by count bits while shifting in zeros.
					std::get<0>(swap)[block] = _mm_packs_epi32(f3_int1, f3_int2);
				}
			} else {
				_mm_rescale_epu16(swap);
			}

			{	// perform the swapping, cannot be done in a vectorized manner
				unsigned __int8 * const data_ptr = reinterpret_cast<unsigned __int8 * const>(std::get<0>(data));
				unsigned __int16 * const swap_array_int = reinterpret_cast<unsigned __int16 * const>(std::get<0>(swap));
				swapArray(data_ptr, swap_array_int, nElements);
			}
		}

		inline void _mm_permute_dp_array_ref(
			const std::tuple<__m128d * const, const size_t>& data,
			const size_t nElements,
			const std::tuple<__m128i * const, const size_t>& swap,
			__m128i& randInts)
		{
			_mm_lfsr32_epu32(swap, randInts);
			_mm_rescale_epu16_ref(swap);

			{	// perform the swapping, cannot be done in a vectorized manner
				__m128d * const tmp = std::get<0>(data);
				double * const data2 = reinterpret_cast<double * const>(tmp);
				unsigned __int16 * const swap_array_int = reinterpret_cast<unsigned __int16 * const>(std::get<0>(swap));
				swapArray(data2, swap_array_int, nElements);
			}
		}

		inline void _mm_permute_dp_array_method2(
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
				unsigned __int16 * const swap_array_int = reinterpret_cast<unsigned __int16 * const>(std::get<0>(swap));
				swapArray(data2, swap_array_int, nElements);
			}
		}
	}

	namespace test {
		void test_mm_permute_epu8_array(
			const size_t nBlocks, 
			const size_t nExperiments, 
			const bool doTests)
		{
			if ((nBlocks * 8) > 0xFFFF) {
				std::cout << "WARNING: t test_mm_permute_epu8: too many blocks=" << nBlocks << std::endl;
				return;
			}
			const size_t nBytes = 16 * nBlocks;
			auto data0_r = _mm_malloc_m128i(nBytes);
			auto data1 = _mm_malloc_m128i(nBytes);
			auto data2 = _mm_malloc_m128i(nBytes);
			auto data3 = _mm_malloc_m128i(nBytes);

			const size_t nElements = nBytes;
			const size_t swap_array_nBytes = nElements << 1;
			auto swap = _mm_malloc_m128i(swap_array_nBytes);

			const __m128i seed = _mm_set_epi32(rand() || rand() << 16, rand() || rand() << 16, rand() || rand() << 16, rand() || rand() << 16);
			__m128i randInt = seed;
			__m128i randInt1 = seed;
			__m128i randInt2 = seed;
			const int N_BITS = 5;

			fillRand_epu8<N_BITS>(data0_r);
			const std::tuple<const __m128i * const, const size_t> data0 = data0_r;


			double min_ref = std::numeric_limits<double>::max();
			double min1 = std::numeric_limits<double>::max();
			double min2 = std::numeric_limits<double>::max();

			for (size_t i = 0; i < nExperiments; ++i) {

				copy(data0, data1);
				timer::reset_and_start_timer();
				hli::priv::_mm_permute_epu8_array_ref(data1, nElements, swap, randInt);
				min_ref = std::min(min_ref, timer::get_elapsed_kcycles());

				{
					copy(data0, data2);
					timer::reset_and_start_timer();
					hli::priv::_mm_permute_epu8_array_method1(data2, nElements, swap, randInt1);
					min1 = std::min(min1, timer::get_elapsed_kcycles());

					if (doTests) {
						for (size_t block = 0; block < nBlocks; ++block) {
							if (!equal(std::get<0>(data1)[block], std::get<0>(data2)[block])) {
								std::cout << "WARNING: test_mm_permute_epu8: result-ref=" << hli::toString_u16(std::get<0>(data1)[block]) << "; result1=" << hli::toString_u16(std::get<0>(data2)[block]) << std::endl;
								return;
							}
						}
						if (!equal(randInt, randInt1)) {
							std::cout << "WARNING: test_mm_permute_epu8: randInt=" << hli::toString_u32(randInt) << "; randInt1=" << hli::toString_u32(randInt1) << std::endl;
							return;
						}
					}
				}
				{
					copy(data0, data3);
					timer::reset_and_start_timer();
					hli::priv::_mm_permute_epu8_array_method2(data3, nElements, swap, randInt2);
					min2 = std::min(min2, timer::get_elapsed_kcycles());

					if (doTests) {
						for (size_t block = 0; block < nBlocks; ++block) {
							if (!equal(std::get<0>(data1)[block], std::get<0>(data2)[block])) {
								std::cout << "WARNING: test_mm_permute_epu8: result-ref=" << hli::toString_u16(std::get<0>(data1)[block]) << "; result2=" << hli::toString_u16(std::get<0>(data3)[block]) << std::endl;
								return;
							}
						}
						if (!equal(randInt, randInt2)) {
							std::cout << "WARNING: test_mm_permute_epu8: randInt=" << hli::toString_u32(randInt) << "; randInt2=" << hli::toString_u32(randInt2) << std::endl;
							return;
						}
					}
				}
			}
			if (doTests)
			{
				const __m128i sum1 = hli::_mm_hadd_epu8<N_BITS>(data1, nElements);
				const __m128i sum2 = hli::_mm_hadd_epu8<N_BITS>(data2, nElements);
				const __m128i sum3 = hli::_mm_hadd_epu8<N_BITS>(data3, nElements);
				if (sum1.m128i_u32[0] != sum2.m128i_u32[0]) {
					std::cout << "WARNING: test_mm_permute_epu8: sums are unequal: sum1=" << sum1.m128i_u32[0] << "; sum2=" << sum2.m128i_u32[0] << std::endl;
				}
				if (sum1.m128i_u32[0] != sum3.m128i_u32[0]) {
					std::cout << "WARNING: test_mm_permute_epu8: sums are unequal: sum1=" << sum1.m128i_u32[0] << "; sum3=" << sum3.m128i_u32[0] << std::endl;
				}
			}

			printf("[_mm_permute_epu8 Ref]    : %2.5f Kcycles\n", min_ref);
			printf("[_mm_permute_epu8 method1]: %2.5f Kcycles; %2.3f times faster than ref\n", min1, min_ref / min1);
			printf("[_mm_permute_epu8 method2]: %2.5f Kcycles; %2.3f times faster than ref\n", min2, min_ref / min2);

			_mm_free2(data0);
			_mm_free2(data1);
			_mm_free2(data2);
			_mm_free2(data3);
			_mm_free2(swap);
		}
		void test_mm_permute_dp_array(
			const size_t nBlocks, 
			const size_t nExperiments, 
			const bool doTests)
		{
			if ((nBlocks * 8) > 0xFFFF) {
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
				hli::priv::_mm_permute_dp_array_ref(data_1, nElements, swap, randInt);
				min_ref = std::min(min_ref, timer::get_elapsed_kcycles());

				{
					memcpy(std::get<0>(data_2), std::get<0>(data), nBytes);
					timer::reset_and_start_timer();
					hli::priv::_mm_permute_dp_array_method2(data_2, nElements, swap, randInt1);
					min1 = std::min(min1, timer::get_elapsed_kcycles());

					if (doTests) {
						for (size_t block = 0; block < nBlocks; ++block) {
							if (!equal(std::get<0>(data_1)[block], std::get<0>(data_2)[block])) {
								std::cout << "WARNING: test_mm_permute_epu8: result-ref=" << hli::toString_f64(std::get<0>(data_1)[block]) << "; result1=" << hli::toString_f64(std::get<0>(data_2)[block]) << std::endl;
								return;
							}
						}
						if (!equal(randInt, randInt1)) {
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

			printf("[_mm_permute_dp_array_ref]    : %2.5f Kcycles\n", min_ref);
			printf("[_mm_permute_dp_array_method1]: %2.5f Kcycles; %2.3f times faster than ref\n", min1, min_ref / min1);

			_mm_free2(data);
			_mm_free2(data_1);
			_mm_free2(data_2);
			_mm_free2(swap);
		}
	}

	inline void _mm_permute_epu8_array(
		const std::tuple<__m128i * const, const size_t>& data,
		const size_t nElements,
		const std::tuple<__m128i * const, const size_t>& swap,
		__m128i& randInts)
	{
		//std::cout << "INFO: _mm_permute_array::_mm_permute_epu8_array: nBytes=" << nBytes << std::endl;
		//priv::_mm_permute_epu8_array_ref(data, swap, randInts);
		//priv::_mm_permute_epu8_array_method1(data, swap, randInts);
		priv::_mm_permute_epu8_array_method2(data, nElements, swap, randInts);
	}

	inline void _mm_permute_dp_array(
		const std::tuple<__m128d * const, const size_t> data,
		const size_t nElements,
		const std::tuple<__m128i * const, const size_t> swap,
		__m128i& randInts)
	{
		//std::cout << "INFO: _mm_permute_array::_mm_permute_dp_array: nBytes=" << nBytes << std::endl;
		//priv::_mm_permute_dp_array_ref(mem_addr, nBytes, randInts, swap_array_nBytes, swap_array);
		priv::_mm_permute_dp_array_method2(data, nElements, swap, randInts);
	}
}