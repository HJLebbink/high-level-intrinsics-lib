#pragma once

#include <iostream>		// for cout
#include <math.h>
#include <algorithm>	// for std::max


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
			for (int i = nElements - 1; i > 0; --i) {
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
			__m128i * const mem_addr,
			const size_t nBytes,
			__m128i& randInts)
		{
			const size_t nElements = nBytes;
			const size_t swap_array_nBytes = resizeNBytes(nElements << 1, 16);
			__m128i * const swap_array = static_cast<__m128i * const>(_mm_malloc(swap_array_nBytes, 16));
			_mm_rand_si128_ref(swap_array, swap_array_nBytes, randInts);
			_mm_rescale_epu16_ref(swap_array, swap_array_nBytes);

			unsigned __int8 * const data = reinterpret_cast<unsigned __int8 * const>(mem_addr);
			unsigned __int16 * const swap_array_int = reinterpret_cast<unsigned __int16 * const>(swap_array);

			swapArray(data, swap_array_int, nElements);
			_mm_free(swap_array);
		}

		inline void _mm_permute_epu8_array_method1(
			__m128i * const mem_addr,
			const size_t nBytes,
			__m128i& randInts)
		{
			const size_t nElements = nBytes;
			const size_t swap_array_nBytes = resizeNBytes(nElements << 1, 16);
			__m128i * const swap_array = static_cast<__m128i * const>(_mm_malloc(swap_array_nBytes, 16));
			_mm_lfsr32_epu32(swap_array, swap_array_nBytes, randInts);
			_mm_rescale_epu16(swap_array, swap_array_nBytes);

			unsigned __int8 * const data = reinterpret_cast<unsigned __int8 * const>(mem_addr);
			unsigned __int16 * const swap_array_int = reinterpret_cast<unsigned __int16 * const>(swap_array);

			swapArray(data, swap_array_int, nElements);
			_mm_free(swap_array);
		}

		inline void _mm_permute_epu8_array_method2(
			__m128i * const mem_addr,
			const size_t nBytes,
			__m128i& randInts)
		{
			const size_t nElements = nBytes;
			//std::cout << "INFO: _mm_permute_array::_mm_permute_epu8_array_method2: nElements=" << nElements << std::endl;
			const size_t swap_array_nBytes = resizeNBytes(nElements << 1, 16);
			__m128i * const swap_array = static_cast<__m128i * const>(_mm_malloc(swap_array_nBytes, 16));

			const size_t nBlocks = nBytes >> 4;

			if (false) {
				for (size_t block = 0; block < nBlocks; ++block) {
					randInts = priv::lfsr32_galois(randInts);
					swap_array[block] = randInts;
				}
			} else {
				_mm_lfsr32_epu32(swap_array, swap_array_nBytes, randInts);
			}
			if (false) {
				const __m128i increment = _mm_set1_epi32(4);
				__m128i m = _mm_set_epi32(4, 3, 2, 1);
				for (size_t block = 0; block < nBlocks; ++block) {
					const __m128i data = swap_array[block];
					const __m128i numbers1 = _mm_cvtepu16_epi32(data);
					const __m128i numbers2 = _mm_cvtepu16_epi32(_mm_shuffle_epi32(data, 0b11101110));
					const __m128i product1 = _mm_mullo_epi32(numbers1, m);
					m = _mm_add_epi32(m, increment);
					const __m128i product2 = _mm_mullo_epi32(numbers2, m);
					m = _mm_add_epi32(m, increment);
					const __m128i f3_int1 = _mm_srli_epi32(product1, 16); //Shifts the 4 signed or unsigned 32 - bit integers in a right by count bits while shifting in zeros.
					const __m128i f3_int2 = _mm_srli_epi32(product2, 16); //Shifts the 4 signed or unsigned 32 - bit integers in a right by count bits while shifting in zeros.
					swap_array[block] = _mm_packs_epi32(f3_int1, f3_int2);

				}
			} else {
				_mm_rescale_epu16(swap_array, swap_array_nBytes);
			}

			{	// perform the swapping, cannot be done in a vectorized manner
				unsigned __int8 * const data = reinterpret_cast<unsigned __int8 * const>(mem_addr);
				unsigned __int16 * const swap_array_int = reinterpret_cast<unsigned __int16 * const>(swap_array);
				swapArray(data, swap_array_int, nElements);
			}
			_mm_free(swap_array);
		}

		inline void _mm_permute_dp_array_ref(
			__m128d * const mem_addr,
			const size_t nBytes,
			__m128i& randInts)
		{
			const size_t nElements = nBytes >> 3;
			//std::cout << "INFO: _mm_permute_array::_mm_permute_dp_array_ref: nElements=" << nElements << std::endl;
			const size_t swap_array_nBytes = resizeNBytes(nElements << 1, 16);
			__m128i * const swap_array = static_cast<__m128i * const>(_mm_malloc(swap_array_nBytes, 16));

			_mm_lfsr32_epu32(swap_array, swap_array_nBytes, randInts);
			_mm_rescale_epu16(swap_array, swap_array_nBytes);

			{	// perform the swapping, cannot be done in a vectorized manner
				double * const data = reinterpret_cast<double * const>(mem_addr);
				unsigned __int16 * const swap_array_int = reinterpret_cast<unsigned __int16 * const>(swap_array);
				swapArray(data, swap_array_int, nElements);
			}
			_mm_free(swap_array);
		}

		inline void _mm_permute_dp_array_method2(
			__m128d * const mem_addr,
			const size_t nBytes,
			__m128i& randInts)
		{
			_mm_permute_dp_array_ref(mem_addr, nBytes, randInts);
		}
	}

	namespace test {
		void test_mm_permute_epu8_array(const size_t nBlocks, const size_t nExperiments, const bool doTests)
		{
			if ((nBlocks * 8) > 0xFFFF) {
				std::cout << "WARNING: t test_mm_permute_epu8: too many blocks=" << nBlocks << std::endl;
				return;
			}

			const size_t nBytes = 16 * nBlocks;
			__m128i * const mem_addr = static_cast<__m128i *>(_mm_malloc(nBytes, 16));
			__m128i * const mem_addr1 = static_cast<__m128i *>(_mm_malloc(nBytes, 16));
			__m128i * const mem_addr2 = static_cast<__m128i *>(_mm_malloc(nBytes, 16));
			__m128i * const mem_addr3 = static_cast<__m128i *>(_mm_malloc(nBytes, 16));

			const __m128i seed = _mm_set_epi32(rand(), rand(), rand(), rand());
			__m128i randInt = seed;
			__m128i randInt1 = seed;
			__m128i randInt2 = seed;
			const int N_BITS = 5;

			fillRand_epu8<N_BITS>(mem_addr, nBytes);

			double min_ref = std::numeric_limits<double>::max();
			double min1 = std::numeric_limits<double>::max();
			double min2 = std::numeric_limits<double>::max();

			for (size_t i = 0; i < nExperiments; ++i) {

				memcpy(mem_addr1, mem_addr, nBytes);
				timer::reset_and_start_timer();
				hli::priv::_mm_permute_epu8_array_ref(mem_addr1, nBytes, randInt);
				min_ref = std::min(min_ref, timer::get_elapsed_kcycles());

				{
					memcpy(mem_addr2, mem_addr, nBytes);
					timer::reset_and_start_timer();
					hli::priv::_mm_permute_epu8_array_method1(mem_addr2, nBytes, randInt1);
					min1 = std::min(min1, timer::get_elapsed_kcycles());

					if (doTests) {
						for (size_t block = 0; block < nBlocks; ++block) {
							for (size_t j = 0; j < 8; ++j) {
								if (std::abs(mem_addr1[block].m128i_u16[j] != mem_addr2[block].m128i_u16[j])) {
									std::cout << "WARNING: test_mm_permute_epu8: result-ref=" << hli::toString_u16(mem_addr1[block]) << "; result1=" << hli::toString_u16(mem_addr2[block]) << std::endl;
									return;
								}
							}
						}
						if (!equal(randInt, randInt1)) {
							std::cout << "WARNING: test_mm_permute_epu8: randInt=" << hli::toString_u32(randInt) << "; randInt1=" << hli::toString_u32(randInt1) << std::endl;
							return;
						}
					}
				}
				{
					memcpy(mem_addr3, mem_addr, nBytes);
					timer::reset_and_start_timer();
					hli::priv::_mm_permute_epu8_array_method2(mem_addr3, nBytes, randInt2);
					min2 = std::min(min2, timer::get_elapsed_kcycles());

					if (doTests) {
						for (size_t block = 0; block < nBlocks; ++block) {
							for (size_t j = 0; j < 8; ++j) {
								if (std::abs(mem_addr1[block].m128i_u16[j] != mem_addr3[block].m128i_u16[j])) {
									std::cout << "WARNING: test_mm_permute_epu8: result-ref=" << hli::toString_u16(mem_addr1[block]) << "; result2=" << hli::toString_u16(mem_addr3[block]) << std::endl;
									return;
								}
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
				const __m128i sum1 = hli::_mm_hadd_epu8<N_BITS>(mem_addr1, nBytes);
				const __m128i sum2 = hli::_mm_hadd_epu8<N_BITS>(mem_addr2, nBytes);
				const __m128i sum3 = hli::_mm_hadd_epu8<N_BITS>(mem_addr3, nBytes);
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

			_mm_free(mem_addr);
			_mm_free(mem_addr1);
			_mm_free(mem_addr2);
			_mm_free(mem_addr3);
		}
		void test_mm_permute_dp_array(const size_t nBlocks, const size_t nExperiments, const bool doTests)
		{
			if ((nBlocks * 8) > 0xFFFF) {
				std::cout << "WARNING: t test_mm_permute_epu8: too many blocks=" << nBlocks << std::endl;
				return;
			}

			const size_t nBytes = 16 * nBlocks;
			__m128d * const mem_addr = static_cast<__m128d *>(_mm_malloc(nBytes, 16));
			__m128d * const mem_addr1 = static_cast<__m128d *>(_mm_malloc(nBytes, 16));
			__m128d * const mem_addr2 = static_cast<__m128d *>(_mm_malloc(nBytes, 16));

			const __m128i seed = _mm_set_epi32(rand(), rand(), rand(), rand());
			__m128i randInt = seed;
			__m128i randInt1 = seed;
			const int N_BITS = 5;

			fillRand_pd(mem_addr, nBytes);

			double min_ref = std::numeric_limits<double>::max();
			double min1 = std::numeric_limits<double>::max();

			for (size_t i = 0; i < nExperiments; ++i) 
			{
				memcpy(mem_addr1, mem_addr, nBytes);
				timer::reset_and_start_timer();
				hli::priv::_mm_permute_dp_array_ref(mem_addr1, nBytes, randInt);
				min_ref = std::min(min_ref, timer::get_elapsed_kcycles());

				{
					memcpy(mem_addr2, mem_addr, nBytes);
					timer::reset_and_start_timer();
					hli::priv::_mm_permute_dp_array_method2(mem_addr2, nBytes, randInt1);
					min1 = std::min(min1, timer::get_elapsed_kcycles());

					if (doTests) {
						for (size_t block = 0; block < nBlocks; ++block) {
							for (size_t j = 0; j < 8; ++j) {
								if (!equal(mem_addr1[block], mem_addr2[block])) {
									std::cout << "WARNING: test_mm_permute_epu8: result-ref=" << hli::toString_f64(mem_addr1[block]) << "; result1=" << hli::toString_f64(mem_addr2[block]) << std::endl;
									return;
								}
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

			_mm_free(mem_addr);
			_mm_free(mem_addr1);
			_mm_free(mem_addr2);
		}
	}

	inline void _mm_permute_epu8_array(
		__m128i * const mem_addr,
		const size_t nBytes,
		__m128i& randInts)
	{
		//std::cout << "INFO: _mm_permute_array::_mm_permute_epu8_array: nBytes=" << nBytes << std::endl;
		//priv::_mm_permute_epu8_array_ref(mem_addr, nBytes, randInts);
		//priv::_mm_permute_epu8_array_method1(mem_addr, nBytes, randInts);
		priv::_mm_permute_epu8_array_method2(mem_addr, nBytes, randInts);
	}

	inline void _mm_permute_dp_array(
		__m128d * const mem_addr,
		const size_t nBytes,
		__m128i& randInts)
	{
		//std::cout << "INFO: _mm_permute_array::_mm_permute_dp_array: nBytes=" << nBytes << std::endl;
		//priv::_mm_permute_dp_array_ref(mem_addr, nBytes, randInts);
		priv::_mm_permute_dp_array_method2(mem_addr, nBytes, randInts);
	}
}