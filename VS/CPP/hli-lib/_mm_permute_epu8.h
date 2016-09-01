#pragma once

#include <iostream>		// for cout
#include "toString.h"

//#include "mmintrin.h"  // mmx
#include "emmintrin.h"  // sse
#include "pmmintrin.h"  // sse3
#include "tmmintrin.h"  // ssse3
#include "smmintrin.h"  // sse4.1
#include "nmmintrin.h"  // sse4.2
#include "immintrin.h"  // avx, avx2, avx512, FP16C, KNCNI, FMA
//#include "ammintrin.h"  // AMD-specific intrinsics

#include "_mm_rand_si128.h"
#include "_mm_rescale_epu16.h"

namespace hli {

	namespace priv {

		// swap the elements of data given pos1 and pos2
		inline void swapElement(
			unsigned __int8 * const data,
			const size_t pos1, 
			const unsigned __int16 pos2)
		{
			const unsigned __int8 temp = data[pos1];
			data[pos1] = data[pos2];
			data[pos2] = temp;
		}

		inline void _mm_permute_epu8_ref(
			__m128i * const mem_addr,
			const size_t nBytes,
			__m128i& randInts)
		{
			const size_t swap_array_size = nBytes << 1;
			__m128i * const swap_array = static_cast<__m128i * const>(_mm_malloc(swap_array_size, 16));
			_mm_rand_si128_ref(swap_array, swap_array_size, randInts);
			_mm_rescale_epu16_ref(swap_array, swap_array_size);

			unsigned __int8 * const data = reinterpret_cast<unsigned __int8 * const>(mem_addr);
			unsigned __int16 * const swap_array_int = reinterpret_cast<unsigned __int16 * const>(swap_array);

			for (size_t i = nBytes - 1; i > 0; --i) {
				swapElement(data, i, swap_array_int[i]);
			}
			_mm_free(swap_array);
		}

		inline void _mm_permute_epu8_method1(
			__m128i * const mem_addr,
			const size_t nBytes,
			__m128i& randInts)
		{
			const size_t swap_array_size = nBytes << 1;
			__m128i * const swap_array = static_cast<__m128i * const>(_mm_malloc(swap_array_size, 16));
			_mm_rand_si128(swap_array, swap_array_size, randInts);
			_mm_rescale_epu16(swap_array, swap_array_size);

			unsigned __int8 * const data = reinterpret_cast<unsigned __int8 * const>(mem_addr);
			unsigned __int16 * const swap_array_int = reinterpret_cast<unsigned __int16 * const>(swap_array);

			for (size_t i = nBytes - 1; i > 0; --i) {
				swapElement(data, i, swap_array_int[i]);
			}
			_mm_free(swap_array);
		}

		inline void _mm_permute_epu8_method2(
			__m128i * const mem_addr,
			const size_t nBytes,
			__m128i& randInts)
		{
			const size_t swap_array_size = nBytes << 1;
			__m128i * const swap_array = static_cast<__m128i * const>(_mm_malloc(swap_array_size, 16));

			const size_t nBlocks = nBytes >> 4;

/*
			for (size_t block = 0; block < nBlocks; ++block) {
				randInts = priv::lfsr32_galois(randInts);
				const __m128i data = randInts;

				const __m128i numbers1 = _mm_cvtepu16_epi32(data);
				const __m128i product1 = _mm_mullo_epi32(numbers1, m);
				const __m128i f3_int1 = _mm_srli_epi32(product1, 16); //Shifts the 4 signed or unsigned 32 - bit integers in a right by count bits while shifting in zeros.
				m = _mm_add_epi32(m, increment);

				const __m128i numbers2 = _mm_cvtepu16_epi32(_mm_shuffle_epi32(data, 0b11101110));
				const __m128i product2 = _mm_mullo_epi32(numbers2, m);
				const __m128i f3_int2 = _mm_srli_epi32(product2, 16); //Shifts the 4 signed or unsigned 32 - bit integers in a right by count bits while shifting in zeros.
				m = _mm_add_epi32(m, increment);

				const __m128i saturated = _mm_packs_epi32(f3_int1, f3_int2);

				//_mm_store_si128(&swap_array[block], saturated);
				swap_array[block] = saturated;
			}
*/
			if (false) {
				for (size_t block = 0; block < nBlocks; ++block) {
					randInts = priv::lfsr32_galois(randInts);
					swap_array[block] = randInts;
				}
			} else {
				_mm_rand_si128(swap_array, swap_array_size, randInts);
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
				_mm_rescale_epu16(swap_array, swap_array_size);
			}

			{	// perform the swapping, cannot be done in a vectorized manner
				unsigned __int8 * const data = reinterpret_cast<unsigned __int8 * const>(mem_addr);
				unsigned __int16 * const swap_array_int = reinterpret_cast<unsigned __int16 * const>(swap_array);

				for (size_t i = nBytes - 1; i > 0; --i) {
					swapElement(data, i, swap_array_int[i]);
				}
			}
			_mm_free(swap_array);
		}
	}

	inline void _mm_permute_epu8(
		__m128i * const mem_addr,
		const size_t nBytes,
		__m128i& randInts)
	{
		//priv::_mm_permute_epu8_ref(mem_addr, nBytes, randInts);
		//priv::_mm_permute_epu8_method1(mem_addr, nBytes, randInts);
		priv::_mm_permute_epu8_method2(mem_addr, nBytes, randInts);
	}

}