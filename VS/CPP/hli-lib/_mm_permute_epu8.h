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
			__int8 * const data, 
			const size_t pos1, 
			const __int16 pos2)
		{
			const __int8 temp = data[pos1];
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

			__int8 * const data = reinterpret_cast<__int8 * const>(mem_addr);
			__int16 * const swap_array_int = reinterpret_cast<__int16 * const>(swap_array);

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

			__int8 * const data = reinterpret_cast<__int8 * const>(mem_addr);
			__int16 * const swap_array_int = reinterpret_cast<__int16 * const>(swap_array);

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
			_mm_rand_si128(swap_array, swap_array_size, randInts);
			_mm_rescale_epu16(swap_array, swap_array_size);

			__int8 * const data = reinterpret_cast<__int8 * const>(mem_addr);
			__int16 * const swap_array_int = reinterpret_cast<__int16 * const>(swap_array);

			for (size_t i = nBytes - 1; i > 0; --i) {
				swapElement(data, i, swap_array_int[i]);
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