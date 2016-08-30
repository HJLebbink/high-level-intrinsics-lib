#pragma once

#include <iostream>		// for cout

//#include "mmintrin.h"  // mmx
#include "emmintrin.h"  // sse
#include "pmmintrin.h"  // sse3
#include "tmmintrin.h"  // ssse3
#include "smmintrin.h"  // sse4.1
#include "nmmintrin.h"  // sse4.2
#include "immintrin.h"  // avx, avx2, avx512, FP16C, KNCNI, FMA
//#include "ammintrin.h"  // AMD-specific intrinsics

#include "toString.h"

namespace hli {

	namespace priv {

		// Galois LFSR 32; see http://en.wikipedia.org/wiki/Linear_feedback_shift_register
		inline unsigned int lfsr32_galois(const unsigned int i)
		{
			//return (i >> 1) ^ (-(signed int)(i & 1u) & 0xD0000001u); // The cast to signed int is needed to prevent the warning that unairy minus on unsigned ints yields unsigned ints.

			unsigned int j = i;
			unsigned int lsb = static_cast<unsigned int>(-static_cast<signed int>(j & 1));   /* Get LSB (i.e., the output bit). */
			j >>= 1;                /* Shift register */
			j ^= lsb & 0xD0000001u;
			return j;
		}

		inline __m128i lfsr32_galois(const __m128i i)
		{
			const __m128i i1 = _mm_and_si128(i, _mm_set1_epi32(1u));
			const __m128i tmp = _mm_set1_epi32(-1);
			const __m128i i2 = _mm_sign_epi32(i1, tmp);
			const __m128i i3 = _mm_and_si128(i2, _mm_set1_epi32(0xD0000001u));
			const __m128i i4 = _mm_srli_epi32(i, 1);
			return _mm_xor_si128(i4, i3);
		}

		inline __m128i lfsr16_galois(const __m128i i)
		{
			const __m128i i1 = _mm_and_si128(i, _mm_set1_epi16(1u));
			const __m128i tmp = _mm_set1_epi16(-1);
			const __m128i i2 = _mm_sign_epi16(i1, tmp);
			const __m128i i3 = _mm_and_si128(i2, _mm_set1_epi16(0xB400u));
			const __m128i i4 = _mm_srli_epi16(i, 1);
			return _mm_xor_si128(i4, i3);
		}

		inline unsigned int nextRandInt(const unsigned int i)
		{
			//return rdrand32();
			return lfsr32_galois(i);
		}

		inline void _mm_rand_si128_ref(
			__m128i * const mem_addr,
			const size_t nBytes,
			__m128i& randInts)
		{
			unsigned int r0 = randInts.m128i_u32[0];
			unsigned int r1 = randInts.m128i_u32[1];
			unsigned int r2 = randInts.m128i_u32[2];
			unsigned int r3 = randInts.m128i_u32[3];

			const size_t nBlocks = nBytes >> 4;
			for (size_t block = 0; block < nBlocks; ++block) {
				r0 = nextRandInt(r0);
				r1 = nextRandInt(r1);
				r2 = nextRandInt(r2);
				r3 = nextRandInt(r3);

				mem_addr[block].m128i_u32[0] = r0;
				mem_addr[block].m128i_u32[1] = r1;
				mem_addr[block].m128i_u32[2] = r2;
				mem_addr[block].m128i_u32[3] = r3;
			}

			randInts.m128i_u32[0] = r0;
			randInts.m128i_u32[1] = r1;
			randInts.m128i_u32[2] = r2;
			randInts.m128i_u32[3] = r3;
		}

		inline void _mm_rescale_epu16_ref(
			__m128i * const mem_addr,
			const size_t nBytes)
		{
			/*
			const size_t length = vector.size();
			const bool showInfo = false;

			for (size_t i = 0; i < length; ++i) {
			//std::cout << "rescaleVector: before: i=" << i << ":" << ptr[i] << std::endl;

			const int original = vector[i];
			const int product = original*static_cast<int>(i + 1);
			const int f3 = product >> 16;
			if (showInfo) std::cout << "rescaleVector_reference: i=" << i << "; original " << original << "; product " << product << "; f3 " << f3 << std::endl;

			vector[i] = static_cast<unsigned short>(f3);
			//std::cout << "rescaleVector: after: i=" << i << ":" << ptr[i] << std::endl;
			}
			*/

			const size_t length = nBytes >> 1;
			const bool showInfo = false;
			unsigned __int16 * const ptr = reinterpret_cast<unsigned __int16 * const>(mem_addr);

			for (size_t i = 0; i < length; ++i) {
				if (showInfo) std::cout << "rescaleVector: before: i=" << i << ":" << ptr[i] << std::endl;

				const unsigned __int16 original = ptr[i];
				const unsigned int product = original*static_cast<unsigned int>(i + 1);
				const unsigned int f3 = product >> 16;
				if (showInfo) std::cout << "rescaleVector_reference: i=" << i << "; original " << original << "; product " << product << "; f3 " << f3 << std::endl;

				ptr[i] = static_cast<unsigned __int16>(f3);
				//std::cout << "rescaleVector: after: i=" << i << ":" << ptr[i] << std::endl;
			}
		}

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
	}

	inline void _mm_rescale_epu16(
		__m128i * const mem_addr,
		const size_t nBytes)
	{
		priv::_mm_rescale_epu16_ref(mem_addr, nBytes);
	}

	inline void _mm_permute_epu8(
		__m128i * const mem_addr,
		const size_t nBytes,
		__m128i& randInts)
	{
		priv::_mm_permute_epu8_ref(mem_addr, nBytes, randInts);
	}

	inline void _mm_rand_si128(
		__m128i * const mem_addr,
		const size_t nBytes,
		__m128i& randInts)
	{
		const size_t nBlocks = nBytes >> 4;
		for (size_t block = 0; block < nBlocks; ++block) {
			randInts = priv::lfsr32_galois(randInts);
			mem_addr[block] = randInts;
		}
	}

}