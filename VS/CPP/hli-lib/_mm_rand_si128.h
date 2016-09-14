#pragma once

#include <iostream>		// for cout
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

namespace hli {

	namespace priv {
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
	}


	inline void _mm_lfsr32_epu32(
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

	template <int N_BITS>
	void fillRand_epu8(__int8 * const mem_addr, const size_t nBytes)
	{
		const int mask = (1 << N_BITS) - 1;
		for (size_t i = 0; i < nBytes; ++i) {
			mem_addr[i] = static_cast<__int8>(mask & rand());
		}
	}

	template <int N_BITS>
	void fillRand_epu8(__m128i * const mem_addr, const size_t nBytes)
	{
		fillRand_epu8<N_BITS>(reinterpret_cast<__int8 * const>(mem_addr), nBytes);
	}

	template <int N_BITS>
	void fillRand_epu8(__m256i * const mem_addr, const size_t nBytes)
	{
		fillRand_epu8<N_BITS>(reinterpret_cast<__int8 * const>(mem_addr), nBytes);
	}

	template <int N_BITS>
	void fillRand_epu8(const std::tuple<__m128i * const, const size_t>& data)
	{
		fillRand_epu8<N_BITS>(std::get<0>(data), std::get<1>(data));
	}


	void fillRand_pd(__m128d * const mem_addr, const size_t nBytes)
	{
		double * const ptr = reinterpret_cast<double * const>(mem_addr);
		const size_t nElements = nBytes >> 3;
		for (size_t i = 0; i < nElements; ++i) {
			ptr[i] = static_cast<double>(rand()) / rand();
		}
	}
	void fillRand_pd(std::tuple<__m128d * const, const size_t> data)
	{
		const size_t nBytes = std::get<1>(data);
		double * const ptr = reinterpret_cast<double * const>(std::get<0>(data));
		const size_t nElements = nBytes >> 3;
		for (size_t i = 0; i < nElements; ++i) {
			ptr[i] = static_cast<double>(rand()) / rand();
		}
	}
}