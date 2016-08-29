#pragma once

//#include "mmintrin.h"  // mmx
#include "emmintrin.h"  // sse
#include "pmmintrin.h"  // sse3
#include "tmmintrin.h"  // ssse3
#include "smmintrin.h"  // sse4.1
#include "nmmintrin.h"  // sse4.2
//#include "immintrin.h"  // avx, avx2, avx512, FP16C, KNCNI, FMA
//#include "ammintrin.h"  // AMD-specific intrinsics

namespace hli {

	// Horizontally add adjacent pairs of 64-bit integers in a, store the result twice in dst.
	// Operation:
	// dst[63:0] := a[63:0] + a[127:64]
	// dst[127:64] := a[63:0] + a[127:64]
	inline __m128i _mm_hadd_epi64(__m128i a)
	{
		const __m128i sum_up = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(a)));
		const __m128i sum_lo = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(a)));
		return _mm_add_epi64(sum_lo, sum_up);
	}

	// Horizontally add all 8-bit integers in mem_addr (with nBytes).
	// Operation:
	// dst[63:0] := sum(mem_addr)
	// dst[127:64] := sum(mem_addr)
	template <int N_BITS = 8>
	inline __m128i _mm_hadd_epu8(
		const __m128i * const mem_addr,
		const size_t nBytes)
	{
		return priv::_mm_hadd_epu8_class<N_BITS>::_mm_hadd_epu8(mem_addr, nBytes);
	}

	namespace priv {

		inline __m128i _mm_hadd_epu8_ref(
			const __m128i * const mem_addr,
			const size_t nBytes)
		{
			const unsigned __int8 * const ptr = reinterpret_cast<const unsigned __int8 * const>(mem_addr);
			unsigned __int64 sum = 0;
			for (size_t i = 0; i < nBytes; ++i) {
				sum += ptr[i];
				//std::cout << "INFO: hli::priv::_mm_hadd_epu8_ref: i=" << i << "; sum="<<sum << std::endl;
			}
			return _mm_set1_epi64x(sum);
		}

		inline __m128i _mm_hadd_epu8_method2(
			const __m128i * const mem_addr,
			const size_t nBytes)
		{
			//assume (nBytes < 2 ^ (32 - 8))

			const size_t nBlocks = nBytes >> 4; // divide by 16 to get the number of __m128i regs (blocks)
			const __m128i and_mask = _mm_set1_epi32(0b11111111);
			//const __m128i shuffle_mask_0 = _mm_set_epi8(15, 14, 13, 12,   11, 10, 9, 8,   7, 6, 5, 4,   3, 2, 1, 0);
			const __m128i shuffle_mask_1 = _mm_set_epi8(15, 14, 13, 13, 11, 10, 9, 9, 7, 6, 5, 5, 3, 2, 1, 1);
			const __m128i shuffle_mask_2 = _mm_set_epi8(15, 14, 13, 14, 11, 10, 9, 10, 7, 6, 5, 6, 3, 2, 1, 2);
			const __m128i shuffle_mask_3 = _mm_set_epi8(15, 14, 13, 15, 11, 10, 9, 11, 7, 6, 5, 7, 3, 2, 1, 3);

			__m128i sum = _mm_setzero_si128();

			for (size_t block = 0; block < nBlocks; ++block) {
				const __m128i d = _mm_load_si128(&mem_addr[block]);

				sum = _mm_add_epi32(sum, _mm_and_si128(d, and_mask));
				sum = _mm_add_epi32(sum, _mm_and_si128(_mm_shuffle_epi8(d, shuffle_mask_1), and_mask));
				sum = _mm_add_epi32(sum, _mm_and_si128(_mm_shuffle_epi8(d, shuffle_mask_2), and_mask));
				sum = _mm_add_epi32(sum, _mm_and_si128(_mm_shuffle_epi8(d, shuffle_mask_3), and_mask));
			}

			const __m128i sum2 = _mm_cvtepi32_epi64(_mm_hadd_epi32(sum, sum));
			return _mm_hadd_epi64(sum2);
		}

		template <int N_BITS>
		class _mm_hadd_epu8_class {
			static_assert(((N_BITS > 0) && (N_BITS <= 8)), "number of bits must be larger than 0 and smaller or equal than 8");
		};

		template <>
		class _mm_hadd_epu8_class < 8 > {
		public:
			inline static __m128i _mm_hadd_epu8(
				const __m128i * const mem_addr,
				const size_t nBytes)
			{
				const size_t nBlocks = nBytes >> 4; // divide by 16 to get the number of __m128i regs (blocks)
				__m128i sum = _mm_setzero_si128();

				for (size_t block = 0; block < nBlocks; ++block) {
					sum = _mm_add_epi64(sum, _mm_sad_epu8(_mm_load_si128(&mem_addr[block]), _mm_setzero_si128()));
				}

				return _mm_hadd_epi64(sum);
			}
		};

		template <>
		class _mm_hadd_epu8_class < 7 > {
		public:
			inline static __m128i _mm_hadd_epu8(
				const __m128i * const mem_addr,
				const size_t nBytes)
			{
				const size_t nBlocks = nBytes >> 4; // divide by 16 to get the number of __m128i regs (blocks)

				__m128i sum = _mm_setzero_si128();
				__m128i sum_p;

				for (size_t block = 0; block < nBlocks - 1; block += 2)
				{
					sum_p = _mm_load_si128(&mem_addr[block]);
					sum_p = _mm_add_epi8(sum_p, _mm_load_si128(&mem_addr[block + 1]));
					sum = _mm_add_epi64(sum, _mm_sad_epu8(sum_p, _mm_setzero_si128()));
				}

				const size_t tail = nBlocks & 0b1;
				if (tail > 0) {
					for (size_t block = (nBlocks - tail); block < nBlocks; ++block) {
						//std::cout << "INFO: hli::_mm_hadd_epu8_class<7>: tail=" << tail << "; block=" << block << std::endl;
						sum = _mm_add_epi64(sum, _mm_sad_epu8(_mm_load_si128(&mem_addr[block]), _mm_setzero_si128()));
					}
				}
				return _mm_hadd_epi64(sum);
			}
		};

		template <>
		class _mm_hadd_epu8_class < 6 > {
		public:
			inline static __m128i _mm_hadd_epu8(
				const __m128i * const mem_addr,
				const size_t nBytes)
			{
				const size_t nBlocks = nBytes >> 4; // divide by 16 to get the number of __m128i regs (blocks)

				__m128i sum = _mm_setzero_si128();
				__m128i sum_p;

				for (size_t block = 0; block < nBlocks - 3; block += 4)
				{
					sum_p = _mm_load_si128(&mem_addr[block]);
					sum_p = _mm_add_epi8(sum_p, _mm_load_si128(&mem_addr[block + 1]));
					sum_p = _mm_add_epi8(sum_p, _mm_load_si128(&mem_addr[block + 2]));
					sum_p = _mm_add_epi8(sum_p, _mm_load_si128(&mem_addr[block + 3]));
					sum = _mm_add_epi64(sum, _mm_sad_epu8(sum_p, _mm_setzero_si128()));
				}

				const size_t tail = nBlocks & 0b11;
				if (tail > 0) {
					for (size_t block = (nBlocks - tail); block < nBlocks; ++block) {
						//std::cout << "INFO: hli:::_mm_hadd_epu8_method3: tail: block=" << block << std::endl;
						sum = _mm_add_epi64(sum, _mm_sad_epu8(_mm_load_si128(&mem_addr[block]), _mm_setzero_si128()));
					}
				}
				return _mm_hadd_epi64(sum);
			}
		};

		template <>
		class _mm_hadd_epu8_class < 5 > {
		public:
			inline static __m128i _mm_hadd_epu8(
				const __m128i * const mem_addr,
				const size_t nBytes)
			{
				const size_t nBlocks = nBytes >> 4; // divide by 16 to get the number of __m128i regs (blocks)

				__m128i sum = _mm_setzero_si128();
				__m128i sum_p;

				for (size_t block = 0; block < nBlocks - 7; block += 8)
				{
					sum_p = _mm_load_si128(&mem_addr[block]);
					sum_p = _mm_add_epi8(sum_p, _mm_load_si128(&mem_addr[block + 1]));
					sum_p = _mm_add_epi8(sum_p, _mm_load_si128(&mem_addr[block + 2]));
					sum_p = _mm_add_epi8(sum_p, _mm_load_si128(&mem_addr[block + 3]));
					sum_p = _mm_add_epi8(sum_p, _mm_load_si128(&mem_addr[block + 4]));
					sum_p = _mm_add_epi8(sum_p, _mm_load_si128(&mem_addr[block + 5]));
					sum_p = _mm_add_epi8(sum_p, _mm_load_si128(&mem_addr[block + 6]));
					sum_p = _mm_add_epi8(sum_p, _mm_load_si128(&mem_addr[block + 7]));
					sum = _mm_add_epi64(sum, _mm_sad_epu8(sum_p, _mm_setzero_si128()));
				}

				const size_t tail = nBlocks & 0b111;
				if (tail > 0) {
					size_t startTail = nBlocks - tail;
					sum_p = _mm_load_si128(&mem_addr[startTail]);
					for (size_t block = startTail + 1; block < nBlocks; ++block) {
						sum_p = _mm_add_epi8(sum_p, _mm_load_si128(&mem_addr[block]));
					}
					sum = _mm_add_epi64(sum, _mm_sad_epu8(sum_p, _mm_setzero_si128()));
				}
				return _mm_hadd_epi64(sum);
			}
		};

	}
}
