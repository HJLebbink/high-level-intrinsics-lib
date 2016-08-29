#pragma once

//#include "mmintrin.h"  // mmx
#include "emmintrin.h"  // sse
#include "pmmintrin.h"  // sse3
#include "tmmintrin.h"  // ssse3
#include "smmintrin.h"  // sse4.1
#include "nmmintrin.h"  // sse4.2
#include "immintrin.h"  // avx, avx2, avx512, FP16C, KNCNI, FMA
//#include "ammintrin.h"  // AMD-specific intrinsics

#include "../hli-lib/toString.h"

namespace hli {

	// Horizontally add adjacent pairs of 64-bit integers in a, store the result 4 times in dst.
	// Operation:
	// tmp := a[63:0] + a[127:64] + b[63:0] + b[127:64]
	// dst[63:0] := tmp
	// dst[127:64] := tmp
	// dst[191:128] := tmp
	// dst[255:192] := tmp
	inline __m256i _mm256_hadd_epi64(__m128i a, __m128i b)
	{
		const __m128i sum_a = _mm_add_epi64(a, b);
		const __m128i sum_b = _mm_castpd_si128(_mm_permute_pd(_mm_castsi128_pd(sum_a), 0b01));
		const __m128i sum = _mm_add_epi64(sum_a, sum_b);

		const __m256i result = _mm256_insertf128_si256(_mm256_castsi128_si256(sum), sum, 1);
		//std::cout << "INFO: hli::_mm256_hadd_epi64: sum_a=" << toString_u64(sum_a) << "; sum_b=" << toString_u64(sum_b) << "; result=" << toString_u64(result) << std::endl;
		return result;
	}

	// Horizontally add all 8-bit integers in mem_addr (with nBytes).
	// Operation:
	// dst[63:0] := sum(mem_addr)
	// dst[127:64] := sum(mem_addr)
	// dst[191:128] := sum(mem_addr)
	// dst[255:192] := sum(mem_addr)
	template <int N_BITS = 8>
	inline __m256i _mm256_hadd_epu8(
		const __m256i * const mem_addr,
		const size_t nBytes)
	{
		return priv::_mm256_hadd_epu8_class<N_BITS>::_mm256_hadd_epu8(mem_addr, nBytes);
	}

	namespace priv {

		inline __m256i _mm256_hadd_epu8_ref(
			const __m256i * const mem_addr,
			const size_t nBytes)
		{
			const unsigned __int8 * const ptr = reinterpret_cast<const unsigned __int8 * const>(mem_addr);
			unsigned __int64 sum = 0;
			for (size_t i = 0; i < nBytes; ++i) {
				sum += ptr[i];
				//std::cout << "INFO: hli::ref::_mm_hadd_epi8: i=" << i << "; sum="<<sum << std::endl;
			}
			return _mm256_set1_epi64x(sum);
		}

		template <int N_BITS>
		class _mm256_hadd_epu8_class {
			static_assert(((N_BITS > 0) && (N_BITS <= 8)), "number of bits must be larger than 0 and smaller or equal than 8");
		};

		template <>
		class _mm256_hadd_epu8_class < 8 > {
		public:
			inline static __m256i _mm256_hadd_epu8(
				const __m256i * const mem_addr,
				const size_t nBytes)
			{
				const size_t nBlocks = nBytes >> 5; // divide by 32 to get the number of __m256i regs (blocks)
				__m128i sum1 = _mm_setzero_si128();
				__m128i sum2 = _mm_setzero_si128();

				for (size_t block = 0; block < nBlocks; ++block) {
					const __m256i d = _mm256_load_si256(&mem_addr[block]);
					sum1 = _mm_add_epi64(sum1, _mm_sad_epu8(_mm256_extractf128_si256(d, 0), _mm_setzero_si128()));
					sum2 = _mm_add_epi64(sum2, _mm_sad_epu8(_mm256_extractf128_si256(d, 1), _mm_setzero_si128()));
				}
				return _mm256_hadd_epi64(sum1, sum2);
			}
		};
	}
}