#pragma once
#include "emmintrin.h"  // sse

namespace hli {
	namespace ref {

		inline __m128i _mm_hadd_epi8(
			const __m128i * const mem_addr,
			const size_t nBytes)
		{
			const __int8 * const ptr = reinterpret_cast<const __int8 * const>(mem_addr);
			__int64 sum = 0;
			for (size_t i = 0; i < nBytes; ++i) {
				sum += ptr[i];
				//std::cout << "INFO: hli::ref::_mm_hadd_epi8: i=" << i << "; sum="<<sum << std::endl;
			}
			return _mm_set1_epi64x(sum);
		}

		inline __m128i _mm_hadd_epu8(
			const __m128i * const mem_addr,
			const size_t nBytes)
		{
			const unsigned __int8 * const ptr = reinterpret_cast<const unsigned __int8 * const>(mem_addr);
			unsigned __int64 sum = 0;
			for (size_t i = 0; i < nBytes; ++i) {
				sum += ptr[i];
				//std::cout << "INFO: hli::ref::_mm_hadd_epi8: i=" << i << "; sum="<<sum << std::endl;
			}
			return _mm_set1_epi64x(sum);
		}
	}
}