#pragma once
#include "emmintrin.h"  // sse

namespace hli {

	inline bool equal(const __m128i a, const __m128i b)
	{
		return (a.m128i_i64[0] == b.m128i_u64[0]) && (a.m128i_i64[1] == b.m128i_u64[1]);
	}

	inline bool equal(const __m128d a, const __m128d b)
	{
		return equal(_mm_castpd_si128(a), _mm_castpd_si128(b));
	}

	inline bool equal(const __m128 a, const __m128 b)
	{
		return equal(_mm_castps_si128(a), _mm_castps_si128(b));
	}
}