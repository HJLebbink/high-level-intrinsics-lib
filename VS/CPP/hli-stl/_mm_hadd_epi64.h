#pragma once

#include <algorithm>	// std::min
#include <limits>		// std::numeric_limits
#include <iostream>		// std::cout
#include <iostream>		// for cout

//#include "mmintrin.h"  // mmx
#include "emmintrin.h"  // sse
#include "pmmintrin.h"  // sse3
#include "tmmintrin.h"  // ssse3
#include "smmintrin.h"  // sse4.1
#include "nmmintrin.h"  // sse4.2
#include "immintrin.h"  // avx, avx2, avx512, FP16C, KNCNI, FMA
//#include "ammintrin.h"  // AMD-specific intrinsics

#include "tools.h"
#include "toString.h"


namespace hli {

	// Horizontally add adjacent pairs of 64-bit integers in a, store the result in dst.
	// Operation:
	// tmp := a[63:0] + a[127:64]
	// dst[31:0] := tmp
	// dst[63:32] := tmp
	// dst[95:64] := tmp
	// dst[127:96] := tmp
	inline __m128i _mm_hadd_epi64(__m128i a)
	{
		__m128i b = _mm_shuffle_epi32(a, _MM_SHUFFLE_EPI32_INT(0, 0, 0, 0));
		__m128i c = _mm_shuffle_epi32(a, _MM_SHUFFLE_EPI32_INT(2, 2, 2, 2));
		__m128i d = _mm_add_epi32(b, c);
		//std::cout << "INFO: hli:::_mm_hadd_epi64: a=" << toString_u32(a) << "; b=" << toString_u32(b) << "; c=" << toString_u32(c) << "; d=" << toString_u32(d) << std::endl;
		return d;
	}
}
