#pragma once

#include <tuple>

//#include "mmintrin.h"  // mmx
#include "emmintrin.h"  // sse
#include "pmmintrin.h"  // sse3
#include "tmmintrin.h"  // ssse3
#include "smmintrin.h"  // sse4.1
#include "nmmintrin.h"  // sse4.2
//#include "immintrin.h"  // avx, avx2, avx512, FP16C, KNCNI, FMA
//#include "ammintrin.h"  // AMD-specific intrinsics

#include "tools.h"

namespace hli {

	namespace priv {

		template <int N_BITS>
		inline __m128d _mm_mi_epu8_ref(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const std::tuple<const __m128i * const, const size_t>& data2)
		{
			const __m128d h1 = _mm_entropy_epu8<N_BITS>(data1);
			const __m128d h2 = _mm_entropy_epu8<N_BITS>(data2);
			const __m128d h1And2 = _mm_entropy_epu8<N_BITS>(data1, data2);
			const __m128d mi = _mm_sub_pd(h1Plush2, h1Andh2);
			return mi;
		}

		template <int N_BITS>
		inline std::tuple<__m128d * const, const size_t> _mm_mi_perm_epu8_ref(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const std::tuple<const __m128i * const, const size_t>& data2,
			const size_t nPermutations,
			__m128i& randInts)
		{
			const size_t nElements = std::get<1>(data1);
			const __m128d h1 = _mm_entropy_epu8<N_BITS>(data1);
			const __m128d h2 = _mm_entropy_epu8<N_BITS>(data2);
			const __m128d h1Plus2 = _mm_add_pd(h1, h2);

			auto data3 = deepCopy(data2);
			auto swap = _mm_malloc_m128i(2 * nElements);
			auto results = _mm_malloc_m128d(8 * nPermutations);

			double * const results_double = reinterpret_cast<double * const>(std::get<0>(results));
			for (size_t permutation = 0; permutation < nPermutations; ++permutation) {
				_mm_permute_epu8_array(data3, swap, randInts);
				const __m128d h1And2 = _mm_entropy_epu8<N_BITS>(data1, data2);
				const __m128d mi = _mm_sub_pd(h1Plush2, h1Andh2);
				std::cout << "INFO: _mm_mi_epu8::_mm_mi_perm_epu8_ref: mi=" << mi.m128d_f64[0] << std::endl;
				results_double[permutation] = mi.m128d_f64[0];
			}

			_mm_free2(data3);
			_mm_free2(swap);

			return results;
		}
	}

	template <int N_BITS>
	inline __m128d _mm_mi_epu8(
		std::tuple<const __m128i * const, const size_t> data1,
		std::tuple<const __m128i * const, const size_t> data2)
	{
		return priv::_mm_mi_epu8_ref<N_BITS>(data1, data2);
	}

	template <int N_BITS>
	inline std::tuple<__m128d * const, const size_t> _mm_mi_perm_epu8(
		std::tuple<const __m128i * const, const size_t> data1,
		std::tuple<const __m128i * const, const size_t> data2,
		const size_t nPermutations,
		__m128i& randInts)
	{
		return priv::_mm_mi_perm_epu8_ref<N_BITS>(data1, data2, nPermutations, randInts);
	}
}