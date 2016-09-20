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
#include "timer.h"
#include "_mm_rand_si128.h"
#include "_mm_entropy_epu8.h"


namespace hli {

	namespace priv {

		template <int N_BITS>
		inline __m128d _mm_mi_epu8_ref(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const std::tuple<const __m128i * const, const size_t>& data2,
			const size_t nElements)
		{
			const __m128d h1 = priv::_mm_entropy_epu8_ref<N_BITS>(data1, nElements);
			const __m128d h2 = priv::_mm_entropy_epu8_ref<N_BITS>(data2, nElements);
			const __m128d h1Plush2 = _mm_add_pd(h1, h2);
			const __m128d h1Andh2 = priv::_mm_entropy_epu8_ref(data1, data2, nElements);
			const __m128d mi = _mm_sub_pd(h1Plush2, h1Andh2);
			return mi;
		}

		namespace perm {

			template <int N_BITS>
			inline void _mm_mi_perm_epu8_method0(
				const std::tuple<const __m128i * const, const size_t>& data1,
				const std::tuple<const __m128i * const, const size_t>& data2,
				const size_t nElements,
				const std::tuple<__m128d * const, const size_t>& results,
				const size_t nPermutations,
				__m128i& randInts)
			{
				const __m128d h1 = _mm_entropy_epu8<N_BITS>(data1, nElements);
				const __m128d h2 = _mm_entropy_epu8<N_BITS>(data2, nElements);
				const __m128d h1Plush2 = _mm_add_pd(h1, h2);

				auto data3 = deepCopy(data2);
				auto swap = _mm_malloc_m128i(nElements << 1);

				double * const results_double = reinterpret_cast<double * const>(std::get<0>(results));
				for (size_t permutation = 0; permutation < nPermutations; ++permutation)
				{
					_mm_permute_epu8_array_ref(data3, swap, randInts);
					const __m128d h1Andh2 = _mm_entropy_epu8<N_BITS>(data1, data3, nElements);
					const __m128d mi = _mm_sub_pd(h1Plush2, h1Andh2);
					results_double[permutation] = mi.m128d_f64[0];
				}
				_mm_free2(data3);
				_mm_free2(swap);
			}
		}
	}

	namespace test {

		void test_mm_mi_epu8(const size_t nBlocks, const size_t nExperiments, const bool doTests)
		{
			const double delta = 0.0000001;
			const size_t nElements = 16 * nBlocks;

			auto data1 = _mm_malloc_m128i(nElements);
			auto data2 = _mm_malloc_m128i(nElements);

			const int N_BITS1 = 2;
			const int N_BITS2 = 2;

			fillRand_epu8<N_BITS1>(data1);
			fillRand_epu8<N_BITS2>(data2);

			double min_ref = std::numeric_limits<double>::max();
			double min1 = std::numeric_limits<double>::max();
			double min2 = std::numeric_limits<double>::max();

			__m128d result_ref, result1, result2;

			for (size_t i = 0; i < nExperiments; ++i) {

				timer::reset_and_start_timer();
				result_ref = hli::priv::_mm_entropy_epu8_ref(data1, data2, nElements);
				min_ref = std::min(min_ref, timer::get_elapsed_kcycles());

				{
					timer::reset_and_start_timer();
					result1 = hli::priv::_mm_entropy_epu8_method0<N_BITS1, N_BITS2>(data1, data2, nElements);
					min1 = std::min(min1, timer::get_elapsed_kcycles());

					if (doTests) {
						if (std::abs(result_ref.m128d_f64[0] - result1.m128d_f64[0]) > delta) {
							std::cout << "WARNING: test _mm_entropy_epu8_method0<2>: result-ref=" << hli::toString_f64(result_ref) << "; result=" << hli::toString_f64(result1) << std::endl;
							return;
						}
					}
				}
				{
					timer::reset_and_start_timer();
					result2 = hli::priv::_mm_entropy_epu8_method1<N_BITS1, N_BITS2>(data1, data2, nElements);
					min2 = std::min(min2, timer::get_elapsed_kcycles());

					if (doTests) {
						if (std::abs(result_ref.m128d_f64[0] - result2.m128d_f64[0]) > delta) {
							std::cout << "WARNING: test _mm_entropy_epu8_method1<2>: result-ref=" << hli::toString_f64(result_ref) << "; result=" << hli::toString_f64(result2) << std::endl;
							return;
						}
					}
				}
			}
			printf("[_mm_corr_epu8_ref ]          : %2.5f Kcycles; %0.14f\n", min_ref, result_ref.m128d_f64[0]);
			printf("[_mm_corr_epu8_method0<%i,%i>]: %2.5f Kcycles; %0.14f; %2.3f times faster than ref\n", N_BITS1, N_BITS2, min1, result1.m128d_f64[0], min_ref / min1);
			printf("[_mm_corr_epu8_method1<%i,%i>]: %2.5f Kcycles; %0.14f; %2.3f times faster than ref\n", N_BITS1, N_BITS2, min2, result2.m128d_f64[0], min_ref / min2);

			_mm_free2(data1);
			_mm_free2(data2);
		}
	}

	template <int N_BITS>
	inline __m128d _mm_mi_epu8(
		const std::tuple<const __m128i * const, const size_t>& data1,
		const std::tuple<const __m128i * const, const size_t>& data2,
		const size_t nElements)
	{
		return priv::_mm_mi_epu8_ref<N_BITS>(data1, data2, nElements);
	}

	template <int N_BITS>
	inline void _mm_mi_perm_epu8(
		const std::tuple<const __m128i * const, const size_t>& data1,
		const std::tuple<const __m128i * const, const size_t>& data2,
		const size_t nElements,
		const std::tuple<__m128d * const, const size_t>& results,
		const size_t nPermutations,
		__m128i& randInts)
	{
		priv::perm::_mm_mi_perm_epu8_method0<N_BITS>(data1, data2, nElements, results, nPermutations, randInts);
	}
}