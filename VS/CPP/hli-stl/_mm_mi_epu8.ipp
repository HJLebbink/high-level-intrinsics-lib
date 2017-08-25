#pragma once

#include <algorithm>	// std::min
#include <limits>		// std::numeric_limits
#include <iostream>		// std::cout
#include <tuple>

//#include "mmintrin.h"  // mmx
#include "emmintrin.h"  // sse
#include "pmmintrin.h"  // sse3
#include "tmmintrin.h"  // ssse3
#include "smmintrin.h"  // sse4.1
#include "nmmintrin.h"  // sse4.2
//#include "immintrin.h"  // avx, avx2, avx512, FP16C, KNCNI, FMA
//#include "ammintrin.h"  // AMD-specific intrinsics

#include "tools.ipp"
#include "timer.ipp"
#include "_mm_rand_si128.ipp"
#include "_mm_entropy_epu8.ipp"


namespace hli {

	namespace priv {

		template <int N_BITS1, int N_BITS2, bool HAS_MV, U8 MV>
		inline __m128d _mm_mi_epu8_method0(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const std::tuple<const __m128i * const, const size_t>& data2,
			const size_t nElements)
		{
			const __m128d h1 = priv::_mm_entropy_epu8_method0<N_BITS1, HAS_MV, MV>(data1, nElements);
			const __m128d h2 = priv::_mm_entropy_epu8_method0<N_BITS2, HAS_MV, MV>(data2, nElements);
			const __m128d h1Plush2 = _mm_add_pd(h1, h2);
			const __m128d h1Andh2 = priv::_mm_entropy_epu8_method0<N_BITS1, N_BITS2, HAS_MV, MV>(data1, data2, nElements);
			const __m128d mi = _mm_sub_pd(h1Plush2, h1Andh2);
			return mi;
		}

		template <int N_BITS1, int N_BITS2, bool HAS_MV, U8 MV>
		inline __m128d _mm_mi_epu8_method1(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const std::tuple<const __m128i * const, const size_t>& data2,
			const size_t nElements)
		{
			//TODO
			return _mm_mi_epu8_method0<N_BITS1, N_BITS2, HAS_MV, MV>(data1, data2, nElements);
		}

	}

	namespace test {

		void test_mm_mi_epu8(
			const size_t nBlocks, 
			const size_t nExperiments, 
			const bool doTests)
		{
			const double delta = 0.0000001;
			const bool HAS_MV = false;
			const U8 MV = 0xFF;

			const size_t nElements = 16 * nBlocks;
			const int N_BITS1 = 2;
			const int N_BITS2 = 2;


			auto data1_r = _mm_malloc_m128i(nElements);
			auto data2_r = _mm_malloc_m128i(nElements);

			fillRand_epu8<N_BITS1>(data1_r);
			fillRand_epu8<N_BITS2>(data2_r);

			const std::tuple<const __m128i * const, const size_t> data1 = data1_r;
			const std::tuple<const __m128i * const, const size_t> data2 = data2_r;

			double min0 = std::numeric_limits<double>::max();
			double min1 = std::numeric_limits<double>::max();

			__m128d result0, result1;

			for (size_t i = 0; i < nExperiments; ++i) 
			{
				timer::reset_and_start_timer();
				result0 = hli::priv::_mm_mi_epu8_method0<N_BITS1, N_BITS2, HAS_MV, MV>(data1, data2, nElements);
				min0 = std::min(min0, timer::get_elapsed_kcycles());

				{
					timer::reset_and_start_timer();
					result1 = hli::priv::_mm_mi_epu8_method1<N_BITS1, N_BITS2, HAS_MV, MV>(data1, data2, nElements);
					min1 = std::min(min1, timer::get_elapsed_kcycles());

					if (doTests) {
						if (std::abs(result0.m128d_f64[0] - result1.m128d_f64[0]) > delta) {
							std::cout << "WARNING: test _mm_mi_epu8_method0: result0=" << hli::toString_f64(result0) << "; result1=" << hli::toString_f64(result1) << std::endl;
							return;
						}
					}
				}
			}
			printf("[_mm_mi_epu8_method0<%i,%i>]: %2.5f Kcycles; %0.14f\n", N_BITS1, N_BITS2, min0, result0.m128d_f64[0]);
			printf("[_mm_mi_epu8_method1<%i,%i>]: %2.5f Kcycles; %0.14f; %2.3f times faster than ref\n", N_BITS1, N_BITS2, min1, result1.m128d_f64[0], min0 / min1);

			_mm_free2(data1);
			_mm_free2(data2);
		}

	}

	template <int N_BITS1, int N_BITS2, bool HAS_MV>
	inline __m128d _mm_mi_epu8(
		const std::tuple<const __m128i * const, const size_t>& data1,
		const std::tuple<const __m128i * const, const size_t>& data2,
		const size_t nElements)
	{
		return priv::_mm_mi_epu8_method0<N_BITS1, N_BITS2, HAS_MV>(data1, data2, nElements);
		//return priv::_mm_mi_epu8_method1<N_BITS1, N_BITS2, HAS_MV>(data1, data2, nElements);
	}

	template <bool HAS_MV>
	inline __m128d _mm_mi_epu8(
		const std::tuple<const __m128i * const, const size_t>& data1,
		const int nBits1,
		const std::tuple<const __m128i * const, const size_t>& data2,
		const int nBits2,
		const size_t nElements)
	{
		switch (nBits1) {
		case 1:
			switch (nBits2) 
			{
			case 1: return _mm_mi_epu8<1, 1, HAS_MV>(data1, data2, nElements);
			case 2: return _mm_mi_epu8<1, 2, HAS_MV>(data1, data2, nElements);
			case 3: return _mm_mi_epu8<1, 3, HAS_MV>(data1, data2, nElements);
			case 4: return _mm_mi_epu8<1, 4, HAS_MV>(data1, data2, nElements);
			case 5: return _mm_mi_epu8<1, 5, HAS_MV>(data1, data2, nElements);
			case 6: return _mm_mi_epu8<1, 6, HAS_MV>(data1, data2, nElements);
			case 7: return _mm_mi_epu8<1, 7, HAS_MV>(data1, data2, nElements);
			default: return _mm_setzero_pd();
			}
		case 2:
			switch (nBits2)
			{
			case 1: return _mm_mi_epu8<2, 1, HAS_MV>(data1, data2, nElements);
			case 2: return _mm_mi_epu8<2, 2, HAS_MV>(data1, data2, nElements);
			case 3: return _mm_mi_epu8<2, 3, HAS_MV>(data1, data2, nElements);
			case 4: return _mm_mi_epu8<2, 4, HAS_MV>(data1, data2, nElements);
			case 5: return _mm_mi_epu8<2, 5, HAS_MV>(data1, data2, nElements);
			case 6: return _mm_mi_epu8<2, 6, HAS_MV>(data1, data2, nElements);
			default: return _mm_setzero_pd();
			}
		case 3:
			switch (nBits2)
			{
			case 1: return _mm_mi_epu8<3, 1, HAS_MV>(data1, data2, nElements);
			case 2: return _mm_mi_epu8<3, 2, HAS_MV>(data1, data2, nElements);
			case 3: return _mm_mi_epu8<3, 3, HAS_MV>(data1, data2, nElements);
			case 4: return _mm_mi_epu8<3, 4, HAS_MV>(data1, data2, nElements);
			case 5: return _mm_mi_epu8<3, 5, HAS_MV>(data1, data2, nElements);
			default: return _mm_setzero_pd();
			}
		case 4:
			switch (nBits2)
			{
			case 1: return _mm_mi_epu8<4, 1, HAS_MV>(data1, data2, nElements);
			case 2: return _mm_mi_epu8<4, 2, HAS_MV>(data1, data2, nElements);
			case 3: return _mm_mi_epu8<4, 3, HAS_MV>(data1, data2, nElements);
			case 4: return _mm_mi_epu8<4, 4, HAS_MV>(data1, data2, nElements);
			default: return _mm_setzero_pd();
			}
		case 5:
			switch (nBits2)
			{
			case 1: return _mm_mi_epu8<5, 1, HAS_MV>(data1, data2, nElements);
			case 2: return _mm_mi_epu8<5, 2, HAS_MV>(data1, data2, nElements);
			case 3: return _mm_mi_epu8<5, 3, HAS_MV>(data1, data2, nElements);
			default: return _mm_setzero_pd();
			}
		case 6:
			switch (nBits2)
			{
			case 1: return _mm_mi_epu8<6, 1, HAS_MV>(data1, data2, nElements);
			case 2: return _mm_mi_epu8<6, 2, HAS_MV>(data1, data2, nElements);
			default: return _mm_setzero_pd();
			}
		case 7:
			switch (nBits2)
			{
			case 1: return _mm_mi_epu8<7, 1, HAS_MV>(data1, data2, nElements);
			default: return _mm_setzero_pd();
			}
		default: return _mm_setzero_pd();
		}
	}
}