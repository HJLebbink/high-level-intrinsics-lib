#pragma once

#include <tuple>

#include "tools.ipp"
#include "timer.ipp"

#include "_mm_rand_si128.ipp"
#include "_mm_mi_epu8.ipp"

namespace hli {

	namespace priv {

		template <int N_BITS1, int N_BITS2, bool HAS_MV, U8 MV>
		inline void _mm_mi_epu8_perm_method0(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const std::tuple<const __m128i * const, const size_t>& data2,
			const size_t nElements,
			const std::tuple<__m128d * const, const size_t>& results,
			const size_t nPermutations,
			__m128i& randInts)
		{
			//std::cout << "INFO: _mm_mi_perm_epu8_method0: N_BITS1=" << N_BITS1 << "; N_BITS2=" << N_BITS2 << "; nElements="<< nElements << std::endl;

			const __m128d h1 = _mm_entropy_epu8<N_BITS1, HAS_MV, MV>(data1, nElements);
			const __m128d h2 = _mm_entropy_epu8<N_BITS2, HAS_MV, MV>(data2, nElements);
			const __m128d h1Plush2 = _mm_add_pd(h1, h2);

			auto data3 = deepCopy(data2);
			auto swap = _mm_malloc_m128i(nElements << 1);

			double * const results_double = reinterpret_cast<double * const>(std::get<0>(results));
			for (size_t permutation = 0; permutation < nPermutations; ++permutation)
			{
				_mm_permute_epu8_array(data3, nElements, swap, randInts);
				const __m128d h1Andh2 = _mm_entropy_epu8<N_BITS1, N_BITS2, HAS_MV, MV>(data1, data3, nElements);
				const __m128d mi = _mm_sub_pd(h1Plush2, h1Andh2);

#					if	_DEBUG
				if (isnan(mi.m128d_f64[0])) std::cout << "WARNING: _mm_mi_perm_epu8_method0<" << N_BITS1 << "," << N_BITS2 << ">: mi is NAN" << std::endl;
				if (mi.m128d_f64[0] <= 0)   std::cout << "WARNING: _mm_mi_perm_epu8_method0<" << N_BITS1 << "," << N_BITS2 << ">: permutation=" << permutation << ": mi=" << mi.m128d_f64[0] << " is smaller than 0. h1=" << h1.m128d_f64[0] << "; h2 = " << h2.m128d_f64[0] << "; h1Plush2std = " << h1Plush2.m128d_f64[0] << "; h1Andh2 = " << h1Andh2.m128d_f64[0] << std::endl;
#					endif
				results_double[permutation] = mi.m128d_f64[0];
			}
			_mm_free2(data3);
			_mm_free2(swap);
		}

		template <int N_BITS1, int N_BITS2, bool HAS_MV, U8 MV>
		inline void _mm_mi_epu8_perm_method1(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const std::tuple<const __m128i * const, const size_t>& data2,
			const size_t nElements,
			const std::tuple<__m128d * const, const size_t>& results,
			const size_t nPermutations,
			__m128i& randInts)
		{
			//TODO
			return _mm_mi_epu8_perm_method1<N_BITS1, N_BITS2, HAS_MV, MV>(data1, data2, nElements, results, nPermutations, randInts);
		}

	}

	namespace test {

		void test_mm_mi_epu8_perm(
			const size_t nBlocks,
			const size_t nPermutations,
			const size_t nExperiments,
			const bool doTests)
		{
			const double delta = 0.0000001;
			const bool HAS_MV = false;
			const U8 MV = 0xFF;
			const size_t nElements = 16 * nBlocks;
			const int N_BITS1 = 2;
			const int N_BITS2 = 2;


			auto data1_r = _mm_malloc_m128i(nElements * 1);
			auto data2_r = _mm_malloc_m128i(nElements * 1);

			const size_t nBytesResults = resizeNBytes(8 * nPermutations, 16);
			//std::cout << "INFO: test_mm_corr_perm_epu8: nPermutations=" << nPermutations << "; nBytesResults=" << nBytesResults << std::endl;
			auto results0 = _mm_malloc_m128d(nBytesResults);
			auto results1 = _mm_malloc_m128d(nBytesResults);


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
							std::cout << "WARNING: test _mm_mi_perm_epu8_method0: result0=" << hli::toString_f64(result0) << "; result1=" << hli::toString_f64(result1) << std::endl;
							return;
						}
					}
				}
			}
			printf("[_mm_mi_perm_epu8_method0<%i,%i>]: %2.5f Kcycles; %0.14f\n", N_BITS1, N_BITS2, min0, result0.m128d_f64[0]);
			printf("[_mm_mi_perm_epu8_method1<%i,%i>]: %2.5f Kcycles; %0.14f; %2.3f times faster than ref\n", N_BITS1, N_BITS2, min1, result1.m128d_f64[0], min0 / min1);

			_mm_free2(data1);
			_mm_free2(data2);
		}
	}

	template <int N_BITS1, int N_BITS2, bool HAS_MV, U8 MV>
	inline void _mm_mi_epu8_perm(
		const std::tuple<const __m128i * const, const size_t>& data1,
		const std::tuple<const __m128i * const, const size_t>& data2,
		const size_t nElements,
		const std::tuple<__m128d * const, const size_t>& results,
		const size_t nPermutations,
		__m128i& randInts)
	{
		priv::_mm_mi_epu8_perm_method0<N_BITS1, N_BITS2, HAS_MV, MV>(data1, data2, nElements, results, nPermutations, randInts);
	}

	template <bool HAS_MV, U8 MV>
	inline void _mm_mi_epu8_perm(
		const std::tuple<const __m128i * const, const size_t>& data1,
		const int nBits1,
		const std::tuple<const __m128i * const, const size_t>& data2,
		const int nBits2,
		const size_t nElements,
		const std::tuple<__m128d * const, const size_t>& results,
		const size_t nPermutations,
		__m128i& randInts)
	{
		switch (nBits1) {
		case 1:
			switch (nBits2)
			{
			case 1: _mm_mi_epu8_perm<1, 1, HAS_MV, MV>(data1, data2, nElements, results, nPermutations, randInts); return;
			case 2: _mm_mi_epu8_perm<1, 2, HAS_MV, MV>(data1, data2, nElements, results, nPermutations, randInts); return;
			case 3: _mm_mi_epu8_perm<1, 3, HAS_MV, MV>(data1, data2, nElements, results, nPermutations, randInts); return;
			case 4: _mm_mi_epu8_perm<1, 4, HAS_MV, MV>(data1, data2, nElements, results, nPermutations, randInts); return;
			case 5: _mm_mi_epu8_perm<1, 5, HAS_MV, MV>(data1, data2, nElements, results, nPermutations, randInts); return;
			case 6: _mm_mi_epu8_perm<1, 6, HAS_MV, MV>(data1, data2, nElements, results, nPermutations, randInts); return;
			case 7: _mm_mi_epu8_perm<1, 7, HAS_MV, MV>(data1, data2, nElements, results, nPermutations, randInts); return;
			default: return;
			}
		case 2:
			switch (nBits2)
			{
			case 1: _mm_mi_epu8_perm<2, 1, HAS_MV, MV>(data1, data2, nElements, results, nPermutations, randInts); return;
			case 2: _mm_mi_epu8_perm<2, 2, HAS_MV, MV>(data1, data2, nElements, results, nPermutations, randInts); return;
			case 3: _mm_mi_epu8_perm<2, 3, HAS_MV, MV>(data1, data2, nElements, results, nPermutations, randInts); return;
			case 4: _mm_mi_epu8_perm<2, 4, HAS_MV, MV>(data1, data2, nElements, results, nPermutations, randInts); return;
			case 5: _mm_mi_epu8_perm<2, 5, HAS_MV, MV>(data1, data2, nElements, results, nPermutations, randInts); return;
			case 6: _mm_mi_epu8_perm<2, 6, HAS_MV, MV>(data1, data2, nElements, results, nPermutations, randInts); return;
			default: return;
			}
		case 3:
			switch (nBits2)
			{
			case 1: _mm_mi_epu8_perm<3, 1, HAS_MV, MV>(data1, data2, nElements, results, nPermutations, randInts); return;
			case 2: _mm_mi_epu8_perm<3, 2, HAS_MV, MV>(data1, data2, nElements, results, nPermutations, randInts); return;
			case 3: _mm_mi_epu8_perm<3, 3, HAS_MV, MV>(data1, data2, nElements, results, nPermutations, randInts); return;
			case 4: _mm_mi_epu8_perm<3, 4, HAS_MV, MV>(data1, data2, nElements, results, nPermutations, randInts); return;
			case 5: _mm_mi_epu8_perm<3, 5, HAS_MV, MV>(data1, data2, nElements, results, nPermutations, randInts); return;
			default: return;
			}
		case 4:
			switch (nBits2)
			{
			case 1: _mm_mi_epu8_perm<4, 1, HAS_MV, MV>(data1, data2, nElements, results, nPermutations, randInts); return;
			case 2: _mm_mi_epu8_perm<4, 2, HAS_MV, MV>(data1, data2, nElements, results, nPermutations, randInts); return;
			case 3: _mm_mi_epu8_perm<4, 3, HAS_MV, MV>(data1, data2, nElements, results, nPermutations, randInts); return;
			case 4: _mm_mi_epu8_perm<4, 4, HAS_MV, MV>(data1, data2, nElements, results, nPermutations, randInts); return;
			default: return;
			}
		case 5:
			switch (nBits2)
			{
			case 1: _mm_mi_epu8_perm<5, 1, HAS_MV, MV>(data1, data2, nElements, results, nPermutations, randInts); return;
			case 2: _mm_mi_epu8_perm<5, 2, HAS_MV, MV>(data1, data2, nElements, results, nPermutations, randInts); return;
			case 3: _mm_mi_epu8_perm<5, 3, HAS_MV, MV>(data1, data2, nElements, results, nPermutations, randInts); return;
			default: return;
			}
		case 6:
			switch (nBits2)
			{
			case 1: _mm_mi_epu8_perm<6, 1, HAS_MV, MV>(data1, data2, nElements, results, nPermutations, randInts); return;
			case 2: _mm_mi_epu8_perm<6, 2, HAS_MV, MV>(data1, data2, nElements, results, nPermutations, randInts); return;
			default: return;
			}
		case 7:
			switch (nBits2)
			{
			case 1: _mm_mi_epu8_perm<7, 1, HAS_MV, MV>(data1, data2, nElements, results, nPermutations, randInts); return;
			default: return;
			}
		default: return;
		}
	}
}