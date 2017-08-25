#pragma once

namespace hli {

	namespace priv {

		template <int N_BITS1, int N_BITS2, bool HAS_MV>
		inline void _mm_mi_corr_epu8_perm_method0(
			const std::tuple<const __m128i * const, const size_t>& data1,
			const std::tuple<const __m128i * const, const size_t>& data2,
			const size_t nElements,
			const std::tuple<__m128d * const, const size_t>& results_mi,
			const std::tuple<__m128d * const, const size_t>& results_corr,
			const size_t nPermutations,
			__m128i& randInts)
		{
			//std::cout << "INFO: _mm_mi_corr_perm_epu8_method0: N_BITS1=" << N_BITS1 << "; N_BITS2=" << N_BITS2 << "; nElements="<< nElements << std::endl;
			//MI stuff
			const __m128d h1 = _mm_entropy_epu8<N_BITS1, HAS_MV>(data1, nElements);
			const __m128d h2 = _mm_entropy_epu8<N_BITS2, HAS_MV>(data2, nElements);
			const __m128d h1Plush2 = _mm_add_pd(h1, h2);

			//Corr stuff
			const __int8 * const ptr1 = reinterpret_cast<const __int8 * const>(std::get<0>(data1));
			const __int8 * const ptr2 = reinterpret_cast<const __int8 * const>(std::get<0>(data2));

			__int32 s11 = 0;
			__int32 s22 = 0;
			__int32 s1 = 0;
			__int32 s2 = 0;

			for (size_t i = 0; i < nElements; ++i)
			{
				const unsigned __int8 d1 = ptr1[i];
				const unsigned __int8 d2 = ptr2[i];
				s11 += d1 * d1;
				s22 += d2 * d2;
				s1 += d1;
				s2 += d2;
			}

			const double s11d = static_cast<double>(s11);
			const double s22d = static_cast<double>(s22);
			const double s1d = static_cast<double>(s1);
			const double s2d = static_cast<double>(s2);

			//Permutation stuff
			auto data3 = deepCopy(data2);
			auto swap = _mm_malloc_m128i(nElements << 1);

			double * const results_mi_double = reinterpret_cast<double * const>(std::get<0>(results_mi));
			double * const results_corr_Double = reinterpret_cast<double * const>(std::get<0>(results_corr));
			const __int8 * const ptr3 = reinterpret_cast<const __int8 * const>(std::get<0>(data3));

			for (size_t permutation = 0; permutation < nPermutations; ++permutation)
			{
				//1] Permutate
				_mm_permute_epu8_array(data3, nElements, swap, randInts);

				// calc MI
				const __m128d h1Andh2 = _mm_entropy_epu8<N_BITS1, N_BITS2, HAS_MV>(data1, data3, nElements);
				const __m128d mi = _mm_sub_pd(h1Plush2, h1Andh2);

#					if	_DEBUG
				if (isnan(mi.m128d_f64[0])) std::cout << "WARNING: _mm_mi_corr_perm_epu8_method0<" << N_BITS1 << "," << N_BITS2 << ">: mi is NAN" << std::endl;
				if (mi.m128d_f64[0] <= 0)   std::cout << "WARNING: _mm_mi_corr_perm_epu8_method0<" << N_BITS1 << "," << N_BITS2 << ">: permutation=" << permutation << ": mi=" << mi.m128d_f64[0] << " is smaller than 0. h1=" << h1.m128d_f64[0] << "; h2 = " << h2.m128d_f64[0] << "; h1Plush2std = " << h1Plush2.m128d_f64[0] << "; h1Andh2 = " << h1Andh2.m128d_f64[0] << std::endl;
#					endif
				results_mi_double[permutation] = mi.m128d_f64[0];

				// calc Corr
				__int32 s12 = 0;
				for (size_t i = 0; i < nElements; ++i)
				{
					const unsigned __int8 d1 = ptr1[i];
					const unsigned __int8 d2 = ptr3[i];
					s12 += d1 * d2;
				}
				const double s12d = static_cast<double>(s12);
				double corr = ((nElements * s12d) - (s1d*s2d)) / (sqrt((nElements*s11d) - (s1d*s1d)) * sqrt((nElements * s22d) - (s2d*s2d)));
				results_corr_Double[permutation] = corr;
			}
			_mm_free2(data3);
			_mm_free2(swap);
		}
	}

	namespace test {

	}

	template <int N_BITS1, int N_BITS2, bool HAS_MV>
	inline void _mm_mi_corr_epu8_perm(
		const std::tuple<const __m128i * const, const size_t>& data1,
		const std::tuple<const __m128i * const, const size_t>& data2,
		const size_t nElements,
		const std::tuple<__m128d * const, const size_t>& results_mi,
		const std::tuple<__m128d * const, const size_t>& results_corr,
		const size_t nPermutations,
		__m128i& randInts)
	{
		priv::_mm_mi_corr_epu8_perm_method0<N_BITS1, N_BITS2, HAS_MV>(data1, data2, nElements, results_mi, results_corr, nPermutations, randInts);
	}

	template <bool HAS_MV>
	inline void _mm_mi_corr_epu8_perm(
		const std::tuple<const __m128i * const, const size_t>& data1,
		const int nBits1,
		const std::tuple<const __m128i * const, const size_t>& data2,
		const int nBits2,
		const size_t nElements,
		const std::tuple<__m128d * const, const size_t>& results_mi,
		const std::tuple<__m128d * const, const size_t>& results_corr,
		const size_t nPermutations,
		__m128i& randInts)
	{
		switch (nBits1) {
		case 1:
			switch (nBits2)
			{
			case 1: _mm_mi_corr_epu8_perm<1, 1, HAS_MV>(data1, data2, nElements, results_mi, results_corr, nPermutations, randInts); return;
			case 2: _mm_mi_corr_epu8_perm<1, 2, HAS_MV>(data1, data2, nElements, results_mi, results_corr, nPermutations, randInts); return;
			case 3: _mm_mi_corr_epu8_perm<1, 3, HAS_MV>(data1, data2, nElements, results_mi, results_corr, nPermutations, randInts); return;
			case 4: _mm_mi_corr_epu8_perm<1, 4, HAS_MV>(data1, data2, nElements, results_mi, results_corr, nPermutations, randInts); return;
			case 5: _mm_mi_corr_epu8_perm<1, 5, HAS_MV>(data1, data2, nElements, results_mi, results_corr, nPermutations, randInts); return;
			case 6: _mm_mi_corr_epu8_perm<1, 6, HAS_MV>(data1, data2, nElements, results_mi, results_corr, nPermutations, randInts); return;
			case 7: _mm_mi_corr_epu8_perm<1, 7, HAS_MV>(data1, data2, nElements, results_mi, results_corr, nPermutations, randInts); return;
			default: return;
			}
		case 2:
			switch (nBits2)
			{
			case 1: _mm_mi_corr_epu8_perm<2, 1, HAS_MV>(data1, data2, nElements, results_mi, results_corr, nPermutations, randInts); return;
			case 2: _mm_mi_corr_epu8_perm<2, 2, HAS_MV>(data1, data2, nElements, results_mi, results_corr, nPermutations, randInts); return;
			case 3: _mm_mi_corr_epu8_perm<2, 3, HAS_MV>(data1, data2, nElements, results_mi, results_corr, nPermutations, randInts); return;
			case 4: _mm_mi_corr_epu8_perm<2, 4, HAS_MV>(data1, data2, nElements, results_mi, results_corr, nPermutations, randInts); return;
			case 5: _mm_mi_corr_epu8_perm<2, 5, HAS_MV>(data1, data2, nElements, results_mi, results_corr, nPermutations, randInts); return;
			case 6: _mm_mi_corr_epu8_perm<2, 6, HAS_MV>(data1, data2, nElements, results_mi, results_corr, nPermutations, randInts); return;
			default: return;
			}
		case 3:
			switch (nBits2)
			{
			case 1: _mm_mi_corr_epu8_perm<3, 1, HAS_MV>(data1, data2, nElements, results_mi, results_corr, nPermutations, randInts); return;
			case 2: _mm_mi_corr_epu8_perm<3, 2, HAS_MV>(data1, data2, nElements, results_mi, results_corr, nPermutations, randInts); return;
			case 3: _mm_mi_corr_epu8_perm<3, 3, HAS_MV>(data1, data2, nElements, results_mi, results_corr, nPermutations, randInts); return;
			case 4: _mm_mi_corr_epu8_perm<3, 4, HAS_MV>(data1, data2, nElements, results_mi, results_corr, nPermutations, randInts); return;
			case 5: _mm_mi_corr_epu8_perm<3, 5, HAS_MV>(data1, data2, nElements, results_mi, results_corr, nPermutations, randInts); return;
			default: return;
			}
		case 4:
			switch (nBits2)
			{
			case 1: _mm_mi_corr_epu8_perm<4, 1, HAS_MV>(data1, data2, nElements, results_mi, results_corr, nPermutations, randInts); return;
			case 2: _mm_mi_corr_epu8_perm<4, 2, HAS_MV>(data1, data2, nElements, results_mi, results_corr, nPermutations, randInts); return;
			case 3: _mm_mi_corr_epu8_perm<4, 3, HAS_MV>(data1, data2, nElements, results_mi, results_corr, nPermutations, randInts); return;
			case 4: _mm_mi_corr_epu8_perm<4, 4, HAS_MV>(data1, data2, nElements, results_mi, results_corr, nPermutations, randInts); return;
			default: return;
			}
		case 5:
			switch (nBits2)
			{
			case 1: _mm_mi_corr_epu8_perm<5, 1, HAS_MV>(data1, data2, nElements, results_mi, results_corr, nPermutations, randInts); return;
			case 2: _mm_mi_corr_epu8_perm<5, 2, HAS_MV>(data1, data2, nElements, results_mi, results_corr, nPermutations, randInts); return;
			case 3: _mm_mi_corr_epu8_perm<5, 3, HAS_MV>(data1, data2, nElements, results_mi, results_corr, nPermutations, randInts); return;
			default: return;
			}
		case 6:
			switch (nBits2)
			{
			case 1: _mm_mi_corr_epu8_perm<6, 1, HAS_MV>(data1, data2, nElements, results_mi, results_corr, nPermutations, randInts); return;
			case 2: _mm_mi_corr_epu8_perm<6, 2, HAS_MV>(data1, data2, nElements, results_mi, results_corr, nPermutations, randInts); return;
			default: return;
			}
		case 7:
			switch (nBits2)
			{
			case 1: _mm_mi_corr_epu8_perm<7, 1, HAS_MV>(data1, data2, nElements, results_mi, results_corr, nPermutations, randInts); return;
			default: return;
			}
		default: return;
		}
	}
}