#pragma once

#include <iostream>		// for cout
#include "toString.h"

//#include "mmintrin.h"  // mmx
#include "emmintrin.h"  // sse
#include "pmmintrin.h"  // sse3
#include "tmmintrin.h"  // ssse3
#include "smmintrin.h"  // sse4.1
#include "nmmintrin.h"  // sse4.2
#include "immintrin.h"  // avx, avx2, avx512, FP16C, KNCNI, FMA
//#include "ammintrin.h"  // AMD-specific intrinsics

namespace hli {

	namespace priv {

		inline void _mm_rescale_epu16_ref(
			const std::tuple<__m128i * const, const size_t>& data)
		{
			/*
			const size_t length = vector.size();
			const bool showInfo = false;

			for (size_t i = 0; i < length; ++i) {
			//std::cout << "rescaleVector: before: i=" << i << ":" << ptr[i] << std::endl;

			const int original = vector[i];
			const int product = original*static_cast<int>(i + 1);
			const int f3 = product >> 16;
			if (showInfo) std::cout << "rescaleVector_reference: i=" << i << "; original " << original << "; product " << product << "; f3 " << f3 << std::endl;

			vector[i] = static_cast<unsigned short>(f3);
			//std::cout << "rescaleVector: after: i=" << i << ":" << ptr[i] << std::endl;
			}
			*/

			__m128i * const ptr = std::get<0>(data);
			const size_t nBytes = std::get<1>(data);

			const size_t length = nBytes >> 1;
			const bool showInfo = false;
			unsigned __int16 * const ptr2 = reinterpret_cast<unsigned __int16 * const>(ptr);

			for (size_t i = 0; i < length; ++i) {
				if (showInfo) std::cout << "rescaleVector: before: i=" << i << ":" << ptr2[i] << std::endl;

				const unsigned __int16 original = ptr2[i];
				const unsigned int product = original*static_cast<unsigned int>(i + 1);
				const unsigned int f3 = product >> 16;
				if (showInfo) std::cout << "rescaleVector_reference: i=" << i << "; original " << original << "; product " << product << "; f3 " << f3 << std::endl;

				ptr2[i] = static_cast<unsigned __int16>(f3);
				//std::cout << "rescaleVector: after: i=" << i << ":" << ptr[i] << std::endl;
			}
		}

		inline void _mm_rescale_epu16_method1(
			const std::tuple<__m128i * const, const size_t>& data)
		{
			__m128i * const ptr = std::get<0>(data);
			const size_t nBytes = std::get<1>(data);

			const bool showInfo = false;
			const size_t nBlocks = nBytes >> 4;

			__m128i m = _mm_set_epi32(4, 3, 2, 1);
			const __m128i increment = _mm_set1_epi32(4);

			if (showInfo) std::cout << "increment=" << toString_i32(increment) << std::endl;

			for (size_t block = 0; block < nBlocks; ++block) {
				const __m128i data = ptr[block];
				if (showInfo) std::cout << "input data=" << toString_u16(data) << std::endl;

				const __m128i numbers1 = _mm_cvtepu16_epi32(data);
				const __m128i numbers2 = _mm_cvtepu16_epi32(_mm_shuffle_epi32(data, 0b11101110));
				if (showInfo) std::cout << "input numbers1=" << toString_u32(numbers1) << "; input numbers2=" << toString_u32(numbers2) << std::endl;

				if (showInfo) std::cout << "max " << toString_i32(m) << std::endl;

				const __m128i product1 = _mm_mullo_epi32(numbers1, m);
				if (showInfo) std::cout << "product1 " << toString_i32(product1) << std::endl;
				m = _mm_add_epi32(m, increment);

				const __m128i product2 = _mm_mullo_epi32(numbers2, m);
				if (showInfo) std::cout << "product2 " << toString_i32(product2) << std::endl;
				m = _mm_add_epi32(m, increment);


				const __m128i f3_int1 = _mm_srli_epi32(product1, 16); //Shifts the 4 signed or unsigned 32 - bit integers in a right by count bits while shifting in zeros.
				if (showInfo) std::cout << "output int1 " << toString_i32(f3_int1) << std::endl;
				const __m128i f3_int2 = _mm_srli_epi32(product2, 16); //Shifts the 4 signed or unsigned 32 - bit integers in a right by count bits while shifting in zeros.
				if (showInfo) std::cout << "output int2 " << toString_i32(f3_int2) << std::endl;
	
				const __m128i saturated = _mm_packs_epi32(f3_int1, f3_int2);
				if (showInfo) std::cout << "output short " << toString_u16(saturated) << std::endl;

				if (showInfo) std::cout << std::endl;
				ptr[block] = saturated;
			}
		}
	}

	inline void _mm_rescale_epu16(
		const std::tuple<__m128i * const, const size_t>& data)
	{
		//priv::_mm_rescale_epu16_ref(data);
		priv::_mm_rescale_epu16_method1(data);

		//for (size_t block = 0; block < (nBytes >> 4); ++block) {
		//	std::cout << "INFO: _mm_rescale_epu16::_mm_rescale_epu16: block="<<block << ": " << toString_u16(mem_addr[block]) << std::endl;
		//}
	}
}