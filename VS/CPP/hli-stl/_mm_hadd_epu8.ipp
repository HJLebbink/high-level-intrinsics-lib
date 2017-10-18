#pragma once

#include <algorithm>	// std::min
#include <limits>		// std::numeric_limits
#include <iostream>		// std::cout

//#include "mmintrin.h"  // mmx
#include "emmintrin.h"  // sse
#include "pmmintrin.h"  // sse3
#include "tmmintrin.h"  // ssse3
#include "smmintrin.h"  // sse4.1
#include "nmmintrin.h"  // sse4.2
#include "immintrin.h"  // avx, avx2, avx512, FP16C, KNCNI, FMA
//#include "ammintrin.h"  // AMD-specific intrinsics

#include "tools.ipp"
#include "timing.ipp"
#include "toString.ipp"
#include "_mm_hadd_epi64.ipp"
#include "_mm_rand_si128.ipp"

namespace hli
{

	namespace priv
	{

		template <bool HAS_MV, U8 MV>
		inline std::tuple<__m128i, __m128i> _mm_hadd_epu8_method0(
			const std::tuple<const __m128i * const, const int>& data,
			const int nElements)
		{
			const int nBytes = std::get<1>(data);
			auto ptr = reinterpret_cast<const U8 * const>(std::get<0>(data));
			unsigned __int32 sum = 0;

			if (HAS_MV)
			{
				unsigned __int32 nElements_No_MV = 0;
				for (int i = 0; i < nElements; ++i)
				{
					U8 d = ptr[i];
					if (d != 0xFF)
					{
						nElements_No_MV++;
						sum += d;
					}
				}
				return std::make_tuple(_mm_set1_epi32(sum), _mm_set1_epi32(nElements_No_MV));
			}
			else
			{
				for (int i = 0; i < nElements; ++i)
				{
					sum += ptr[i];
				}
				return std::make_tuple(_mm_set1_epi32(sum), _mm_set1_epi32((int)nElements));
			}
		}

		template <int N_BITS, bool HAS_MV, U8 MV>
		inline std::tuple<__m128i, __m128i> _mm_hadd_epu8_method1(
			const std::tuple<const __m128i * const, const int>& data,
			const int nElements)
		{
			static_assert((N_BITS > 0) && (N_BITS <= 8), "Number of bits must be in range 1 to 8.");
			#			pragma warning( disable: 280) 
			switch (N_BITS)
			{
				case 8: return _mm_hadd_epu8_method1_nBits8<HAS_MV, MV>(data, nElements);
				case 7: return _mm_hadd_epu8_method1_nBits7<HAS_MV, MV>(data, nElements);
				case 6: return _mm_hadd_epu8_method1_nBits6<HAS_MV, MV>(data, nElements);
				case 5:
				case 4:
				case 3:
				case 2:
				case 1: return _mm_hadd_epu8_method1_nBits5<HAS_MV, MV>(data, nElements);
				default:
					return std::make_tuple(_mm_setzero_si128(), _mm_setzero_si128());
			}
		}

		template <bool HAS_MV, U8 MV>
		inline std::tuple<__m128i, __m128i> _mm_hadd_epu8_method2(
			const std::tuple<const __m128i * const, const int>& data,
			const int nElements)
		{
			static_assert(HAS_MV == false, "not implemented for missing values");
			//assume (nBytes < 2 ^ (32 - 8))

			const int nBytes = std::get<1>(data);
			const int nBlocks = nBytes >> 4; // divide by 16 to get the number of __m128i regs (blocks)

			if (nBytes != nElements) std::cout << "WARNING: test _mm_hadd_epu8_method2: nElements is not equal to number of bytes" << std::endl;

			const __m128i and_mask = _mm_set1_epi32(0b11111111);
			//const __m128i shuffle_mask_0 = _mm_set_epi8(15, 14, 13, 12,   11, 10, 9, 8,   7, 6, 5, 4,   3, 2, 1, 0);
			const __m128i shuffle_mask_1 = _mm_set_epi8(15, 14, 13, 13, 11, 10, 9, 9, 7, 6, 5, 5, 3, 2, 1, 1);
			const __m128i shuffle_mask_2 = _mm_set_epi8(15, 14, 13, 14, 11, 10, 9, 10, 7, 6, 5, 6, 3, 2, 1, 2);
			const __m128i shuffle_mask_3 = _mm_set_epi8(15, 14, 13, 15, 11, 10, 9, 11, 7, 6, 5, 7, 3, 2, 1, 3);

			__m128i sum = _mm_setzero_si128();

			for (int block = 0; block < nBlocks; ++block)
			{
				const __m128i data_Block = std::get<0>(data)[block];
				sum = _mm_add_epi32(sum, _mm_and_si128(data_Block, and_mask));
				sum = _mm_add_epi32(sum, _mm_and_si128(_mm_shuffle_epi8(data_Block, shuffle_mask_1), and_mask));
				sum = _mm_add_epi32(sum, _mm_and_si128(_mm_shuffle_epi8(data_Block, shuffle_mask_2), and_mask));
				sum = _mm_add_epi32(sum, _mm_and_si128(_mm_shuffle_epi8(data_Block, shuffle_mask_3), and_mask));
			}

			const __m128i sum2 = _mm_cvtepi32_epi64(_mm_hadd_epi32(sum, sum));
			return std::make_tuple(_mm_hadd_epi64(sum2), _mm_set1_epi32((int)nElements));
		}

		template <bool HAS_MV, U8 MV>
		inline std::tuple<__m128i, __m128i> _mm_hadd_epu8_method3(
			const std::tuple<const __m128i * const, const int>& data,
			const int nElements)
		{
			static_assert(HAS_MV == false, "not implemented for missing values");
			const int nBytes = std::get<1>(data);
			if (nBytes != nElements) std::cout << "WARNING: test _mm_hadd_epu8_method3: nElements is not equal to number of bytes" << std::endl;
			const int nBlocks = nBytes >> 4; // divide by 16 to get the number of __m128i regs (blocks)
			__m128i sum = _mm_setzero_si128();
			for (int block = 0; block < nBlocks; ++block)
			{
				sum = _mm_add_epi64(sum, _mm_sad_epu8(std::get<0>(data)[block], _mm_setzero_si128()));
			}
			return std::make_tuple(_mm_hadd_epi64(sum), _mm_set1_epi32((int)nElements));
		}

		template <bool HAS_MV, U8 MV>
		inline std::tuple<__m128i, __m128i> _mm_hadd_epu8_method1_nBits8(
			const std::tuple<const __m128i * const, const int>& data,
			const int nElements)
		{
			//static_assert(HAS_MV == false, "not implemented for missing values"); //TODO
			const int nBytes = std::get<1>(data);
			if (nBytes != nElements) std::cout << "WARNING: test _mm_hadd_epu8_method1: nElements is not equal to number of bytes" << std::endl;
			const int nBlocks = nBytes >> 4; // divide by 16 to get the number of __m128i regs (blocks)
			__m128i sum = _mm_setzero_si128();

			for (int block = 0; block < nBlocks; ++block)
			{
				sum = _mm_add_epi64(sum, _mm_sad_epu8(std::get<0>(data)[block], _mm_setzero_si128()));
			}
			return std::make_tuple(_mm_hadd_epi64(sum), _mm_set1_epi32((int)nElements));
		}

		template <bool HAS_MV, U8 MV>
		inline std::tuple<__m128i, __m128i> _mm_hadd_epu8_method1_nBits7(
			const std::tuple<const __m128i * const, const int>& data,
			const int nElements)
		{
			//static_assert(HAS_MV == false, "not implemented for missing values");
			const int nBytes = std::get<1>(data);
			if (nBytes != nElements) std::cout << "WARNING: test _mm_hadd_epu8_method1_nBits7: nElements is not equal to number of bytes" << std::endl;
			const int nBlocks = nBytes >> 4; // divide by 16 to get the number of __m128i regs (blocks)

			__m128i sum = _mm_setzero_si128();

			for (int block = 0; block < nBlocks - 1; block += 2)
			{
				__m128i sum_p = _mm_add_epi8(std::get<0>(data)[block], std::get<0>(data)[block + 1]);
				sum = _mm_add_epi64(sum, _mm_sad_epu8(sum_p, _mm_setzero_si128()));
			}

			const int tail = nBlocks & 0b1;
			if (tail > 0)
			{
				for (int block = (nBlocks - tail); block < nBlocks; ++block)
				{
					//std::cout << "INFO: hli::_mm_hadd_epu8_class<7>: tail=" << tail << "; block=" << block << std::endl;
					sum = _mm_add_epi64(sum, _mm_sad_epu8(std::get<0>(data)[block], _mm_setzero_si128()));
				}
			}
			return std::make_tuple(_mm_hadd_epi64(sum), _mm_set1_epi32((int)nElements));
		}

		template <bool HAS_MV, U8 MV>
		inline std::tuple<__m128i, __m128i> _mm_hadd_epu8_method1_nBits6(
			const std::tuple<const __m128i * const, const int>& data,
			const int nElements)
		{
			//static_assert(HAS_MV == false, "not implemented for missing values");
			const int nBytes = std::get<1>(data);
			if (nBytes != nElements) std::cout << "WARNING: test _mm_hadd_epu8_method1_nBits6: nElements is not equal to number of bytes" << std::endl;
			const int nBlocks = static_cast<int>(nBytes >> 4); // divide by 16 to get the number of __m128i regs (blocks)

			__m128i sum = _mm_setzero_si128();
			__m128i sum_p;

			for (int block = 0; block < (nBlocks - 3); block += 4)
			{
				sum_p = std::get<0>(data)[block];
				sum_p = _mm_add_epi8(sum_p, std::get<0>(data)[block + 1]);
				sum_p = _mm_add_epi8(sum_p, std::get<0>(data)[block + 2]);
				sum_p = _mm_add_epi8(sum_p, std::get<0>(data)[block + 3]);
				sum = _mm_add_epi64(sum, _mm_sad_epu8(sum_p, _mm_setzero_si128()));
			}
			const int tail = nBlocks & 0b11;
			if (tail > 0)
			{
				for (int block = (nBlocks - tail); block < nBlocks; ++block)
				{
					//std::cout << "INFO: hli:::_mm_hadd_epu8_method3: tail: block=" << block << std::endl;
					sum = _mm_add_epi64(sum, _mm_sad_epu8(std::get<0>(data)[block], _mm_setzero_si128()));
				}
			}
			return std::make_tuple(_mm_hadd_epi64(sum), _mm_set1_epi32((int)nElements));
		}

		template <bool HAS_MV, U8 MV>
		inline std::tuple<__m128i, __m128i> _mm_hadd_epu8_method1_nBits5(
			const std::tuple<const __m128i * const, const int>& data,
			const int nElements)
		{
			//static_assert(HAS_MV == false, "not implemented for missing values");
			const int nBytes = std::get<1>(data);
			if (nBytes != nElements) std::cout << "WARNING: test _mm_hadd_epu8_method1_nBits5: nElements is not equal to number of bytes" << std::endl;
			const int nBlocks = static_cast<int>(nBytes >> 4); // divide by 16 to get the number of __m128i regs (blocks)

			__m128i sum = _mm_setzero_si128();
			__m128i sum_p;

			for (int block = 0; block < nBlocks - 7; block += 8)
			{
				sum_p = std::get<0>(data)[block];
				sum_p = _mm_add_epi8(sum_p, std::get<0>(data)[block + 1]);
				sum_p = _mm_add_epi8(sum_p, std::get<0>(data)[block + 2]);
				sum_p = _mm_add_epi8(sum_p, std::get<0>(data)[block + 3]);
				sum_p = _mm_add_epi8(sum_p, std::get<0>(data)[block + 4]);
				sum_p = _mm_add_epi8(sum_p, std::get<0>(data)[block + 5]);
				sum_p = _mm_add_epi8(sum_p, std::get<0>(data)[block + 6]);
				sum_p = _mm_add_epi8(sum_p, std::get<0>(data)[block + 7]);
				sum = _mm_add_epi64(sum, _mm_sad_epu8(sum_p, _mm_setzero_si128()));
			}

			const int tail = nBlocks & 0b111;
			if (tail > 0)
			{
				int startTail = nBlocks - tail;
				sum_p = std::get<0>(data)[startTail];
				for (int block = startTail + 1; block < nBlocks; ++block)
				{
					sum_p = _mm_add_epi8(sum_p, std::get<0>(data)[block]);
				}
				sum = _mm_add_epi64(sum, _mm_sad_epu8(sum_p, _mm_setzero_si128()));
			}
			return std::make_tuple(_mm_hadd_epi64(sum), _mm_set1_epi32((int)nElements));
		}
	}

	namespace test
	{
		using namespace tools::timing;

		void _mm_hadd_epu8_speed_test_1(const int nBlocks, const int nExperiments, const bool doTests)
		{
			const bool HAS_MV = false;
			const U8 MV = 0xFF;
			const int nElements = nBlocks * 16;
			auto data_r = _mm_malloc_m128i(nElements);
			fillRand_epu8<5>(data_r);
			const std::tuple<const __m128i * const, const int> data = data_r; // make a new variable that is const

			{
				double min0 = std::numeric_limits<double>::max();
				double min1 = std::numeric_limits<double>::max();
				double min2 = std::numeric_limits<double>::max();
				double min3 = std::numeric_limits<double>::max();
				double min4 = std::numeric_limits<double>::max();
				double min5 = std::numeric_limits<double>::max();
				double min6 = std::numeric_limits<double>::max();

				for (int i = 0; i < nExperiments; ++i)
				{
					reset_and_start_timer();
					const std::tuple<__m128i, __m128i> result_ref = hli::priv::_mm_hadd_epu8_method0<HAS_MV, MV>(data, nElements);
					const unsigned int sum0 = std::get<0>(result_ref).m128i_u32[0];
					min0 = std::min(min0, get_elapsed_kcycles());

					{
						reset_and_start_timer();
						const std::tuple<__m128i, __m128i> result = hli::priv::_mm_hadd_epu8_method1<8, HAS_MV, MV>(data, nElements);
						min1 = std::min(min1, get_elapsed_kcycles());

						if (doTests)
						{
							if (sum0 != std::get<0>(result).m128i_u32[0])
							{
								std::cout << "WARNING: test _mm_hadd_epu8_method1<8>: result0=" << sum0 << "; result=" << std::get<0>(result).m128i_u32[0] << std::endl;
								return;
							}
						}
					}
					{
						reset_and_start_timer();
						const std::tuple<__m128i, __m128i> result = hli::priv::_mm_hadd_epu8_method1<7, HAS_MV, MV>(data, nElements);
						min2 = std::min(min2, get_elapsed_kcycles());

						if (doTests)
						{
							if (sum0 != std::get<0>(result).m128i_u32[0])
							{
								std::cout << "WARNING: test _mm_hadd_epu8_method1<7>: result-ref=" << sum0 << "; result=" << std::get<0>(result).m128i_u32[0] << std::endl;
								return;
							}
						}
					}
					{
						reset_and_start_timer();
						const std::tuple<__m128i, __m128i> result = hli::priv::_mm_hadd_epu8_method1<6, HAS_MV, MV>(data, nElements);
						min3 = std::min(min3, get_elapsed_kcycles());

						if (doTests)
						{
							if (sum0 != std::get<0>(result).m128i_u32[0])
							{
								std::cout << "WARNING: test _mm_hadd_epu8_method1<6>: result-ref=" << sum0 << "; result=" << std::get<0>(result).m128i_u32[0] << std::endl;
								return;
							}
						}
					}
					{
						reset_and_start_timer();
						const std::tuple<__m128i, __m128i> result = hli::priv::_mm_hadd_epu8_method1<5, HAS_MV, MV>(data, nElements);
						min4 = std::min(min4, get_elapsed_kcycles());

						if (doTests)
						{
							if (sum0 != std::get<0>(result).m128i_u32[0])
							{
								std::cout << "WARNING: test _mm_hadd_epu8_method1<5>: result-ref=" << sum0 << "; result=" << std::get<0>(result).m128i_u32[0] << std::endl;
								return;
							}
						}
					}
					{
						reset_and_start_timer();
						const std::tuple<__m128i, __m128i> result = hli::priv::_mm_hadd_epu8_method2<HAS_MV, MV>(data, nElements);
						min5 = std::min(min5, get_elapsed_kcycles());

						if (doTests)
						{
							if (sum0 != std::get<0>(result).m128i_u32[0])
							{
								std::cout << "WARNING: test _mm_hadd_epu8_method2: result-ref=" << sum0 << "; result=" << std::get<0>(result).m128i_u32[0] << std::endl;
								return;
							}
						}
					}
					{
						reset_and_start_timer();
						const std::tuple<__m128i, __m128i> result = hli::priv::_mm_hadd_epu8_method3<HAS_MV, MV>(data, nElements);
						min6 = std::min(min6, get_elapsed_kcycles());

						if (doTests)
						{
							if (sum0 != std::get<0>(result).m128i_u32[0])
							{
								std::cout << "WARNING: test _mm_hadd_epu8_method3: result-ref=" << sum0 << "; result=" << std::get<0>(result).m128i_u32[0] << std::endl;
								return;
							}
						}
					}
				}
				printf("[_mm_hadd_epu8_method0]   : %2.5f Kcycles\n", min0);
				printf("[_mm_hadd_epu8_method1<8>]: %2.5f Kcycles; %2.3f times faster than ref\n", min1, min0 / min1);
				printf("[_mm_hadd_epu8_method1<7>]: %2.5f Kcycles; %2.3f times faster than ref\n", min2, min0 / min2);
				printf("[_mm_hadd_epu8_method1<6>]: %2.5f Kcycles; %2.3f times faster than ref\n", min3, min0 / min3);
				printf("[_mm_hadd_epu8_method1<5>]: %2.5f Kcycles; %2.3f times faster than ref\n", min4, min0 / min4);
				printf("[_mm_hadd_epu8_method2]:    %2.5f Kcycles; %2.3f times faster than ref\n", min5, min0 / min5);
				printf("[_mm_hadd_epu8_method3]:    %2.5f Kcycles; %2.3f times faster than ref\n", min6, min0 / min6);
			}

			_mm_free2(data);
		}
	}

	// Horizontally add all 8-bit integers in data (with nBytes). return sum and nTrueElements
	// Operation:
	// tmp := sum(mem_addr)
	// dst[31:0] := tmp
	// dst[63:32] := tmp
	// dst[95:64] := tmp
	// dst[127:96] := tmp
	template <int N_BITS, bool HAS_MV, U8 MV>
	inline std::tuple<__m128i, __m128i> _mm_hadd_epu8(
		const std::tuple<const __m128i * const, const int>& data,
		const int nElements)
	{
		#if _DEBUG
		const int nBytes = std::get<1>(data);
		if (nBytes < nElements) std::cout << "ERROR: _mm_hadd_epu8: nElements (" << nElements << ") is too large for the number of bytes (" << nBytes << ")" << std::endl;
		#endif

		return priv::_mm_hadd_epu8_method0<HAS_MV, MV>(data, nElements);
		//return priv::_mm_hadd_epu8_method1<N_BITS, HAS_MV, MV>(data, nElements);
		//return priv::_mm_hadd_epu8_method2<N_BITS, HAS_MV, MV>(data);
	}
}