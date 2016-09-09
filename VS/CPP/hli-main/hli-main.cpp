#ifdef _MSC_VER
#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
#if !defined(NOMINMAX)
#define NOMINMAX 1 
#endif
#if !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS 1
#endif
#endif

#include <iostream>		// for cout
#include <stdlib.h>		// for printf
#include <time.h>
#include <algorithm>	// for std::min
#include <limits>		// for numeric_limits

#include <tuple>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm> // for std::min

#include "..\hli-lib\toString.h"
#include "..\hli-lib\timer.h"
#include "..\hli-lib\Equal.h"
#include "..\hli-lib\tools.h"

#include "..\hli-lib\_mm_hadd_epu8.h"
#include "..\hli-lib\_mm256_hadd_epu8.h"
#include "..\hli-lib\_mm_variance_epu8.h"
#include "..\hli-lib\_mm_corr_epu8.h"

#include "..\hli-lib\_mm_rand_si128.h"
#include "..\hli-lib\_mm_rescale_epu16.h"
#include "..\hli-lib\_mm_permute_array.h"


/*
namespace stats {

	// Covariance population reference
	inline double covar_pop_ref(
		const __m128i * const data1,
		const __m128i * const data2,
		const size_t nElements)
	{
		double average_d1 = sum_8bit_ref(data1, nElements) / nElements;
		double average_d2 = sum_8bit_ref(data2, nElements) / nElements;

		double sum = 0;
		for (size_t i = 0; i < nElements; ++i) {
			sum += ((static_cast<double>(data1[i]) - average_d1) * (static_cast<double>(data2[i]) - average_d2));
		}
		return sum / nElements;
	}

	inline __m128d std_dev_pop(
		const __m128i * const data1,
		const size_t nElements)
	{
		return _mm_sqrt_pd(var_pop(data1, nElements));
	}

	// Standard deviation population reference
	inline __m128d std_dev_pop_ref(
		const __m128i * const data1,
		const size_t nElements)
	{
		return sqrt(var_pop_ref(data1, nElements));
	}

	// Correlation population: return 2 x double correlation
	inline __m128d corr_pop(
		const __m128i * const data1,
		const __m128i * const data2,
		const size_t nElements)
	{
		const __m128d nElementsDouble = _mm_set1_pd(static_cast<double>(nElements));

		const __m128d average_d1 = _mm_div_pd(_mm_castsi128_pd(sum_6bit(data1, nElements)), nElementsDouble);
		const __m128d average_d2 = _mm_div_pd(_mm_castsi128_pd(sum_6bit(data2, nElements)), nElementsDouble);
		//std_dev_d1 = std_dev_pop(d1)
		//std_dev_d2 = std_dev_pop(d2)

		//return covar_pop_2x2bit(d1, d2, nElements) / (std_dev_d1 * std_dev_d2)
		return _mm_setzero_pd();
	}

	// Correlation population reference
	inline __m128d corr_pop_ref(
		const __m128i * const data1,
		const __m128i * const data2,
		const size_t nElements)
	{
		const double std_dev_d1 = std_dev_pop_ref(data1, nElements);
		const double std_dev_d2 = std_dev_pop_ref(data2, nElements);
		return covar_pop_ref(data1, data2, nElements) / (std_dev_d1 * std_dev_d2);
	}
}
*/
namespace hli {

	void test_mm256_hadd_epu8(const size_t nBlocks, const size_t nExperiments, const bool doTests)
	{
		const size_t nBytes = resizeNBytes(16 * nBlocks, 16);
		__m256i * const mem_addr = static_cast<__m256i *>(_mm_malloc(nBytes, 32));
		fillRand_epu8<5>(mem_addr, nBytes);
		{
			double min_ref = std::numeric_limits<double>::max();
			double min = std::numeric_limits<double>::max();

			for (size_t i = 0; i < nExperiments; ++i) {

				timer::reset_and_start_timer();
				const __m256i result_ref = hli::priv::_mm256_hadd_epu8_ref(mem_addr, nBytes);
				min_ref = std::min(min_ref, timer::get_elapsed_kcycles());
				{
					timer::reset_and_start_timer();
					const __m256i result = hli::_mm256_hadd_epu8<8>(mem_addr, nBytes);
					min = std::min(min, timer::get_elapsed_kcycles());

					if (doTests) {
						if (result_ref.m256i_i64[0] != result.m256i_i64[0]) {
							std::cout << "INFO: test _mm256_hadd_epu8: result-ref=" << hli::toString_i64(result_ref) << "; result=" << hli::toString_i64(result) << std::endl;
						}
					}
				}
			}
			printf("[_mm256_hadd_epu8 Ref]    : %2.5f Kcycles\n", min_ref);
			printf("[_mm256_hadd_epu8 Method1]: %2.5f Kcycles; %2.3f times faster than ref\n", min, min_ref / min);
		}

		_mm_free(mem_addr);
	}

	void test_mm_variance_epu8(const size_t nBlocks, const size_t nExperiments, const bool doTests)
	{
		const double delta = 0.0000001;
		const size_t nBytes = resizeNBytes(16 * nBlocks, 16);
		__m128i * const mem_addr = static_cast<__m128i *>(_mm_malloc(nBytes, 16));
		fillRand_epu8<5>(mem_addr, nBytes);

		{
			double min_ref = std::numeric_limits<double>::max();
			double min1 = std::numeric_limits<double>::max();
			double min2 = std::numeric_limits<double>::max();
			double min3 = std::numeric_limits<double>::max();
			double min4 = std::numeric_limits<double>::max();

			for (size_t i = 0; i < nExperiments; ++i) {

				timer::reset_and_start_timer();
				const __m128d result_ref = hli::priv::_mm_variance_epu8_ref(mem_addr, nBytes);
				min_ref = std::min(min_ref, timer::get_elapsed_kcycles());

				{
					timer::reset_and_start_timer();
					const __m128d result = hli::_mm_variance_epu8<8>(mem_addr, nBytes);
					min1 = std::min(min1, timer::get_elapsed_kcycles());

					if (doTests) {
						if (std::abs(result_ref.m128d_f64[0] - result.m128d_f64[0]) > delta) {
							std::cout << "WARNING: test _mm_variance_epu8<8>: result-ref=" << hli::toString_f64(result_ref) << "; result=" << hli::toString_f64(result) << std::endl;
							return;
						}
					}
				}
				{
					timer::reset_and_start_timer();
					const __m128d result = hli::_mm_variance_epu8<7>(mem_addr, nBytes);
					min2 = std::min(min2, timer::get_elapsed_kcycles());

					if (doTests) {
						if (std::abs(result_ref.m128d_f64[0] - result.m128d_f64[0]) > delta) {
							std::cout << "WARNING: test _mm_variance_epu8<7>: result-ref=" << hli::toString_f64(result_ref) << "; result=" << hli::toString_f64(result) << std::endl;
							return;
						}
					}
				}
				{
					timer::reset_and_start_timer();
					const __m128d result = hli::_mm_variance_epu8<6>(mem_addr, nBytes);
					min3 = std::min(min3, timer::get_elapsed_kcycles());

					if (doTests) {
						if (std::abs(result_ref.m128d_f64[0] - result.m128d_f64[0]) > delta) {
							std::cout << "WARNING: test _mm_variance_epu8<6>: result-ref=" << hli::toString_f64(result_ref) << "; result=" << hli::toString_f64(result) << std::endl;
							return;
						}
					}
				}
				{
					timer::reset_and_start_timer();
					const __m128d result = hli::_mm_variance_epu8<5>(mem_addr, nBytes);
					min4 = std::min(min4, timer::get_elapsed_kcycles());

					if (doTests) {
						if (std::abs(result_ref.m128d_f64[0] - result.m128d_f64[0]) > delta) {
							std::cout << "WARNING: test _mm_variance_epu8<5>: result-ref=" << hli::toString_f64(result_ref) << "; result=" << hli::toString_f64(result) << std::endl;
							return;
						}
					}
				}
			}
			printf("[_mm_variance_epu8 Ref]    : %2.5f Kcycles\n", min_ref);
			printf("[_mm_variance_epu8<8>]     : %2.5f Kcycles; %2.3f times faster than ref\n", min1, min_ref / min1);
			printf("[_mm_variance_epu8<7>]     : %2.5f Kcycles; %2.3f times faster than ref\n", min2, min_ref / min2);
			printf("[_mm_variance_epu8<6>]     : %2.5f Kcycles; %2.3f times faster than ref\n", min3, min_ref / min3);
			printf("[_mm_variance_epu8<5>]     : %2.5f Kcycles; %2.3f times faster than ref\n", min4, min_ref / min4);
		}

		_mm_free(mem_addr);
	}

	void test_mm_rand_si128(const size_t nBlocks, const size_t nExperiments, const bool doTests)
	{
		const size_t nBytes = resizeNBytes(16 * nBlocks, 16);
		__m128i * const mem_addr1 = static_cast<__m128i *>(_mm_malloc(nBytes, 16));
		__m128i * const mem_addr2 = static_cast<__m128i *>(_mm_malloc(nBytes, 16));
		const __m128i randSeeds = _mm_set_epi32(rand(), rand(), rand(), rand());

		__m128i randInts1 = randSeeds;
		__m128i randInts2 = randSeeds;

		double min_ref = std::numeric_limits<double>::max();
		double min1 = std::numeric_limits<double>::max();

		for (size_t i = 0; i < nExperiments; ++i) {

			timer::reset_and_start_timer();
			hli::priv::_mm_rand_si128_ref(mem_addr1, nBytes, randInts1);
			min_ref = std::min(min_ref, timer::get_elapsed_kcycles());

			{
				timer::reset_and_start_timer();
				hli::_mm_lfsr32_epu32(mem_addr2, nBytes, randInts2);
				min1 = std::min(min1, timer::get_elapsed_kcycles());

				if (doTests) {
					for (size_t j = 0; j < 4; ++j) {
						if (randInts1.m128i_u32[j] != randInts2.m128i_u32[j]) {
							std::cout << "WARNING: test _mm_rand_si128: randInts1=" << hli::toString_u32(randInts1) << "; randInts2=" << hli::toString_u32(randInts2) << std::endl;
							return;
						}
					}
					for (size_t block = 0; block < nBlocks; ++block) {
						for (size_t j = 0; j < 4; ++j) {
							if (std::abs(mem_addr1[block].m128i_u32[j] != mem_addr2[block].m128i_u32[j])) {
								std::cout << "WARNING: test _mm_rand_si128: result-ref=" << hli::toString_u32(mem_addr1[block]) << "; result=" << hli::toString_u32(mem_addr2[block]) << std::endl;
								return;
							}
						}
					}
				}
			}
		}
		printf("[_mm_rand_si128 Ref] : %2.5f Kcycles\n", min_ref);
		printf("[_mm_rand_si128]     : %2.5f Kcycles; %2.3f times faster than ref\n", min1, min_ref / min1);

		_mm_free(mem_addr1);
		_mm_free(mem_addr2);
	}

	void test_mm_rescale_epu16(const size_t nBlocks, const size_t nExperiments, const bool doTests)
	{
		if ((nBlocks * 8) > 0xFFFF) {
			std::cout << "WARNING: test mm_rescale_epu16: too many blocks=" << nBlocks << std::endl;
			return;
		}

		const size_t nBytes = resizeNBytes(16 * nBlocks, 16);
		__m128i * const mem_addr = static_cast<__m128i *>(_mm_malloc(nBytes, 16));
		__m128i * const mem_addr1 = static_cast<__m128i *>(_mm_malloc(nBytes, 16));
		__m128i * const mem_addr2 = static_cast<__m128i *>(_mm_malloc(nBytes, 16));
		const __m128i seed = _mm_set_epi32(rand(), rand(), rand(), rand());
		__m128i randInt = seed;
		hli::_mm_lfsr32_epu32(mem_addr, nBytes, randInt);

		double min_ref = std::numeric_limits<double>::max();
		double min1 = std::numeric_limits<double>::max();

		for (size_t i = 0; i < nExperiments; ++i) {

			memcpy(mem_addr1, mem_addr, nBytes);
			timer::reset_and_start_timer();
			hli::priv::_mm_rescale_epu16_ref(mem_addr1, nBytes);
			min_ref = std::min(min_ref, timer::get_elapsed_kcycles());

			{
				memcpy(mem_addr2, mem_addr, nBytes);
				timer::reset_and_start_timer();
				hli::_mm_rescale_epu16(mem_addr2, nBytes);
				min1 = std::min(min1, timer::get_elapsed_kcycles());

				if (doTests) {
					for (size_t block = 0; block < nBlocks; ++block) {
						for (size_t j = 0; j < 8; ++j) {
							if (std::abs(mem_addr1[block].m128i_u16[j] != mem_addr2[block].m128i_u16[j])) {
								std::cout << "WARNING: test mm_rescale_epu16: result-ref=" << hli::toString_u16(mem_addr1[block]) << "; result=" << hli::toString_u16(mem_addr2[block]) << std::endl;
								return;
							}
						}
					}
				}
			}
		}
		if (doTests) {
			__int16 k = 0;
			for (size_t block = 0; block < nBlocks; ++block) {
				for (size_t j = 0; j < 8; ++j) {
					if (mem_addr1[block].m128i_u16[j] > k) {
						std::cout << "WARNING: test mm_rescale_epu16: position " << k << " has value " << mem_addr1[block].m128i_u16[j] << " which is too large" << std::endl;
						return;
					}
					k++;
				}
				//std::cout << "INFO: test mm_rescale_epu16: block=" << block << "; result=" << hli::toString_u16(mem_addr1[block]) << std::endl;
			}
		}

		printf("[_mm_rescale_epu16 Ref] : %2.5f Kcycles\n", min_ref);
		printf("[_mm_rescale_epu16]     : %2.5f Kcycles; %2.3f times faster than ref\n", min1, min_ref / min1);

		_mm_free(mem_addr);
		_mm_free(mem_addr1);
		_mm_free(mem_addr2);
	}

}

int main()
{
#	ifdef _MSC_VER
#		if _DEBUG
	_CrtSetDbgFlag(_CrtSetDbgFlag(_CRTDBG_REPORT_FLAG) | _CRTDBG_LEAK_CHECK_DF);
#		endif
#	endif
	{
		const auto start = std::chrono::system_clock::now();

		const size_t nExperiments = 10000;
		//hli::test_endianess();

		//hli::test::test_mm_hadd_epu8(10010, nExperiments, true);
		//hli::test_mm256_hadd_epu8(10010, nExperiments, true);
		//hli::test_mm_variance_epu8(10010, nExperiments, true);
		hli::test::test_mm_corr_epu8(1010, nExperiments, true);

		//hli::test_mm_rand_si128(1010, nExperiments, true);
		//hli::test_mm_rescale_epu16(2010, nExperiments, true);
		//hli::test::test_mm_permute_epu8_array(3102, nExperiments, true);
		//hli::test::test_mm_permute_dp_array(3102, nExperiments, true);

		//hli::test::test_mm_corr_perm_epu8(100, 1000, nExperiments, true);


		const auto diff = std::chrono::system_clock::now() - start;
		std::cout << std::endl
			<< "DONE: passed time: "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() << " ms = "
			<< std::chrono::duration_cast<std::chrono::seconds>(diff).count() << " sec = "
			<< std::chrono::duration_cast<std::chrono::minutes>(diff).count() << " min = "
			<< std::chrono::duration_cast<std::chrono::hours>(diff).count() << " hours" << std::endl;

		printf("\n-------------------\n");
		printf("\nPress RETURN to finish:");
	}

#	ifdef _MSC_VER
#		if _DEBUG
	_CrtDumpMemoryLeaks();
#		endif
#	endif

	getchar();
	return 0;
}

