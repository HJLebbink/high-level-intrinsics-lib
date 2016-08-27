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

#include "..\hli-lib\hadd.h"
#include "..\hli-lib\hadd-ref.h"
#include "..\hli-lib\toString.h"
#include "..\hli-lib\timer.h"

namespace hli {

	void fillRand(__m128i * const mem_addr, const size_t nBytes) {
		__int8 * const ptr = reinterpret_cast<__int8 * const>(mem_addr);
		for (size_t i = 0; i < nBytes; ++i) {
			ptr[i] = static_cast<__int8>(rand());
		}
	}

	void test_mm_hadd_epi8(const size_t nBlocks, const size_t nExperiments) 
	{
		const size_t nBytes = 16 * nBlocks;
		__m128i * const mem_addr = static_cast<__m128i *>(_mm_malloc(nBytes, 16));
		fillRand(mem_addr, nBytes);
		{
			double min_ref = std::numeric_limits<double>::max();
			double min = std::numeric_limits<double>::max();

			for (size_t i = 0; i < nExperiments; ++i) {

				timer::reset_and_start_timer();
				const __m128i result_ref = hli::ref::_mm_hadd_epi8(mem_addr, nBytes);
				min_ref = std::min(min_ref, timer::get_elapsed_kcycles());

				timer::reset_and_start_timer();
				const __m128i result = hli::_mm_hadd_epi8(mem_addr, nBytes);
				min = std::min(min, timer::get_elapsed_kcycles());

				if (result_ref.m128i_i64[0] != result.m128i_i64[0]) {
					std::cout << "INFO: test _mm_hadd_epu8: result-ref=" << hli::toString_i64(result_ref) << "; result=" << hli::toString_i64(result) << std::endl;
				}
			}
			printf("[Reference]    : %2.5f Kcycles\n", min_ref);
			printf("[Method1]      : %2.5f Kcycles; %2.3f times faster\n", min, min_ref / min);
		}

		_mm_free(mem_addr);
	}

	void test_mm_hadd_epu8(const size_t nBlocks, const size_t nExperiments)
	{
		const size_t nBytes = 16 * nBlocks;
		__m128i * const mem_addr = static_cast<__m128i *>(_mm_malloc(nBytes, 16));
		fillRand(mem_addr, nBytes);

		{
			double min_ref = std::numeric_limits<double>::max();
			double min = std::numeric_limits<double>::max();

			for (__int64 i = 0; i < nExperiments; ++i) {

				timer::reset_and_start_timer();
				const __m128i result_ref = hli::ref::_mm_hadd_epu8(mem_addr, nBytes);
				min_ref = std::min(min_ref, timer::get_elapsed_kcycles());

				timer::reset_and_start_timer();
				const __m128i result = hli::_mm_hadd_epu8(mem_addr, nBytes);
				min = std::min(min, timer::get_elapsed_kcycles());

				if (result_ref.m128i_u64[0] != result.m128i_u64[0]) {
					std::cout << "INFO: test _mm_hadd_epu8: result-ref=" << hli::toString_u64(result_ref) << "; result=" << hli::toString_u64(result) << std::endl;
				}
			}
			printf("[Reference]    : %2.5f Kcycles\n", min_ref);
			printf("[Method1]      : %2.5f Kcycles; %2.5f\n", min, min_ref/min);
		}

		_mm_free(mem_addr);
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

		hli::test_mm_hadd_epu8(200, 10000);

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

