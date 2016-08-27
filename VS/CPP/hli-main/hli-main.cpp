
#include <iostream>		// for cout
#include <stdlib.h>		// for printf
#include <time.h>
#include <algorithm>	// for std::min
#include <limits>		// for numeric_limits

#include <tuple>
#include <vector>
#include <string>
#include <chrono>

#include "..\hli-lib\hadd.h"
#include "..\hli-lib\toString.h"

using namespace hli;


void testHadd() {

	size_t nBlocks = 10;
	size_t nBytes = 16 * nBlocks;

	__m128i * mem_addr = static_cast<__m128i *>(_mm_malloc(nBytes, 16));

	for (size_t block = 0; block < nBlocks; ++block) {
		mem_addr[block] = _mm_setzero_si128();
	}

	const __m128i result = _mm_hadd_epi8(mem_addr, nBytes);
	std::cout << "INFO: testHadd: result=" << toString_u64(result) << std::endl;


	_mm_free(mem_addr);
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

		testHadd();

		const auto diff = std::chrono::system_clock::now() - start;
		std::cout << "DONE: passed time: "
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

