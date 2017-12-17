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

#include "..\hli-stl\toString.ipp"
#include "..\hli-stl\timing.ipp"
#include "..\hli-stl\equal.ipp"
#include "..\hli-stl\tools.ipp"

#include "..\hli-stl\_mm_hadd_epu8.ipp"
#include "..\hli-stl\_mm_variance_epu8.ipp"
#include "..\hli-stl\_mm_corr_epu8.ipp"
#include "..\hli-stl\_mm_corr_epu8_perm.ipp"

#include "..\hli-stl\_mm_corr_pd.ipp"
#include "..\hli-stl\_mm512_corr_pd.ipp"



#include "..\hli-stl\_mm_rand_si128.ipp"
#include "..\hli-stl\_mm_rescale_epu16.ipp"
#include "..\hli-stl\_mm_permute_epu8_array.ipp"
#include "..\hli-stl\_mm_permute_pd_array.ipp"
#include "..\hli-stl\_mm_entropy_epu8.ipp"

#include "..\hli-stl\_mm_mi_epu8.ipp"
#include "..\hli-stl\_mm_mi_epu8_perm.ipp"


void testAll() {

	const size_t nExperiments = 1000;

	hli::test::_mm_hadd_epu8_speed_test_1(10010, nExperiments, true);
	hli::test::_mm_variance_epu8_speed_test_1(10010, nExperiments, true);
	hli::test::_mm_corr_epu8_speed_test_1(1010, nExperiments, true);
	hli::test::_mm_corr_pd_speed_test_1(1010, nExperiments, true);
	//hli::test::_mm512_corr_pd_speed_test_1(1010, nExperiments, true);

	hli::test::_mm_rand_si128_speed_test_1(1010, nExperiments, true);
	hli::test::_mm_rescale_epu16_speed_test_1(2102, nExperiments, true);
	hli::test::_mm_permute_epu8_array_speed_test_1(2102, nExperiments, true);
	hli::test::_mm_permute_pd_array_speed_test_1(3102, nExperiments, true);

	hli::test::_mm_corr_epu8_perm_speed_test_1(110, 1000, nExperiments, true);

	hli::test::_mm_entropy_epu8_speed_test_1(100, nExperiments, true);
	hli::test::_mm_mi_epu8_speed_test_1(100, nExperiments, true);
	hli::test::_mm_mi_epu8_perm_speed_test_1(100, 1000, nExperiments, true);
}


int main()
{
	{
		const auto start = std::chrono::system_clock::now();

		if (false) {
			testAll();
		} else {
			const int nExperiments = 1000;
			const int nElements = 200* 128 * 8;

			//hli::test::_mm_permute_epu8_array_speed_test_1(139, nExperiments, true);
			hli::test::_mm_corr_pd_speed_test_1(nElements, nExperiments, true);
			//hli::test::_mm512_corr_pd_speed_test_1(nElements, nExperiments, true);
		}

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
	getchar();
	return 0;
}

