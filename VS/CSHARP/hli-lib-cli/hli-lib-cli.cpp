// This is the main DLL file.

#include "stdafx.h"
#include "intrin.h"
#include <vector>
#include <tuple>
#include <array>


#include "StatsLibCli.h"

#pragma unmanaged
#include "../../CPP/hli-lib/_mm_hadd_epu8.h"
#include "../../CPP/hli-lib/_mm_corr_pd.h"
#include "../../CPP/hli-lib/_mm_corr_epu8.h"
#include "../../CPP/hli-lib/_mm_mi_epu8.h"


using namespace System;
using namespace System::Collections::Generic;

namespace StatsLibCli {

	namespace priv {

#		pragma managed
		void copy_data(
			List<Byte>^ src, 
			const std::tuple<__int8 * const, const size_t>& dst)
		{
			__int8 * const dst2 = std::get<0>(dst);

			if (true) {
				int nElements = src->Count;
				for (int i = 0; i < nElements; ++i) {
					dst2[i] = src[i];
				}
				size_t nBytes = std::get<1>(dst);
				int nElements2 = static_cast<int>(nBytes);
				int tail = nElements2 - nElements;
				for (int i = nElements2 - tail; i < nElements2; ++i) {
					dst2[i] = 0;
				}
			} else {

			}
		}

#		pragma managed
		void copy_data(
			List<Double>^ src, 
			const std::tuple<__int8 * const, const size_t>& dst)
		{
			double * const dst2 = reinterpret_cast<double * const>(std::get<0>(dst));

			if (true) {
				int nElements = src->Count;

				for (int i = 0; i < nElements; ++i) {
					dst2[i] = src[i];
				}
			} else {

			}
		}

#		pragma managed
		void copy_data(
			std::tuple<__int8 * const, const size_t>& src, 
			List<Double>^ dst)
		{
			double * const src2 = reinterpret_cast<double * const>(std::get<0>(src));

			if (true) {
				int nElements = dst->Count;

				for (int i = 0; i < nElements; ++i) {
					dst[i] = src2[i];
				}
			}
			else {

			}
		}
	}

#	pragma unmanaged
	int _mm_hadd_epu8_unmanaged(
		const std::tuple<const __int8 * const, const size_t>& data,
		const size_t nElements)
	{
		return _mm_cvtsi128_si32(hli::_mm_hadd_epu8<8>(hli::_mm_cast_m128i(data), nElements));
	}

#	pragma managed
	int Class1::_mm_hadd_epu8(
		List<Byte>^ data)
	{
		int nElement = data->Count;
		auto data_x = hli::_mm_malloc_xmm(nElement);
		priv::copy_data(data, data_x);
		int result = _mm_hadd_epu8_unmanaged(data_x, nElement);
		hli::_mm_free2(data_x);
		return result;
	}

#	pragma unmanaged
	double _mm_corr_epu8_unmanaged(
		const std::tuple<const __int8 * const, const size_t>& data1,
		const std::tuple<const __int8 * const, const size_t>& data2,
		const size_t nElements)
	{
		return _mm_cvtsd_f64(hli::_mm_corr_epu8<8>(hli::_mm_cast_m128i(data1), hli::_mm_cast_m128i(data2), nElements));
	}

#	pragma managed
	double Class1::_mm_corr_epu8(
		List<Byte>^ data1, 
		List<Byte>^ data2)
	{
		int nElements = data1->Count;
		auto data1_x = hli::_mm_malloc_xmm(nElements);
		auto data2_x = hli::_mm_malloc_xmm(nElements);
		priv::copy_data(data1, data1_x);
		priv::copy_data(data2, data2_x);
		double result = _mm_corr_epu8_unmanaged(data1_x, data2_x, nElements);
		hli::_mm_free2(data1_x);
		hli::_mm_free2(data2_x);
		return result;
	}

#	pragma unmanaged
	double _mm_corr_pd_unmanaged(
		const std::tuple<const __int8 * const, const size_t>& data1,
		const std::tuple<const __int8 * const, const size_t>& data2,
		const size_t nElements)
	{
		return _mm_cvtsd_f64(hli::_mm_corr_pd(hli::_mm_cast_m128d(data1), hli::_mm_cast_m128d(data2), nElements));
	}

#	pragma managed
	double Class1::_mm_corr_pd(
		List<Double>^ data1,
		List<Double>^ data2)
	{
		int nElements = data1->Count;
		auto data1_x = hli::_mm_malloc_xmm(nElements * 8);
		auto data2_x = hli::_mm_malloc_xmm(nElements * 8);
		priv::copy_data(data1, data1_x);
		priv::copy_data(data2, data2_x);
		double result = _mm_corr_pd_unmanaged(data1_x, data2_x, nElements);
		hli::_mm_free2(data1_x);
		hli::_mm_free2(data2_x);
		return result;
	}

#	pragma unmanaged
	void _mm_corr_perm_epu8_unmanaged(
		const std::tuple<const __int8 * const, const size_t>& data1,
		const std::tuple<const __int8 * const, const size_t>& data2,
		const size_t nElements,
		const std::tuple<__int8 * const, const size_t>& results,
		const size_t nPermutations,
		std::array<UInt32, 4>& randInts)
	{
		__m128i randInts2 = _mm_set_epi32(randInts[3], randInts[2], randInts[1], randInts[0]);

		hli::_mm_corr_perm_epu8<8>(
			hli::_mm_cast_m128i(data1),
			hli::_mm_cast_m128i(data2),
			nElements,
			hli::_mm_cast_m128d(results),
			nPermutations,
			randInts2);

		randInts[3] = randInts2.m128i_u32[3];
		randInts[2] = randInts2.m128i_u32[2];
		randInts[1] = randInts2.m128i_u32[1];
		randInts[0] = randInts2.m128i_u32[0];
	}

#	pragma managed
	void Class1::_mm_corr_perm_epu8(
		List<Byte>^ data1,
		List<Byte>^ data2,
		List<Double>^ results,
		array<UInt32>^ randInts)
	{
		int nElements = data1->Count;
		auto data1_x = hli::_mm_malloc_xmm(nElements);
		auto data2_x = hli::_mm_malloc_xmm(nElements);
		priv::copy_data(data1, data1_x);
		priv::copy_data(data2, data2_x);

		int nPermutations = results->Count;
		auto results_x = hli::_mm_malloc_xmm(nPermutations * 8);

		std::array<UInt32, 4> randInts2 = { randInts[3], randInts[2], randInts[1], randInts[0] };
		_mm_corr_perm_epu8_unmanaged(data1_x, data2_x, nElements, results_x, nPermutations, randInts2);
		
		randInts[0] = randInts2[0];
		randInts[1] = randInts2[1];
		randInts[2] = randInts2[2];
		randInts[3] = randInts2[3];

		priv::copy_data(results_x, results);
		hli::_mm_free2(data1_x);
		hli::_mm_free2(data2_x);
		hli::_mm_free2(results_x);
	}

#	pragma unmanaged
	double _mm_mi_epu8_unmanaged(
		const std::tuple<const __int8 * const, const size_t>& data1,
		const std::tuple<const __int8 * const, const size_t>& data2,
		const size_t nElements)
	{
		return _mm_cvtsd_f64(hli::_mm_mi_epu8<8>(hli::_mm_cast_m128i(data1), hli::_mm_cast_m128i(data2), nElements));
	}

#	pragma managed
	double Class1::_mm_mi_epu8(
		List<Byte>^ data1,
		List<Byte>^ data2)
	{
		int nElements = data1->Count;
		auto data1_x = hli::_mm_malloc_xmm(nElements);
		auto data2_x = hli::_mm_malloc_xmm(nElements);
		priv::copy_data(data1, data1_x);
		priv::copy_data(data2, data2_x);
		double result = _mm_mi_epu8_unmanaged(data1_x, data2_x, nElements);
		hli::_mm_free2(data1_x);
		hli::_mm_free2(data2_x);
		return result;
	}

#	pragma unmanaged
	void _mm_mi_perm_epu8_unmanaged(
		const std::tuple<const __int8 * const, const size_t>& data1,
		const std::tuple<const __int8 * const, const size_t>& data2,
		const size_t nElements,
		const std::tuple<__int8 * const, const size_t>& results,
		const size_t nPermutations,
		std::array<UInt32, 4>& randInts)
	{
		__m128i randInts2 = _mm_set_epi32(randInts[3], randInts[2], randInts[1], randInts[0]);

		hli::_mm_mi_perm_epu8<8>(
			hli::_mm_cast_m128i(data1),
			hli::_mm_cast_m128i(data2),
			nElements,
			hli::_mm_cast_m128d(results),
			nPermutations,
			randInts2);

		randInts[3] = randInts2.m128i_u32[3];
		randInts[2] = randInts2.m128i_u32[2];
		randInts[1] = randInts2.m128i_u32[1];
		randInts[0] = randInts2.m128i_u32[0];
	}

#	pragma managed
	void Class1::_mm_mi_perm_epu8(
		List<Byte>^ data1,
		List<Byte>^ data2,
		List<Double>^ results,
		array<UInt32>^ randInts)
	{
		int nElements = data1->Count;
		auto data1_x = hli::_mm_malloc_xmm(nElements);
		auto data2_x = hli::_mm_malloc_xmm(nElements);
		priv::copy_data(data1, data1_x);
		priv::copy_data(data2, data2_x);

		int nPermutations = results->Count;
		auto results_x = hli::_mm_malloc_xmm(nPermutations * 8);

		std::array<UInt32, 4> randInts2 = { randInts[3], randInts[2], randInts[1], randInts[0] };
		_mm_mi_perm_epu8_unmanaged(data1_x, data2_x, nElements, results_x, nPermutations, randInts2);

		randInts[0] = randInts2[0];
		randInts[1] = randInts2[1];
		randInts[2] = randInts2[2];
		randInts[3] = randInts2[3];

		priv::copy_data(results_x, results);
		hli::_mm_free2(data1_x);
		hli::_mm_free2(data2_x);
		hli::_mm_free2(results_x);
	}
}
