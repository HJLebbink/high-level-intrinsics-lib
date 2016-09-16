// This is the main DLL file.

#include "stdafx.h"
#include "intrin.h"
#include <vector>
#include <tuple>

#include "StatsLibCli.h"

#pragma unmanaged
#include "../../CPP/hli-lib/_mm_hadd_epu8.h"
#include "../../CPP/hli-lib/_mm_corr_pd.h"
#include "../../CPP/hli-lib/_mm_corr_epu8.h"

using namespace System;
using namespace System::Collections::Generic;

namespace StatsLibCli {

	namespace priv {

#		pragma managed
		void copy_data(List<Byte>^ data, const std::tuple<__int8 * const, const size_t>& data2)
		{
			__int8 * const ptr = std::get<0>(data2);

			if (true) {
				int nElements = data->Count;
				for (int i = 0; i < nElements; ++i) {
					ptr[i] = data[i];
				}
				size_t nBytes = std::get<1>(data2);
				int nElements2 = static_cast<int>(nBytes);
				int tail = nElements2 - nElements;
				for (int i = nElements2 - tail; i < nElements2; ++i) {
					ptr[i] = 0;
				}
			} else {

			}
		}

#		pragma managed
		void copy_data(List<Double>^ data, const std::tuple<__int8 * const, const size_t>& data2)
		{
			double * const ptr = reinterpret_cast<double * const>(std::get<0>(data2));

			if (true) {
				int nElements = data->Count;

				for (int i = 0; i < nElements; ++i) {
					ptr[i] = data[i];
				}
			} else {

			}
		}
	}

#	pragma managed
	array<Double>^ Class1::test2_managed(
		List<Byte> data, 
		const Int64 nPermutations)
	{
		const size_t size_data = data.Count;// TODO create proper size
		__m128i * const data2 = static_cast<__m128i * const>(_mm_malloc(size_data, 16));
		__m128d * const results_raw = static_cast<__m128d * const>(_mm_malloc(nPermutations * 64, 16));


		array<Double>^ result = gcnew array<Double>(static_cast<int>(nPermutations));
		double * const resultsRawDouble = reinterpret_cast<double * const>(results_raw);
		for (int permutation = 0; permutation < static_cast<int>(nPermutations); ++permutation) {
			result[permutation] = resultsRawDouble[permutation];
		}

		_mm_free(data2);
		return result;
	}

#	pragma unmanaged
	int _mm_hadd_epu8_unmanaged(
		const std::tuple<const __int8 * const, const size_t>& data)
	{
		return _mm_cvtsi128_si32(hli::_mm_hadd_epu8<8>(hli::_mm_cast_m128i(data)));
	}

#	pragma managed
	int Class1::_mm_hadd_epu8(
		List<Byte>^ data)
	{
		auto data_x = hli::_mm_malloc_xmm(data->Count);
		priv::copy_data(data, data_x);
		int result = _mm_hadd_epu8_unmanaged(data_x);
		hli::_mm_free2(data_x);
		return result;
	}

#	pragma unmanaged
	double _mm_corr_pd_unmanaged(
		const std::tuple<const __int8 * const, const size_t>& data1,
		const std::tuple<const __int8 * const, const size_t>& data2)
	{
		return _mm_cvtsd_f64(hli::_mm_corr_pd(hli::_mm_cast_m128d(data1), hli::_mm_cast_m128d(data2)));
	}

#	pragma managed
	double Class1::_mm_corr_pd(
		List<Double>^ data1, 
		List<Double>^ data2)
	{
		auto data1_x = hli::_mm_malloc_xmm(data1->Count * 8);
		auto data2_x = hli::_mm_malloc_xmm(data2->Count * 8);
		priv::copy_data(data1, data1_x);
		priv::copy_data(data2, data2_x);
		double result = _mm_corr_pd_unmanaged(data1_x, data2_x);
		hli::_mm_free2(data1_x);
		hli::_mm_free2(data2_x);
		return result;
	}

#	pragma unmanaged
	double _mm_corr_epu8_unmanaged(
		const std::tuple<const __int8 * const, const size_t>& data1,
		const std::tuple<const __int8 * const, const size_t>& data2)
	{
		return _mm_cvtsd_f64(hli::_mm_corr_epu8<8>(hli::_mm_cast_m128i(data1), hli::_mm_cast_m128i(data2)));
	}

#	pragma managed
	double Class1::_mm_corr_epu8(
		List<Byte>^ data1, 
		List<Byte>^ data2)
	{
		auto data1_x = hli::_mm_malloc_xmm(data1->Count);
		auto data2_x = hli::_mm_malloc_xmm(data2->Count);
		priv::copy_data(data1, data1_x);
		priv::copy_data(data2, data2_x);
		double result = _mm_corr_epu8_unmanaged(data1_x, data2_x);
		hli::_mm_free2(data1_x);
		hli::_mm_free2(data2_x);
		return result;
	}
}