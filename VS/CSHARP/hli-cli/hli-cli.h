// StatsLibCli.h

#pragma once
#include <vector>

using namespace System;
using namespace System::Collections::Generic;

namespace hli_cli {

#	pragma managed
	public ref class HliCli
	{
	public:
		static int _mm_hadd_epu8(
			List<Byte>^ data,
			bool HAS_MVs
		);

		static double _mm_corr_epu8(
			List<Byte>^ data1, 
			List<Byte>^ data2,
			bool HAS_MVs);
			
		static double _mm_corr_pd(
			List<Double>^ data1, 
			List<Double>^ data2,
			bool HAS_MVs);

		static void _mm_corr_perm_epu8(
			List<Byte>^ data1,
			List<Byte>^ data2,
			bool HAS_MVs,
			List<Double>^ results,
			array<UInt32>^ randInts);

		static double _mm_entropy_epu8(
			List<Byte>^ data,
			int nBits,
			bool HAS_MVs);

		static double _mm_entropy_epu8(
			List<Byte>^ data1,
			int nBits1,
			List<Byte>^ data2,
			int nBits2,
			bool HAS_MVs);

		static double _mm_mi_epu8(
			List<Byte>^ data1,
			int nBits1,
			List<Byte>^ data2,
			int nBits2,
			bool HAS_MVs);

		static void _mm_mi_perm_epu8(
			List<Byte>^ data1,
			int nBits1,
			List<Byte>^ data2,
			int nBits2,
			bool HAS_MVs,
			List<Double>^ results,
			array<UInt32>^ randInts);

		static void _mm_mi_corr_perm_epu8(
			List<Byte>^ data1,
			int nBits1,
			List<Byte>^ data2,
			int nBits2,
			bool HAS_MVs,
			List<Double>^ results_mi,
			List<Double>^ results_corr,
			array<UInt32>^ randInts);

	};
}
