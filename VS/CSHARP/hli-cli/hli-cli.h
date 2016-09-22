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
			List<Byte>^ data);

		static double _mm_corr_epu8(
			List<Byte>^ data1, 
			List<Byte>^ data2);
			
		static double _mm_corr_pd(
			List<Double>^ data1, 
			List<Double>^ data2);

		static void _mm_corr_perm_epu8(
			List<Byte>^ data1,
			List<Byte>^ data2,
			List<Double>^ results,
			array<UInt32>^ randInts);

		static double _mm_entropy_epu8(
			List<Byte>^ data,
			int nBits);

		static double _mm_entropy_epu8(
			List<Byte>^ data1,
			int nBits1,
			List<Byte>^ data2,
			int nBits2);

		static double _mm_mi_epu8(
			List<Byte>^ data1,
			int nBits1,
			List<Byte>^ data2,
			int nBits2);

		static void _mm_mi_perm_epu8(
			List<Byte>^ data1,
			int nBits1,
			List<Byte>^ data2,
			int nBits2,
			List<Double>^ results,
			array<UInt32>^ randInts);
	};
}
