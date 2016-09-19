// StatsLibCli.h

#pragma once
#include <vector>

using namespace System;
using namespace System::Collections::Generic;

namespace StatsLibCli {

#	pragma managed
	public ref class Class1
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

		static double _mm_mi_epu8(
			List<Byte>^ data1,
			List<Byte>^ data2);

		static void _mm_mi_perm_epu8(
			List<Byte>^ data1,
			List<Byte>^ data2,
			List<Double>^ results,
			array<UInt32>^ randInts);
	};
}
