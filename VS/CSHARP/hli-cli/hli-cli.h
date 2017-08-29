// StatsLibCli.h

#pragma once
#include <array>
#include <list>


using namespace System;
using namespace System::Collections::Generic;

namespace hli_cli {

#	pragma managed
	public ref class HliCli
	{
	public:

		// Horizontally add adjacent pairs of 8-bit unsigned integers in data, and pack the unsigned 8-bit results in dst.
		static int _mm_hadd_epu8(
			List<Byte>^ data,
			bool has_missing_values
		);

		static double _mm_corr_epu8(
			List<Byte>^ data1, 
			int nBits1,
			List<Byte>^ data2,
			int nBits2,
			bool has_missing_values);
			
		static double _mm_corr_pd(
			List<Double>^ data1, 
			List<Double>^ data2,
			bool has_missing_values);

		static void _mm_corr_epu8_perm(
			List<Byte>^ data1,
			int nBits1,
			List<Byte>^ data2,
			int nBits2,
			bool has_missing_values,
			List<Double>^ results,
			array<UInt32>^ randInts);

		static double _mm_entropy_epu8(
			List<Byte>^ data,
			int nBits,
			bool has_missing_values);

		static double _mm_entropy_epu8(
			List<Byte>^ data1,
			int nBits1,
			List<Byte>^ data2,
			int nBits2,
			bool has_missing_values);

		static double _mm_mi_epu8(
			List<Byte>^ data1,
			int nBits1,
			List<Byte>^ data2,
			int nBits2,
			bool has_missing_values);

		static void _mm_mi_epu8_perm(
			List<Byte>^ data1,
			int nBits1,
			List<Byte>^ data2,
			int nBits2,
			bool has_missing_values,
			List<Double>^ results,
			array<UInt32>^ randInts);

		static void _mm_mi_corr_epu8_perm(
			List<Byte>^ data1,
			int nBits1,
			List<Byte>^ data2,
			int nBits2,
			bool has_missing_values,
			List<Double>^ results_mi,
			List<Double>^ results_corr,
			array<UInt32>^ randInts);

	};
}
