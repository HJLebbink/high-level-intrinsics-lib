#pragma once

#include <string>
#include <sstream>
#include <bitset>
#include <iomanip>      // std::setprecision

#include "get_set.ipp"

namespace hli
{



	inline std::string toString_f64(const __m256 i)
	{
		std::ostringstream stringStream;
		for (size_t j = 0; j < 4; ++j) stringStream << std::setprecision(16) << get_f64(i, j) << " ";
		return stringStream.str();
	}
	inline std::string toString_u64(const __m256i i)
	{
		std::ostringstream stringStream;
		for (size_t j = 0; j < 4; ++j) stringStream << get_u64(i, j) << " ";
		return stringStream.str();
	}
	inline std::string toString_i64(const __m256i i)
	{
		std::ostringstream stringStream;
		for (size_t j = 0; j < 4; ++j) stringStream << get_i64(i, j) << " ";
		return stringStream.str();
	}


	inline std::string toString_f64(const __m128d i)
	{
		std::ostringstream stringStream;
		for (size_t j = 0; j < 2; ++j) stringStream << std::setprecision(16) << get_f64(i, j) << " ";
		return stringStream.str();
	}
	inline std::string toString_u64(const __m128i i)
	{
		std::ostringstream stringStream;
		for (size_t j = 0; j < 2; ++j) stringStream << get_u64(i, j) << " ";
		return stringStream.str();
	}
	inline std::string toString_i64(const __m128i i)
	{
		std::ostringstream stringStream;
		for (size_t j = 0; j < 2; ++j) stringStream << get_i64(i, j) << " ";
		return stringStream.str();
	}


	inline std::string toString_f32(const __m128 i)
	{
		std::ostringstream stringStream;
		for (size_t j = 0; j < 4; ++j) stringStream << std::setprecision(16) << get_f32(i, j) << " ";
		return stringStream.str();
	}
	inline std::string toString_u32(const __m128i i)
	{
		std::ostringstream stringStream;
		for (size_t j = 0; j < 4; ++j) stringStream << get_u32(i, j) << " ";
		return stringStream.str();
	}
	inline std::string toString_i32(const __m128i i)
	{
		std::ostringstream stringStream;
		for (size_t j = 0; j < 4; ++j) stringStream << get_i32(i, j) << " ";
		return stringStream.str();
	}

	inline std::string toString_i16(const __m64 i)
	{
		std::ostringstream stringStream;
		for (size_t j = 0; j < 4; ++j) stringStream << get_i16(i, j) << " ";
		return stringStream.str();
	}
	inline std::string toString_u16(const __m128i i)
	{
		std::ostringstream stringStream;
		for (size_t j = 0; j < 8; ++j) stringStream << get_u16(i, j) << " ";
		return stringStream.str();
	}

	inline std::string toString_u8(const __m128i i)
	{
		std::ostringstream stringStream;
		for (size_t j = 0; j < 16; ++j) stringStream << static_cast<int>(get_u8(i, j)) << " ";
		return stringStream.str();
	}
	inline std::string toString_u8(const __m128 i)
	{
		std::ostringstream stringStream;
		for (size_t j = 0; j < 16; ++j) stringStream << static_cast<int>(get_u8(i, j)) << " ";
		return stringStream.str();
	}
	/*	inline std::string toString_u8(const __m256i i)
	{
	std::ostringstream stringStream;
	for (size_t j = 0; j < 32; ++j) stringStream << static_cast<int>(i.m256i_u8[j]) << " ";
	return stringStream.str();
	}
	*/
	inline std::string toString_i8(const __m128i i)
	{
		std::ostringstream stringStream;
		for (size_t j = 0; j < 16; ++j) stringStream << static_cast<int>(get_i8(i, j)) << " ";
		return stringStream.str();
	}

	inline std::string toBinary_u8(const __m128i i)
	{
		std::ostringstream stringStream;
		for (size_t j = 0; j < 16; ++j) stringStream << std::bitset<8>(get_u8(i, j)) << " ";
		return stringStream.str();
	}
	/*	inline std::string toBinary_u8(const __m256i i)
	{
	std::ostringstream stringStream;
	for (size_t j = 0; j < 32; ++j) stringStream << std::bitset<8>(get_u8(i, j)) << " ";
	return stringStream.str();
	}
	*/
	inline std::string toBinary_u32(const __m128i i)
	{
		std::ostringstream stringStream;
		for (size_t j = 0; j < 4; ++j) stringStream << std::bitset<32>(get_u32(i, j)) << " ";
		return stringStream.str();
	}
	inline std::string toBinary_u8(const __m128 i)
	{
		std::ostringstream stringStream;
		for (size_t j = 0; j < 16; ++j) stringStream << std::bitset<8>(get_u8(i, j)) << " ";
		return stringStream.str();
	}


	template <size_t NBITS>
	inline std::string toBinary(const unsigned long long i)
	{
		std::ostringstream stringStream;
		std::bitset<NBITS> x(i);
		stringStream << x;
		return stringStream.str();
	}

	inline std::string toBinary(const __m128i i)
	{
		return toBinary_u8(i);
	}
	inline std::string toBinary(const float floatVar)
	{
		#ifdef _MSC_VER
		int fl = *(int*)&floatVar;
		std::ostringstream stringStream;
		for (int i = 31; i >= 0; --i)
		{
			if (i == 22) stringStream << ":";
			stringStream << ((((1 << i) & fl) != 0) ? "1" : "0");
		}
		return stringStream.str();
		#else
		DEBUG_BREAK();
		return "";
		#endif
	}
	inline std::string toBinary(const double doubleVar)
	{
		#ifdef _MSC_VER
		long long fl = *(long long*)&doubleVar;
		std::ostringstream stringStream;
		for (int i = 63; i >= 0; --i)
		{
			if (i == 51) stringStream << ":";
			stringStream << ((((1ull << i) & fl) != 0) ? "1" : "0");
		}
		return stringStream.str();
		#else
		DEBUG_BREAK();
		return "";
		#endif
	}
}
