#pragma once
#include "emmintrin.h"  // sse

namespace hli {

	/******************************************
	* get
	*******************************************/
	inline double get_f64(const __m128& V, const size_t i)
	{
		//BOOST_ASSERT_MSG_HJ(i < 2, "");
		union {
			__m128 v;
			double a[2];
		} converter;
		converter.v = V;
		return converter.a[i];
	}

	inline unsigned long long get_u64(const __m128i& V, const size_t i)
	{
		//BOOST_ASSERT_MSG_HJ(i < 2, "");
		union {
			__m128i v;
			unsigned long long a[2];
		} converter;
		converter.v = V;
		return converter.a[i];
	}
	inline long long get_i64(const __m128i& V, const size_t i)
	{
		//BOOST_ASSERT_MSG_HJ(i < 2, "");
		union {
			__m128i v;
			long long a[2];
		} converter;
		converter.v = V;
		return converter.a[i];
	}

	inline float get_f32(const __m128& V, const size_t i)
	{
		//BOOST_ASSERT_MSG_HJ(i < 4, "");
		union {
			__m128 v;
			float a[4];
		} converter;
		converter.v = V;
		return converter.a[i];
	}
	inline int get_i32(const __m128i& V, const size_t i)
	{
		//BOOST_ASSERT_MSG_HJ(i < 4, "");
		union {
			__m128i v;
			int a[4];
		} converter;
		converter.v = V;
		return converter.a[i];
	}
	inline unsigned int get_u32(const __m128i& V, const size_t i)
	{
		//BOOST_ASSERT_MSG_HJ(i < 4, "");
		union {
			__m128i v;
			unsigned int a[4];
		} converter;
		converter.v = V;
		return converter.a[i];
	}
	inline short get_i16(const __m128i& V, const size_t i)
	{
		//BOOST_ASSERT_MSG_HJ(i < 8, "");
		union {
			__m128i v;
			short a[8];
		} converter;
		converter.v = V;
		return converter.a[i];
	}
	inline short get_i16(const __m64& V, const size_t i)
	{
		//BOOST_ASSERT_MSG_HJ(i < 4, "");
		union {
			__m64 v;
			short a[4];
		} converter;
		converter.v = V;
		return converter.a[i];
	}
	inline unsigned short get_u16(const __m128i& V, const size_t i)
	{
		//BOOST_ASSERT_MSG_HJ(i < 8, "");
		union {
			__m128i v;
			unsigned short a[8];
		} converter;
		converter.v = V;
		return converter.a[i];
	}
	inline unsigned char get_u8(const __m128i& V, const size_t i)
	{
		//BOOST_ASSERT_MSG_HJ(i < 16, "");
		union {
			__m128i v;
			unsigned char a[16];
		} converter;
		converter.v = V;
		return converter.a[i];
	}
	inline unsigned char get_u8(const __m128& V, const size_t i)
	{
		//BOOST_ASSERT_MSG_HJ(i < 16, "");
		union {
			__m128 v;
			unsigned char a[16];
		} converter;
		converter.v = V;
		return converter.a[i];
	}
	/*	inline unsigned char get_u8(const __m256i V, const size_t i)
	{
	BOOST_ASSERT_MSG_HJ(i < 32, "");
	union
	{
	__m256i v;
	unsigned char a[32];
	} converter;
	converter.v = V;
	return converter.a[i];
	}
	*/	inline signed char get_i8(const __m128i& V, const size_t i)
	{
		//BOOST_ASSERT_MSG_HJ(i < 16, "");
		union {
			__m128i v;
			signed char a[16];
		} converter;
		converter.v = V;
		return converter.a[i];
	}


	/******************************************
	* set
	*******************************************/
	inline void set_u8(__m128i& v, const size_t i, const unsigned char d)
	{
		//BOOST_ASSERT_MSG_HJ(i < 16, "");
		union {
			__m128i v;
			unsigned char a[16];
		} converter;
		converter.v = v;
		converter.a[i] = d;
		v = converter.v;
	}
	inline void set_u16(__m128i& v, const size_t i, const unsigned short d)
	{
		//BOOST_ASSERT_MSG_HJ(i < 8, "");
		union {
			__m128i v;
			unsigned short a[8];
		} converter;
		converter.v = v;
		converter.a[i] = d;
		v = converter.v;
	}
	inline void set_u32(__m128i& v, const size_t i, const unsigned int d)
	{
		//BOOST_ASSERT_MSG_HJ(i < 4, "");

		union {
			__m128i v;
			unsigned int a[4];
		} converter;
		converter.v = v;
		converter.a[i] = d;
		v = converter.v;
	}
	inline void set_u64(__m128i& v, const size_t i, const unsigned long long d)
	{
		//BOOST_ASSERT_MSG_HJ(i < 2, "");

		union {
			__m128i v;
			unsigned long long a[2];
		} converter;
		converter.v = v;
		converter.a[i] = d;
		v = converter.v;
	}
}