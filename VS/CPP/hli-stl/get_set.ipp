#pragma once
#include "emmintrin.h"  // SSE
#include "immintrin.h"  // AVX

#include "tools.ipp"


namespace hli
{

	/******************************************
	* get
	*******************************************/

	inline double get_f64(const __m256& V, const size_t i)
	{
		//BOOST_ASSERT_MSG_HJ(i < 2, "");
		union
		{
			__m256 v;
			double a[4];
		} converter;
		converter.v = V;
		return converter.a[i];
	}

	inline unsigned __int64 get_u64(const __m256i& V, const size_t i)
	{
		//BOOST_ASSERT_MSG_HJ(i < 2, "");
		union
		{
			__m256i v;
			unsigned __int64 a[4];
		} converter;
		converter.v = V;
		return converter.a[i];
	}
	inline __int64 get_i64(const __m256i& V, const size_t i)
	{
		//BOOST_ASSERT_MSG_HJ(i < 2, "");
		union
		{
			__m256i v;
			__int64 a[4];
		} converter;
		converter.v = V;
		return converter.a[i];
	}


	inline double get_f64(const __m128d& V, const size_t i)
	{
		//BOOST_ASSERT_MSG_HJ(i < 2, "");
		union
		{
			__m128d v;
			double a[2];
		} converter;
		converter.v = V;
		return converter.a[i];
	}

	inline unsigned __int64 get_u64(const __m128i& V, const size_t i)
	{
		//BOOST_ASSERT_MSG_HJ(i < 2, "");
		union
		{
			__m128i v;
			unsigned __int64 a[2];
		} converter;
		converter.v = V;
		return converter.a[i];
	}
	inline __int64 get_i64(const __m128i& V, const size_t i)
	{
		//BOOST_ASSERT_MSG_HJ(i < 2, "");
		union
		{
			__m128i v;
			__int64 a[2];
		} converter;
		converter.v = V;
		return converter.a[i];
	}

	inline float get_f32(const __m128& V, const size_t i)
	{
		//BOOST_ASSERT_MSG_HJ(i < 4, "");
		union
		{
			__m128 v;
			float a[4];
		} converter;
		converter.v = V;
		return converter.a[i];
	}
	inline __int32 get_i32(const __m128i& V, const size_t i)
	{
		//BOOST_ASSERT_MSG_HJ(i < 4, "");
		union
		{
			__m128i v;
			__int32 a[4];
		} converter;
		converter.v = V;
		return converter.a[i];
	}
	inline unsigned __int32 get_u32(const __m128i& V, const size_t i)
	{
		//BOOST_ASSERT_MSG_HJ(i < 4, "");
		union
		{
			__m128i v;
			unsigned __int32 a[4];
		} converter;
		converter.v = V;
		return converter.a[i];
	}
	inline signed __int16 get_i16(const __m128i& V, const size_t i)
	{
		//BOOST_ASSERT_MSG_HJ(i < 8, "");
		union
		{
			__m128i v;
			signed __int16 a[8];
		} converter;
		converter.v = V;
		return converter.a[i];
	}
	inline signed __int16 get_i16(const __m64& V, const size_t i)
	{
		//BOOST_ASSERT_MSG_HJ(i < 4, "");
		union
		{
			__m64 v;
			signed __int16 a[4];
		} converter;
		converter.v = V;
		return converter.a[i];
	}
	inline U16 get_u16(const __m128i& V, const size_t i)
	{
		//BOOST_ASSERT_MSG_HJ(i < 8, "");
		union
		{
			__m128i v;
			U16 a[8];
		} converter;
		converter.v = V;
		return converter.a[i];
	}
	inline U8 get_u8(const __m128i& V, const size_t i)
	{
		//BOOST_ASSERT_MSG_HJ(i < 16, "");
		union
		{
			__m128i v;
			U8 a[16];
		} converter;
		converter.v = V;
		return converter.a[i];
	}
	inline U8 get_u8(const __m128& V, const size_t i)
	{
		//BOOST_ASSERT_MSG_HJ(i < 16, "");
		union
		{
			__m128 v;
			U8 a[16];
		} converter;
		converter.v = V;
		return converter.a[i];
	}
	/*	inline U8 get_u8(const __m256i V, const size_t i)
	{
	BOOST_ASSERT_MSG_HJ(i < 32, "");
	union
	{
	__m256i v;
	U8 a[32];
	} converter;
	converter.v = V;
	return converter.a[i];
	}
	*/	inline signed __int8 get_i8(const __m128i& V, const size_t i)
	{
		//BOOST_ASSERT_MSG_HJ(i < 16, "");
		union
		{
			__m128i v;
			signed __int8 a[16];
		} converter;
		converter.v = V;
		return converter.a[i];
	}


	/******************************************
	* set
	*******************************************/
	inline void set_u8(__m128i& v, const size_t i, const U8 d)
	{
		//BOOST_ASSERT_MSG_HJ(i < 16, "");
		union
		{
			__m128i v;
			U8 a[16];
		} converter;
		converter.v = v;
		converter.a[i] = d;
		v = converter.v;
	}
	inline void set_u16(__m128i& v, const size_t i, const U16 d)
	{
		//BOOST_ASSERT_MSG_HJ(i < 8, "");
		union
		{
			__m128i v;
			U16 a[8];
		} converter;
		converter.v = v;
		converter.a[i] = d;
		v = converter.v;
	}
	inline void set_u32(__m128i& v, const size_t i, const unsigned __int32 d)
	{
		//BOOST_ASSERT_MSG_HJ(i < 4, "");

		union
		{
			__m128i v;
			unsigned __int32 a[4];
		} converter;
		converter.v = v;
		converter.a[i] = d;
		v = converter.v;
	}
	inline void set_u64(__m128i& v, const size_t i, const unsigned __int64 d)
	{
		//BOOST_ASSERT_MSG_HJ(i < 2, "");

		union
		{
			__m128i v;
			unsigned __int64 a[2];
		} converter;
		converter.v = v;
		converter.a[i] = d;
		v = converter.v;
	}
}