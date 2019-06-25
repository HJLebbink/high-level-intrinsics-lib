#pragma once

#include <tuple>
#include "emmintrin.h"

#include "intrin.h"

namespace hli
{

	using U8 = unsigned __int8;
	using U16 = unsigned __int16;



	// shuffle: NO shuffle = _MM_SHUFFLE_EPI32_INT(3, 2, 1, 0)
	//constexpr const int _MM_SHUFFLE_EPI32_INT(int d, int c, int b, int a)
	//{
	//	return ((d & 0b11) << 6) | ((c & 0b11) << 4) | ((b & 0b11) << 2) | ((a & 0b11) << 0);
	//}
	#	define _MM_SHUFFLE_EPI32_INT(d, c, b, a) ((((d) & 0b11) << 6) | (((c) & 0b11) << 4) | (((b) & 0b11) << 2) | (((a) & 0b11) << 0))


		// NO shuffle = 3, 2, 1, 0
	static const int _MM_SHUFFLE_EPI32_3210 = _MM_SHUFFLE_EPI32_INT(3, 2, 1, 0);



	/*
	Swap high 64-bits with low 64-bits
	*/
	inline __m128i _mm_swap_64(const __m128i d)
	{
		return _mm_shuffle_epi32(d, _MM_SHUFFLE_EPI32_INT(1, 0, 3, 2));
	}

	/*
	Swap high 64-bits with low 64-bits
	*/
	inline __m128d _mm_swap_64(const __m128d d)
	{
		return _mm_castsi128_pd(_mm_swap_64(_mm_castpd_si128(d)));
	}

	/*
	Swap high 64-bits with low 64-bits
	*/
	inline __m128 _mm_swap_64(const __m128 d)
	{
		return _mm_castsi128_ps(_mm_swap_64(_mm_castps_si128(d)));
	}

	template <int ALIGN>
	inline int resizeNBytes(int nBytes)
	{
		switch (nBytes)
		{
			case 16: return nBytes + (((nBytes & 0b1111) == 0) ? 0 : (16 - (nBytes & 0b1111)));
			case 32: return nBytes + (((nBytes & 0b11111) == 0) ? 0 : (32 - (nBytes & 0b11111)));
			case 64: return nBytes + (((nBytes & 0b111111) == 0) ? 0 : (64 - (nBytes & 0b111111)));
			default: return nBytes;
		}
	}

	inline int resizeNBytes(int nBytes, int align)
	{
		switch (align)
		{
			case 16: return resizeNBytes<16>(nBytes);
			case 32: return resizeNBytes<32>(nBytes);
			case 64: return resizeNBytes<64>(nBytes);
			default: return nBytes;
		}
	}

	template <class T>
	inline void copy(
		const std::tuple<const T * const, const int>& src,
		const std::tuple<T * const, const int>& dst)
	{
		memcpy(std::get<0>(dst), std::get<0>(src), std::get<1>(dst));
	}


	inline std::tuple<__m128d * const, const int> deepCopy(
		const std::tuple<const __m128d * const, const int>& in)
	{
		const int nBytes = std::get<1>(in);
		__m128d * const copy = static_cast<__m128d * const>(_mm_malloc(nBytes, 16));
		memcpy(copy, std::get<0>(in), nBytes);
		return std::make_tuple(copy, nBytes);
	}

	inline std::tuple<__m128i * const, const int> deepCopy(
		const std::tuple<const __m128i * const, const int>& in)
	{
		const int nBytes = std::get<1>(in);
		__m128i * const copy = static_cast<__m128i * const>(_mm_malloc(nBytes, 16));
		memcpy(copy, std::get<0>(in), nBytes);
		return std::make_tuple(copy, nBytes);
	}

	template <class T, class S>
	inline void _mm_free2(const std::tuple<T * const, const S>& t)
	{
		const void * const ptr = reinterpret_cast<const void *>(std::get<0>(t));
		_mm_free(const_cast<void *>(ptr));
	}

	inline std::tuple<U8 * const, const int> _mm_malloc_xmm(int nBytes)
	{
		const int nBytes2 = resizeNBytes<16>(nBytes);
		return std::make_tuple(static_cast<U8 * const>(_mm_malloc(nBytes2, 16)), nBytes2);
	}
	inline std::tuple<U8 * const, const int> _mm_malloc_ymm(int nBytes)
	{
		const int nBytes2 = resizeNBytes<32>(nBytes);
		return std::make_tuple(static_cast<U8 * const>(_mm_malloc(nBytes2, 32)), nBytes2);
	}
	inline std::tuple<U8 * const, const int> _mm_malloc_zmm(int nBytes)
	{
		const int nBytes2 = resizeNBytes<64>(nBytes);
		return std::make_tuple(static_cast<U8 * const>(_mm_malloc(nBytes2, 64)), nBytes2);
	}


	inline std::tuple<__m128d * const, const int> _mm_cast_m128d(
		const std::tuple<U8 * const, const int>& data)
	{
		return std::make_tuple(reinterpret_cast<__m128d * const>(std::get<0>(data)), std::get<1>(data));
	}
	inline std::tuple<const __m128d * const, const int> _mm_cast_m128d_c(
		const std::tuple<const U8 * const, const int>& data)
	{
		return std::make_tuple(reinterpret_cast<const __m128d * const>(std::get<0>(data)), std::get<1>(data));
	}
	inline std::tuple<__m128i * const, const int> _mm_cast_m128i(
		const std::tuple<U8 * const, const int>& data)
	{
		return std::make_tuple(reinterpret_cast<__m128i * const>(std::get<0>(data)), std::get<1>(data));
	}
	inline std::tuple<const __m128i * const, const int> _mm_cast_m128i(
		const std::tuple<const U8 * const, const int>& data)
	{
		return std::make_tuple(reinterpret_cast<const __m128i * const>(std::get<0>(data)), std::get<1>(data));
	}


	inline std::tuple<__m512d * const, const int> _mm_cast_m512d(
		const std::tuple<U8 * const, const int>& data)
	{
		return std::make_tuple(reinterpret_cast<__m512d * const>(std::get<0>(data)), std::get<1>(data));
	}
	inline std::tuple<const __m512d * const, const int> _mm_cast_m512d(
		const std::tuple<const U8 * const, const int>& data)
	{
		return std::make_tuple(reinterpret_cast<const __m512d * const>(std::get<0>(data)), std::get<1>(data));
	}
	inline std::tuple<__m512i * const, const int> _mm_cast_m512i(
		const std::tuple<U8 * const, const int>& data)
	{
		return std::make_tuple(reinterpret_cast<__m512i * const>(std::get<0>(data)), std::get<1>(data));
	}
	inline std::tuple<const __m512i * const, const int> _mm_cast_m512i(
		const std::tuple<const U8 * const, const int>& data)
	{
		return std::make_tuple(reinterpret_cast<const __m512i * const>(std::get<0>(data)), std::get<1>(data));
	}



	inline std::tuple<__m128d * const, const int> _mm_malloc_m128d(int nBytes)
	{
		return _mm_cast_m128d(_mm_malloc_xmm(nBytes));
	}
	inline std::tuple<__m128i * const, const int> _mm_malloc_m128i(int nBytes)
	{
		return _mm_cast_m128i(_mm_malloc_xmm(nBytes));
	}

	inline std::tuple<__m512d * const, const int> _mm_malloc_m512d(int nBytes)
	{
		return _mm_cast_m512d(_mm_malloc_zmm(nBytes));
	}
	inline std::tuple<__m512i * const, const int> _mm_malloc_m512i(int nBytes)
	{
		return _mm_cast_m512i(_mm_malloc_zmm(nBytes));
	}


	inline std::tuple<__m128d, __m128d, __m128d, __m128d, __m128d, __m128d, __m128d, __m128d> _mm_cvt_epu8_pd(__m128i data)
	{
		const __m128i d1 = _mm_cvtepu8_epi32(data);
		const __m128d r0 = _mm_cvtepi32_pd(d1);
		const __m128d r1 = _mm_cvtepi32_pd(_mm_swap_64(d1));

		const __m128i d2 = _mm_cvtepu8_epi32(_mm_shuffle_epi32(data, _MM_SHUFFLE_EPI32_INT(1, 1, 1, 1)));
		const __m128d r2 = _mm_cvtepi32_pd(d2);
		const __m128d r3 = _mm_cvtepi32_pd(_mm_swap_64(d2));

		const __m128i d3 = _mm_cvtepu8_epi32(_mm_shuffle_epi32(data, _MM_SHUFFLE_EPI32_INT(2, 2, 2, 2)));
		const __m128d r4 = _mm_cvtepi32_pd(d3);
		const __m128d r5 = _mm_cvtepi32_pd(_mm_swap_64(d3));

		const __m128i d4 = _mm_cvtepu8_epi32(_mm_shuffle_epi32(data, _MM_SHUFFLE_EPI32_INT(3, 3, 3, 3)));
		const __m128d r6 = _mm_cvtepi32_pd(d4);
		const __m128d r7 = _mm_cvtepi32_pd(_mm_swap_64(d4));

		return std::make_tuple(r0, r1, r2, r3, r4, r5, r6, r7);
	}

	inline std::tuple<__m128d, __m128d, __m128d, __m128d, __m128d, __m128d, __m128d, __m128d> _mm_sub_pd(
		const std::tuple<__m128d, __m128d, __m128d, __m128d, __m128d, __m128d, __m128d, __m128d>& tup,
		const __m128d value)
	{
		const __m128d r0 = _mm_sub_pd(std::get<0>(tup), value);
		const __m128d r1 = _mm_sub_pd(std::get<1>(tup), value);
		const __m128d r2 = _mm_sub_pd(std::get<2>(tup), value);
		const __m128d r3 = _mm_sub_pd(std::get<3>(tup), value);
		const __m128d r4 = _mm_sub_pd(std::get<4>(tup), value);
		const __m128d r5 = _mm_sub_pd(std::get<5>(tup), value);
		const __m128d r6 = _mm_sub_pd(std::get<6>(tup), value);
		const __m128d r7 = _mm_sub_pd(std::get<7>(tup), value);
		return std::make_tuple(r0, r1, r2, r3, r4, r5, r6, r7);
	}

	inline double getDouble(
		const std::tuple<const __m128d * const, const int>& data,
		const int i)
	{
		const double * const ptr = reinterpret_cast<const double * const>(std::get<0>(data));
		return ptr[i];
	}
}