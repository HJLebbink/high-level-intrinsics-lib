#pragma once

#include <tuple>
#include "emmintrin.h"

namespace hli {

	// shuffle: NO shuffle = _MM_SHUFFLE_EPI32_INT(3, 2, 1, 0)
	constexpr int _MM_SHUFFLE_EPI32_INT(int d, int c, int b, int a)
	{
		return ((d & 0b11) << 6) | ((c & 0b11) << 4) | ((b & 0b11) << 2) | ((a & 0b11) << 0);
	}

	// NO shuffle = 3, 2, 1, 0
	static const int _MM_SHUFFLE_EPI32_3210 = _MM_SHUFFLE_EPI32_INT(3, 2, 1, 0);



	/*
	Swap high 64-bits with low 64-bits
	*/
	inline __m128i _mm_swap_64(const __m128i d) {
		return _mm_shuffle_epi32(d, _MM_SHUFFLE_EPI32_INT(1, 0, 3, 2));
	}

	/*
	Swap high 64-bits with low 64-bits
	*/
	inline __m128d _mm_swap_64(const __m128d d) {
		return _mm_castsi128_pd(_mm_swap_64(_mm_castpd_si128(d)));
	}

	/*
	Swap high 64-bits with low 64-bits
	*/
	inline __m128 _mm_swap_64(const __m128 d) {
		return _mm_castsi128_ps(_mm_swap_64(_mm_castps_si128(d)));
	}

	inline size_t resizeNBytes(size_t nBytes, size_t align)
	{
		if (align == 16) {
			size_t result = nBytes + (((nBytes & 0b1111) == 0) ? 0 : (16 - (nBytes & 0b1111)));
			//std::cout << "INFO: resizeNBytes: align=" << align << "; nBytes=" << nBytes << "; result=" << result << std::endl;
			return result;
		}
		else if (align == 32) {
			return nBytes + (((nBytes & 0b11111) == 0) ? 0 : (32 - (nBytes & 0b11111)));
		}
		else if (align == 64) {
			return nBytes + (((nBytes & 0b111111) == 0) ? 0 : (64 - (nBytes & 0b111111)));
		}
		else {
			return nBytes;
		}
	}

	template <size_t ALIGN>
	inline size_t resizeNBytes(size_t nBytes)
	{
		if (ALIGN == 16) {
			size_t result = nBytes + (((nBytes & 0b1111) == 0) ? 0 : (16 - (nBytes & 0b1111)));
			//std::cout << "INFO: resizeNBytes: align=" << align << "; nBytes=" << nBytes << "; result=" << result << std::endl;
			return result;
		}
		else if (ALIGN == 32) {
			return nBytes + (((nBytes & 0b11111) == 0) ? 0 : (32 - (nBytes & 0b11111)));
		}
		else if (ALIGN == 64) {
			return nBytes + (((nBytes & 0b111111) == 0) ? 0 : (64 - (nBytes & 0b111111)));
		}
		else {
			return nBytes;
		}
	}

	inline std::tuple<__m128d * const, const size_t> deepCopy(const std::tuple<__m128d * const, const size_t>& in) {
		const size_t nBytes = std::get<1>(in);
		__m128d * const copy = static_cast<__m128d * const>(_mm_malloc(nBytes, 16));
		memcpy(copy, std::get<0>(in), nBytes);
		return std::make_tuple(copy, nBytes);
	}

	inline std::tuple<__m128i * const, const size_t> deepCopy(const std::tuple<__m128i * const, const size_t>& in) {
		const size_t nBytes = std::get<1>(in);
		__m128i * const copy = static_cast<__m128i * const>(_mm_malloc(nBytes, 16));
		memcpy(copy, std::get<0>(in), nBytes);
		return std::make_tuple(copy, nBytes);
	}

	template <class T>
	inline void _mm_free2(std::tuple<T * const, const size_t> t) {
		_mm_free(std::get<0>(t));
	}

	inline std::tuple<__m128d * const, const size_t> _mm_malloc_m128d(size_t nBytes)
	{
		const size_t nBytes2 = resizeNBytes<16>(nBytes);
		return std::make_tuple(static_cast<__m128d * const>(_mm_malloc(nBytes2, 16)), nBytes2);
	}
	inline std::tuple<__m128i * const, const size_t> _mm_malloc_m128i(size_t nBytes)
	{
		const size_t nBytes2 = resizeNBytes<16>(nBytes);
		return std::make_tuple(static_cast<__m128i * const>(_mm_malloc(nBytes2, 16)), nBytes2);
	}
}