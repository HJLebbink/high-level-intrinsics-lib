#pragma once

#include <tuple>


namespace hli {

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