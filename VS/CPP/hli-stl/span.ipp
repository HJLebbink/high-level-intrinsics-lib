#include "intrin.h"

namespace hj {

	template <class T>
	using span = struct { 
		typename T * const ptr; 
		size_t length;

		size_t nBytes() { return this.length << 4; }

	};

	template <class T>
	inline span<T> _mm_malloc_xmm(size_t nElements) {
		const size_t nBytes2 = resizeNBytes<16>(nBytes);
		
		span<__int8> result = new span<__int8>();
		result.ptr = static_cast<__int8 * const>(_mm_malloc(nBytes2, 16)), nBytes2);
		result.length = 
		
		{;
	}


	template <class T>
	void test(const span<T>& data) {

		for (int i = 0; i < data.length; ++i) {

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

}
