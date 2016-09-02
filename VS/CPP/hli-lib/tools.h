#pragma once

namespace hli {
	inline size_t resizeNBytes(size_t nBytes, size_t align)
	{
		if (align == 16) {
			size_t result = nBytes + (((nBytes & 0b1111) == 0) ? 0 : (16 - (nBytes & 0b1111)));
			//std::cout << "INFO: resizeNBytes: align=" << align << "; nBytes=" << nBytes << "; result=" << result << std::endl;
			return result;
		} else if (align == 32) {
			return nBytes + (((nBytes & 0b11111) == 0) ? 0 : (32 - (nBytes & 0b11111)));
		} else if (align == 64) {
			return nBytes + (((nBytes & 0b111111) == 0) ? 0 : (64 - (nBytes & 0b111111)));
		} else {
			return nBytes;
		}
	}
}