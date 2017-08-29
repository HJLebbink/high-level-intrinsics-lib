#pragma once

#include <iostream>		// std::cout

namespace hli {

	// swap the elements of data given pos1 and pos2
	template <class T>
	inline void swapElement(
		T * const data,
		const int pos1,
		const U16 pos2)
	{
		const T temp = data[pos1];
		//std::cout << "INFO: _mm_permute_array::swapElement: swapping pos1=" << pos1 << " with pos2=" << pos2 << "; data=" << temp << std::endl;
		data[pos1] = data[pos2];
		data[pos2] = temp;
	}

	template <class T>
	inline void swapArray(
		T * const data,
		const U16 * const swap_array,
		const size_t nElements)
	{
		for (int i = static_cast<int>(nElements) - 1; i > 0; --i) {
#		if	_DEBUG 
			if (i >= static_cast<int>(nElements)) {
				std::cout << "ERROR: hli::swapArray i=" << i << "; nElements=" << nElements << std::endl;
				return;
			}
			if (swap_array[i] >= nElements) {
				std::cout << "ERROR: hli::swapArray i=" << i << "; swap_array[i]=" << swap_array[i] << "; nElements = " << nElements << std::endl;
				return;
			}
#		endif
			swapElement(data, i, swap_array[i]);
		}
	}
}

