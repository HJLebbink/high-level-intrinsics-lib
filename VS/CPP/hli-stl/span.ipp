#include "intrin.h"

#include <tuple>

namespace hli {

	template <class T>
	using span = std::tuple<typename T * const, const size_t>;
	
	template <class T>
	using data = struct {
		typename * const ptr;
		const size_t length;
	};


}
