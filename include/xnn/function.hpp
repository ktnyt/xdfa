#ifndef __XNN_FUNCTION_HPP__
#define __XNN_FUNCTION_HPP__

#include "xtensor/xarray.hpp"

#include <functional>

namespace xnn {

template <class T>
using Function = std::function<xt::xarray<T>(xt::xarray<T>)>;

}  // namespace xnn

#endif  // __XNN_FUNCTION_HPP__
