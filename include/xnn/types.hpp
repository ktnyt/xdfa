#ifndef __XNN_TYPES_HPP__
#define __XNN_TYPES_HPP__

#include "xtensor/xarray.hpp"

namespace xnn {

using Expression = xt::xexpression<float>;
using Arrayf = xt::xarray<float>;
using Arrayi = xt::xarray<int>;
using Arrayu = xt::xarray<unsigned int>;

}  // namespace xnn

#endif  // __XNN_TYPES_HPP__
