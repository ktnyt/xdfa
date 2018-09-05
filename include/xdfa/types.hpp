#ifndef __XDFA_TYPES_HPP__
#define __XDFA_TYPES_HPP__

#include "xtensor/xarray.hpp"

namespace xdfa {

using Expression = xt::xexpression<float>;
using Arrayf = xt::xarray<float>;
using Arrayi = xt::xarray<int>;
using Arrayu = xt::xarray<unsigned int>;

}  // namespace xdfa

#endif  // __XDFA_TYPES_HPP__
