#ifndef __XNN_FUNCTIONS_ACTIVATION_SOFTMAX_HPP__
#define __XNN_FUNCTIONS_ACTIVATION_SOFTMAX_HPP__

#include "xnn/function.hpp"

#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"

#include <iostream>

namespace xnn {
namespace functions {
namespace activation {

xt::xarray<float> softmax(xt::xarray<float> x) {
  xt::transpose(x) -= xt::amax(x, {1});
  xt::xarray<float> e = xt::exp(x);
  xt::transpose(e) /= xt::sum(e, {1});
  return e;
}

}  // namespace activation
}  // namespace functions
}  // namespace xnn

#endif  // __XNN_FUNCTIONS_ACTIVATION_SOFTMAX_HPP__
