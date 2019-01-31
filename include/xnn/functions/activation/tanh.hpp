#ifndef __XNN_FUNCTIONS_ACTIVATION_TANH_HPP__
#define __XNN_FUNCTIONS_ACTIVATION_TANH_HPP__

#include "xnn/function.hpp"

#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"

namespace xnn {
namespace functions {
namespace activation {

class Tanh final : public Function<float> {
 public:
  xt::xarray<float> operator()(xt::xarray<float> x) override {
    return xt::tanh(x);
  }
};

class TanhGrad : public Function<float> {
 public:
  xt::xarray<float> operator()(xt::xarray<float> x) override {
    return 1.0 - (x * x);
  }
};

xt::xarray<float> tanh(xt::xarray<float> x) { return Tanh()(x); }

xt::xarray<float> tanh_grad(xt::xarray<float> x) { return TanhGrad()(x); }

}  // namespace activation
}  // namespace functions
}  // namespace xnn

#endif  // __XNN_FUNCTIONS_ACTIVATION_TANH_HPP__
