#ifndef __XNN_FUNCTIONS_ACTIVATION_SIGMOID_HPP__
#define __XNN_FUNCTIONS_ACTIVATION_SIGMOID_HPP__

#include "xnn/function.hpp"

#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"

namespace xnn {
namespace functions {
namespace activation {

class Sigmoid final : public Function<float> {
 public:
  xt::xarray<float> operator()(xt::xarray<float> x) override {
    return xt::tanh(x * 0.5) * 0.5 + 0.5;
  }
};

class SigmoidGrad : public Function<float> {
 public:
  xt::xarray<float> operator()(xt::xarray<float> x) override {
    return x * (1.0 - x);
  }
};

inline xt::xarray<float> sigmoid(xt::xarray<float> x) { return Sigmoid()(x); }

inline xt::xarray<float> sigmoid_grad(xt::xarray<float> x) {
  return SigmoidGrad()(x);
}

}  // namespace activation
}  // namespace functions
}  // namespace xnn

#endif  // __XNN_FUNCTIONS_ACTIVATION_SIGMOID_HPP__
