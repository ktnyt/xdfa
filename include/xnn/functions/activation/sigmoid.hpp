#ifndef __XNN_FUNCTIONS_ACTIVATION_SIGMOID_HPP__
#define __XNN_FUNCTIONS_ACTIVATION_SIGMOID_HPP__

#include "xnn/function.hpp"

namespace xnn {
namespace functions {
namespace activation {

class Sigmoid final : public Function<float> {
 public:
  xt::xarray<float> operator()(xt::xarray<float> x) {
    return xt::tanh(x * 0.5) * 0.5 + 0.5;
  }
};

class SigmoidGrad : public Function<float> {
 public:
  xt::xarray<float> operator()(xt::xarray<float> x) {
    return x * (1.0 - x);
  }
};

xt::xarray<float> sigmoid(xt::xarray<float> x) { return Sigmoid()(x); }

xt::xarray<float> sigmoid_grad(xt::xarray<float> x) { return SigmoidGrad()(x); }

}  // namespace activation
}  // namespace functions
}  // namespace xnn

#endif  // __XNN_FUNCTIONS_ACTIVATION_SIGMOID_HPP__
