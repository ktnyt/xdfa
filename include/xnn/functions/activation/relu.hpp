#ifndef __XNN_FUNCTIONS_ACTIVATION_RELU_HPP__
#define __XNN_FUNCTIONS_ACTIVATION_RELU_HPP__

#include "xnn/function.hpp"

#include "xtensor/xarray.hpp"
#include "xtensor/xindex_view.hpp"
#include "xtensor/xmath.hpp"

namespace xnn {
namespace functions {
namespace activation {

class ReLU : public Function<float> {
 public:
  xt::xarray<float> operator()(xt::xarray<float> x) override {
    return xt::fmax(x, 0.0);
  }
};

class ReLUGrad : public Function<float> {
 public:
  xt::xarray<float> operator()(xt::xarray<float> x) override {
    xt::filtration(x, x > 0) = 1.0;
    return x;
  }
};

inline xt::xarray<float> relu(xt::xarray<float> x) { return ReLU()(x); }

inline xt::xarray<float> relu_grad(xt::xarray<float> x) {
  return ReLUGrad()(x);
}

}  // namespace activation
}  // namespace functions
}  // namespace xnn

#endif  // __XNN_FUNCTIONS_ACTIVATION_RELU_HPP__
