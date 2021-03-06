#ifndef __XNN_FUNCTIONS_ACTIVATION_LEAKY_RELU_HPP__
#define __XNN_FUNCTIONS_ACTIVATION_LEAKY_RELU_HPP__

#include "xnn/function.hpp"

#include "xtensor/xarray.hpp"
#include "xtensor/xindex_view.hpp"
#include "xtensor/xmath.hpp"

namespace xnn {
namespace functions {
namespace activation {

class LeakyReLU : public Function<float> {
 public:
  LeakyReLU(float slope = 0.2) : slope(slope) {}

  xt::xarray<float> operator()(const xt::xarray<float>& x) override {
    xt::xarray<float> y = x;
    xt::filtration(y, y < 0) *= slope;
    return y;
  }

 private:
  float slope;
};

class LeakyReLUGrad : public Function<float> {
 public:
  LeakyReLUGrad(float slope = 0.2) : slope(slope) {}

  xt::xarray<float> operator()(const xt::xarray<float>& x) override {
    xt::xarray<float> y = x;
    xt::filtration(y, y > 0) = 1.0f;
    xt::filtration(y, y <= 0) = slope;
    return y;
  }

 private:
  float slope;
};

inline xt::xarray<float> leaky_relu(
    const xt::xarray<float>& x, float slope = 0.2) {
  return LeakyReLU(slope)(x);
}

inline xt::xarray<float> leaky_relu_grad(
    const xt::xarray<float>& x, float slope = 0.2) {
  return LeakyReLUGrad(slope)(x);
}

}  // namespace activation
}  // namespace functions
}  // namespace xnn

#endif  // __XNN_FUNCTIONS_ACTIVATION_LEAKY_RELU_HPP__
