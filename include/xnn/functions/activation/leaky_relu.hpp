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

  xt::xarray<float> operator(xt::xarray<float> x) {
    xt::filtration(x, x < 0) *= slope;
  }

 private:
  float slope;
};

class LeakyReLUGrad : public Function<float> {
 public:
  LeakyReLU(float slope = 0.2) : slope(slope) {}

  xt::xarray<float> operator(xt::xarray<float> x) {
    xt::filtration(x, x > 0) = 1.0;
    xt::filtration(x, x <= 0) = slope;
    return x;
  }

 private:
  float slope;
};

xt::xarray<float> leaky_relu(xt::xarray<float> x, float slope = 0.2) {
  return LeakyReLU(slope)(x);
}

xt::xarray<float> leaky_relu_grad(xt::xarray<float> x, float slope = 0.2) {
  return LeakyReLUGrad(slope)(x);
}

}  // namespace activation
}  // namespace functions
}  // namespace xnn

#endif  // __XNN_FUNCTIONS_ACTIVATION_LEAKY_RELU_HPP__
