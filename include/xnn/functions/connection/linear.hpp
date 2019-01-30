#ifndef __XNN_FUNCTIONS_CONNECTION_LINEAR_HPP__
#define __XNN_FUNCTIONS_CONNECTION_LINEAR_HPP__

#include "xnn/function.hpp"

#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xarray.hpp"

#include <queue>

namespace xnn {
namespace functions {
namespace connection {

class Linear final : public Function<float> {
 public:
  Linear(xt::xarray<float>& W, xt::xarray<float>& b) : W(W), b(b) {}

  xt::xarray<float> operator()(xt::xarray<float> x) override {
    return xt::linalg::dot(x, W) + b;
  }

 private:
  xt::xarray<float>& W;
  xt::xarray<float>& b;
};

class LinearGrad final : public Function<float> {
 public:
  LinearGrad(xt::xarray<float>& W) : W(W) {}

  xt::xarray<float> operator()(xt::xarray<float> x) override {
    return xt::linalg::dot(x, xt::transpose(W));
  }

 private:
  xt::xarray<float>& W;
};

xt::xarray<float> linear(
    xt::xarray<float> x, xt::xarray<float>& W, xt::xarray<float>& b) {
  return Linear(W, b)(x);
}

xt::xarray<float> linear_grad(xt::xarray<float> x, xt::xarray<float>& W) {
  return LinearGrad(W)(x);
}

}  // namespace connection
}  // namespace functions
}  // namespace xnn

#endif  // __XNN_FUNCTIONS_CONNECTION_LINEAR_HPP__
