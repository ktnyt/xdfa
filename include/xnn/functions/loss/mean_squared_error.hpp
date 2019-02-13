#ifndef __XNN_FUNCTIONS_LOSS_MEAN_SQUARED_ERROR_HPP__
#define __XNN_FUNCTIONS_LOSS_MEAN_SQUARED_ERROR_HPP__

#include "xnn/function.hpp"

#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xarray.hpp"

namespace xnn {
namespace functions {
namespace loss {

class MeanSquaredError final : public Function<float> {
 public:
  MeanSquaredError(const xt::xarray<float>& t) : t(t) {}

  xt::xarray<float> operator()(const xt::xarray<float>& x) override {
    xt::xarray<float> d = x - t;
    return xt::linalg::dot(d, d) / static_cast<float>(d.size());
  }

 private:
  const xt::xarray<float>& t;
};

class MeanSquaredErrorGrad final : public Function<float> {
 public:
  MeanSquaredErrorGrad(const xt::xarray<float>& t) : t(t) {}

  xt::xarray<float> operator()(const xt::xarray<float>& x) override {
    return x - t;
  }

 private:
  const xt::xarray<float>& t;
};

xt::xarray<float> mean_squared_error(
    const xt::xarray<float>& x, const xt::xarray<float>& t) {
  return MeanSquaredError(t)(x);
}

xt::xarray<float> mean_squared_error_grad(
    const xt::xarray<float>& x, const xt::xarray<float>& t) {
  return MeanSquaredErrorGrad(t)(x);
}

}  // namespace loss
}  // namespace functions
}  // namespace xnn

#endif  // __XNN_FUNCTIONS_LOSS_MEAN_SQUARED_ERROR_HPP__
