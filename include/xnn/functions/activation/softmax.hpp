#ifndef __XNN_FUNCTIONS_ACTIVATION_SOFTMAX_HPP__
#define __XNN_FUNCTIONS_ACTIVATION_SOFTMAX_HPP__

#include "xnn/function.hpp"

#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"

namespace xnn {
namespace functions {
namespace activation {

class Softmax final : public Function<float> {
 public:
  xt::xarray<float> operator()(const xt::xarray<float>& x) override {
    xt::xarray<float> y = x;
    xt::transpose(y) -= xt::amax(y, {1});
    xt::xarray<float> e = xt::exp(y);
    xt::transpose(e) /= xt::sum(e, {1});
    return e;
  }
};

inline xt::xarray<float> softmax(const xt::xarray<float>& x) {
  return Softmax()(x);
}

}  // namespace activation
}  // namespace functions
}  // namespace xnn

#endif  // __XNN_FUNCTIONS_ACTIVATION_SOFTMAX_HPP__
