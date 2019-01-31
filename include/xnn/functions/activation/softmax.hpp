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
  xt::xarray<float> operator()(xt::xarray<float> x) override {
    xt::transpose(x) -= xt::amax(x, {1});
    xt::xarray<float> e = xt::exp(x);
    xt::transpose(e) /= xt::sum(e, {1});
    return e;
  }
};

xt::xarray<float> softmax(xt::xarray<float> x) { return Softmax()(x); }

}  // namespace activation
}  // namespace functions
}  // namespace xnn

#endif  // __XNN_FUNCTIONS_ACTIVATION_SOFTMAX_HPP__
