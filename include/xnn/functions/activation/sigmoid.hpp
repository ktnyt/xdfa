#ifndef __XNN_FUNCTIONS_ACTIVATION_SIGMOID_HPP__
#define __XNN_FUNCTIONS_ACTIVATION_SIGMOID_HPP__

#include "xnn/function.hpp"

#include "xtensor/xmath.hpp"

namespace xnn {
namespace functions {
namespace activation {

class Sigmoid final : public Function<float> {
public:
  xt::xarray<float> forward(xt::xarray<float> x) override {
    return memory = xt::tanh(x * 0.5) * 0.5 + 0.5;
  }

  xt::xarray<float> backward(xt::xarray<float> d) override {
    return d * memory * (1.0 - memory);
  }

private:
  xt::xarray<float> memory;
};

} // namespace activation
} // namespace functions
} // namespace xnn

#endif // __XNN_FUNCTIONS_ACTIVATION_SIGMOID_HPP__
