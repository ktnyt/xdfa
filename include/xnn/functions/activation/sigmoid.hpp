#ifndef __XNN_FUNCTIONS_ACTIVATION_SIGMOID_HPP__
#define __XNN_FUNCTIONS_ACTIVATION_SIGMOID_HPP__

#include "xnn/function.hpp"

#include "xtensor/xmath.hpp"

#include <queue>

namespace xnn {
namespace functions {
namespace activation {

class Sigmoid final : public Function<float> {
  class Impl final : public Function<float>::Impl {
  public:
    xt::xarray<float> forward(xt::xarray<float> x) override {
      queue.push(xt::tanh(x * 0.5) * 0.5 + 0.5);
      return queue.front();
    }

    xt::xarray<float> backward(xt::xarray<float> d) override {
      xt::xarray<float> y = queue.front();
      queue.pop();
      return d * y * (1.0 - y);
    }

  private:
    std::queue<xt::xarray<float>> queue;
  };

public:
  Sigmoid() : Function<float>(std::make_shared<Impl>()) {}
};

} // namespace activation
} // namespace functions
} // namespace xnn

#endif // __XNN_FUNCTIONS_ACTIVATION_SIGMOID_HPP__
