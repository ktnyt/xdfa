#ifndef __XNN_LAYERS_ACTIVATION_LEAKY_RELU_HPP__
#define __XNN_LAYERS_ACTIVATION_LEAKY_RELU_HPP__

#include "xnn/functions/activation/leaky_relu.hpp"
#include "xnn/layer.hpp"

#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"

#include <queue>

namespace xnn {
namespace layers {
namespace activation {

class LeakyReLU final : public Layer<float> {
  class Impl final : public Layer<float>::Impl {
   public:
    Impl(float slope) : slope(slope) {}

    xt::xarray<float> forward(const xt::xarray<float>& x) override {
      queue.push(functions::activation::leaky_relu(x, slope));
      return queue.front();
    }

    xt::xarray<float> backward(const xt::xarray<float>& d) override {
      xt::xarray<float> y = queue.front();
      queue.pop();
      return d * functions::activation::leaky_relu_grad(y, slope);
    }

   private:
    float slope;
    std::queue<xt::xarray<float>> queue;
  };

 public:
  LeakyReLU(float slope = 0.2) : Layer<float>(std::make_shared<Impl>(slope)) {}
};

}  // namespace activation
}  // namespace layers
}  // namespace xnn

#endif  // __XNN_LAYERS_ACTIVATION_LEAKY_RELU_HPP__
