#ifndef __XNN_LAYERS_ACTIVATION_RELU_HPP__
#define __XNN_LAYERS_ACTIVATION_RELU_HPP__

#include "xnn/layer.hpp"
#include "xnn/functions/activation/relu.hpp"

#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"

#include <queue>

namespace xnn {
namespace layers {
namespace activation {

class ReLU final : public Layer<float> {
  class Impl final : public Layer<float>::Impl {
   public:
    xt::xarray<float> forward(xt::xarray<float> x) override {
      queue.push(functions::activation::relu(x));
      return queue.front();
    }

    xt::xarray<float> backward(xt::xarray<float> d) override {
      xt::xarray<float> y = queue.front();
      queue.pop();
      return d * functions::activation::relu_grad(y);
    }

   private:
    std::queue<xt::xarray<float>> queue;
  };

 public:
  ReLU() : Layer<float>(std::make_shared<Impl>()) {}
};

}  // namespace activation
}  // namespace layers
}  // namespace xnn

#endif  // __XNN_LAYERS_ACTIVATION_RELU_HPP__
