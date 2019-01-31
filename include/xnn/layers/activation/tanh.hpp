#ifndef __XNN_LAYERS_ACTIVATION_TANH_HPP__
#define __XNN_LAYERS_ACTIVATION_TANH_HPP__

#include "xnn/functions/activation/tanh.hpp"
#include "xnn/layer.hpp"

#include "xtensor/xarray.hpp"

#include <queue>

namespace xnn {
namespace layers {
namespace activation {

class Tanh final : public Layer<float> {
  class Impl final : public Layer<float>::Impl {
   public:
    xt::xarray<float> forward(xt::xarray<float> x) override {
      queue.push(functions::activation::tanh(x));
      return queue.front();
    }

    xt::xarray<float> backward(xt::xarray<float> d) override {
      xt::xarray<float> y = queue.front();
      queue.pop();
      return d * functions::activation::tanh_grad(y);
    }

   private:
    std::queue<xt::xarray<float>> queue;
  };

 public:
  Tanh() : Layer<float>(std::make_shared<Impl>()) {}
};

}  // namespace activation
}  // namespace layers
}  // namespace xnn

#endif  // __XNN_LAYERS_ACTIVATION_TANH_HPP__
