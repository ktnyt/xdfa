#ifndef __XNN_LAYERS_ACTIVATION_SIGMOID_HPP__
#define __XNN_LAYERS_ACTIVATION_SIGMOID_HPP__

#include "xnn/layer.hpp"
#include "xnn/functions/activation/sigmoid.hpp"

#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"

#include <queue>

namespace xnn {
namespace layers {
namespace activation {

class Sigmoid final : public Layer<float> {
  class Impl final : public Layer<float>::Impl {
   public:
    xt::xarray<float> forward(xt::xarray<float> x) override {
      queue.push(functions::activation::sigmoid(x));
      return queue.front();
    }

    xt::xarray<float> backward(xt::xarray<float> d) override {
      xt::xarray<float> y = queue.front();
      queue.pop();
      return d * functions::activation::sigmoid_grad(y);
    }

   private:
    std::queue<xt::xarray<float>> queue;
  };

 public:
  Sigmoid() : Layer<float>(std::make_shared<Impl>()) {}
};

}  // namespace activation
}  // namespace layers
}  // namespace xnn

#endif  // __XNN_LAYERS_ACTIVATION_SIGMOID_HPP__
