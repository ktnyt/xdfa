#ifndef __XNN_LAYERS_LOSS_SOFTMAX_CROSS_ENTROPY_HPP__
#define __XNN_LAYERS_LOSS_SOFTMAX_CROSS_ENTROPY_HPP__

#include "xnn/functions/activation/softmax.hpp"
#include "xnn/functions/loss/cross_entropy.hpp"
#include "xnn/layer.hpp"
#include "xnn/utils/helpers.hpp"

#include "xtensor/xarray.hpp"
#include "xtensor/xindex_view.hpp"

#include <memory>
#include <vector>

namespace xnn {
namespace layers {
namespace loss {

class SoftmaxCrossEntropy : public Layer<float> {
  class Impl final : public Layer<float>::Impl {
   public:
    void set_labels(xt::xarray<int> t) { labels = t; }

    xt::xarray<float> forward(xt::xarray<float> x) override {
      memory = functions::activation::softmax(x);
      return functions::loss::cross_entropy(memory, labels);
    }

    xt::xarray<float> backward(xt::xarray<float> x) override {
      auto idx = utils::to_index(labels);
      xt::xarray<float> p = memory;
      xt::index_view(p, idx) -= 1;
      return p / labels.shape()[0];
    }

   private:
    xt::xarray<int> labels;
    xt::xarray<float> memory;
  };

 public:
  SoftmaxCrossEntropy() : Layer<float>(std::make_shared<Impl>()) {}

  SoftmaxCrossEntropy& with(xt::xarray<int> t) {
    auto impl = std::dynamic_pointer_cast<Impl>(ptr);
    impl->set_labels(t);
    return *this;
  }

  xt::xarray<float> grads() {
    auto impl = std::dynamic_pointer_cast<Impl>(ptr);
    return impl->backward(0.0);
  }
};

}  // namespace loss
}  // namespace layers
}  // namespace xnn

#endif  // __XNN_LAYERS_LOSS_SOFTMAX_CROSS_ENTROPY_HPP__
