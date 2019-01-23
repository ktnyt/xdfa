#ifndef __XNN_FUNCTIONS_LOSS_SOFTMAX_CROSS_ENTROPY_HPP__
#define __XNN_FUNCTIONS_LOSS_SOFTMAX_CROSS_ENTROPY_HPP__

#include "xnn/function.hpp"

#include "xtensor/xindex_view.hpp"

#include <memory>
#include <vector>

namespace xnn {
namespace functions {
namespace loss {

namespace internal {

xt::xarray<float> softmax(xt::xarray<float> x) {
  xt::transpose(x) -= xt::amax(x, {1});
  xt::xarray<float> e = xt::exp(x);
  xt::transpose(e) /= xt::sum(e, {1});
  return e;
}

std::vector<std::vector<int>> indices(xt::xarray<int> t) {
  std::vector<std::vector<int>> ret;
  for (int i = 0; i < t.size(); ++i) {
    ret.push_back({i, t[i]});
  }
  return ret;
}

} // namespace internal

class SoftmaxCrossEntropy : public Function<float> {
  class Impl final : public Function<float>::Impl {
  public:
    void set_labels(xt::xarray<int> t) { labels = t; }

    xt::xarray<float> forward(xt::xarray<float> x) override {
      using namespace internal;
      memory = softmax(x);
      auto idx = indices(labels);
      auto y = xt::index_view(memory, idx);
      return xt::mean(-xt::log(y));
    }

    xt::xarray<float> backward(xt::xarray<float> x) override {
      using namespace internal;
      auto idx = indices(labels);
      auto p = memory;
      xt::index_view(p, idx) -= 1;
      return p / labels.shape()[0];
    }

  private:
    xt::xarray<int> labels;
    xt::xarray<float> memory;
  };

public:
  SoftmaxCrossEntropy() : Function<float>(std::make_shared<Impl>()) {}

  SoftmaxCrossEntropy &with(xt::xarray<int> t) {
    auto impl = std::dynamic_pointer_cast<Impl>(ptr);
    impl->set_labels(t);
    return *this;
  }

  xt::xarray<float> grads() {
    auto impl = std::dynamic_pointer_cast<Impl>(ptr);
    return impl->backward(0.0);
  }
};

} // namespace loss
} // namespace functions
} // namespace xnn

#endif // __XNN_FUNCTIONS_LOSS_SOFTMAX_CROSS_ENTROPY_HPP__