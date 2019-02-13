#ifndef __XNN_LAYERS_LOSS_MEAN_SQUARED_ERROR_HPP__
#define __XNN_LAYERS_LOSS_MEAN_SQUARED_ERROR_HPP__

#include "xnn/functions/loss/mean_squared_error.hpp"
#include "xnn/layer.hpp"

#include "xtensor/xarray.hpp"

namespace xnn {
namespace layers {
namespace loss {

class MeanSquaredError : public Layer<float> {
  class Impl final : public Layer<float>::Impl {
   public:
    void set_labels(xt::xarray<float>& v) { t = v; }

    xt::xarray<float> forward(const xt::xarray<float>& x) override {
      return functions::loss::mean_squared_error(x, t);
    }

    xt::xarray<float> backward(const xt::xarray<float>& x) override {
      return functions::loss::mean_squared_error_grad(x, t);
    }

   private:
    xt::xarray<float> t;
  };

 public:
  MeanSquaredError() : Layer<float>(std::make_shared<Impl>()) {}

  MeanSquaredError& with(xt::xarray<float>& t) {
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

#endif  // __XNN_LAYERS_LOSS_MEAN_SQUARED_ERROR_HPP__
