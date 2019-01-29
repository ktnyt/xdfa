#ifndef __XNN_LAYERS_NOISE_DROPOUT_HPP__
#define __XNN_LAYERS_NOISE_DROPOUT_HPP__

#include "xnn/functions/noise/dropout.hpp"
#include "xnn/layer.hpp"

#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"

namespace xnn {
namespace layers {
namespace noise {

class Dropout final : public Layer<float> {
  class Impl : public Layer<float>::Impl {
   public:
    Impl(float ratio) : ratio(ratio) {}

    xt::xarray<float> forward(xt::xarray<float> x) override {
      if (x.shape() != mask.shape()) {
        mask = xt::random::rand(x.shape(), 0.0, 1.0) < ratio;
      }
      return functions::noise::dropout(x, mask);
    }

    xt::xarray<float> backward(xt::xarray<float> dy) override {
      return functions::noise::dropout(dy, mask);
    }

   private:
    float ratio;
    xt::xarray<bool> mask;
  };

 public:
  Dropout(float ratio) : Layer<float>(std::make_shared<Impl>(ratio)) {}
};

}  // namespace noise
}  // namespace layers
}  // namespace xnn

#endif  // __XNN_LAYERS_NOISE_DROPOUT_HPP__
