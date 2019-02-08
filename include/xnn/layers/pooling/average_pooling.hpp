#ifndef __XNN_LAYERS_POOLING_AVERAGE_POOLING_HPP__
#define __XNN_LAYERS_POOLING_AVERAGE_POOLING_HPP__

#include "xnn/functions/pooling/average_pooling.hpp"
#include "xnn/layer.hpp"

#include "xtensor/xarray.hpp"

#include <queue>
#include <tuple>

namespace xnn {
namespace layers {
namespace pooling {

class AveragePooling2D : public Layer<float> {
  class Impl final : public Layer<float>::Impl {
   public:
    Impl(
        std::size_t kh,
        std::size_t kw,
        std::size_t sy,
        std::size_t sx,
        std::size_t ph,
        std::size_t pw,
        bool cover_all)
        : kh(kh),
          kw(kw),
          sy(sy),
          sx(sx),
          ph(ph),
          pw(pw),
          cover_all(cover_all) {}

    xt::xarray<float> forward(const xt::xarray<float>& x) override {
      return functions::pooling::average_pooling_2d(
          x, kh, kw, sy, sx, ph, pw, cover_all);
    }

    xt::xarray<float> backward(const xt::xarray<float>& dy) override {
      return functions::pooling::average_pooling_2d_grad(
          dy, kh, kw, sy, sx, ph, pw, cover_all);
    }

   private:
    std::size_t kh;
    std::size_t kw;
    std::size_t sy;
    std::size_t sx;
    std::size_t ph;
    std::size_t pw;
    bool cover_all;
  };

 public:
  AveragePooling2D(
      std::size_t kh,
      std::size_t kw,
      std::size_t sy,
      std::size_t sx,
      std::size_t ph,
      std::size_t pw,
      bool cover_all = false)
      : Layer<float>(
            std::make_shared<Impl>(kh, kw, sy, sx, ph, pw, cover_all)) {}

  AveragePooling2D(
      std::size_t k, std::size_t s, std::size_t p, bool cover_all = false)
      : AveragePooling2D(k, k, s, s, p, p, cover_all) {}
};

}  // namespace pooling
}  // namespace layers
}  // namespace xnn

#endif  // __XNN_LAYERS_POOLING_AVERAGE_POOLING_HPP__
