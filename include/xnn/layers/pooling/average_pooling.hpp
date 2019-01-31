#ifndef __XNN_LAYERS_POOLING_AVERAGE_POOLING_HPP__
#define __XNN_LAYERS_POOLING_AVERAGE_POOLING_HPP__

#include "xnn/functions/pooling/average_pooling.hpp"
#include "xnn/layer.hpp"

#include "xtensor/xarray.hpp"

#include <pair>
#include <queue>

namespace xnn {
namespace layers {
namespace pooling {

class AveragePooling2D : public Layer<float> {
  class Impl : Layer<float>::Impl {
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

    xt::xarray<float> forward(xt::xarray<float> x) override {
      queue.emplace(x.shape()[2], x.shape()[3]);
      return functions::pooling::average_pooling_2d(
          x, kh, kw, sy, sx, ph, pw, cover_all);
    }

    xt::xarray<float> backward(xt::xarray<float> dy) override {
      std::size_t h;
      std::size_t w;
      std::tie(h, w) = queue.front();
      queue.pop();
      return functions::pooling::average_pooling_2d_grad(
          x, h, w, kh, kw, sy, sx, ph, pw, cover_all);
    }

   private:
    std::size_t kh;
    std::size_t kw;
    std::size_t sy;
    std::size_t sx;
    std::size_t ph;
    std::size_t pw;
    bool cover_all;

    std::queue<std::pair<std::size_t, std::size_t>> queue;
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
      : Impl(std::make_shared<Impl>(kh, kw, sy, sx, ph, pw, cover_all)) {}
};

}  // namespace pooling
}  // namespace layers
}  // namespace xnn

#endif  // __XNN_LAYERS_POOLING_AVERAGE_POOLING_HPP__
