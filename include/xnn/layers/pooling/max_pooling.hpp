#ifndef __XNN_LAYERS_POOLING_MAX_POOLING_HPP__
#define __XNN_LAYERS_POOLING_MAX_POOLING_HPP__

#include "xnn/functions/pooling/max_pooling.hpp"
#include "xnn/layer.hpp"

#include "xtensor/xarray.hpp"

#include <queue>

namespace xnn {
namespace layers {
namespace pooling {

class MaxPooling2D final : public Layer<float> {
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
      xt::xarray<float> y;
      xt::xarray<std::size_t> i;
      std::tie(y, i) = functions::pooling::max_pooling_2d(
          x, kh, kw, sy, sx, ph, pw, cover_all);
      queue.push(i);
      return y;
    }

    xt::xarray<float> backward(const xt::xarray<float>& dy) override {
      xt::xarray<std::size_t> i = queue.front();
      queue.pop();
      return functions::pooling::max_pooling_2d_grad(
          dy, i, kh, kw, sy, sx, ph, pw, cover_all);
    }

   private:
    std::size_t kh;
    std::size_t kw;
    std::size_t sy;
    std::size_t sx;
    std::size_t ph;
    std::size_t pw;
    bool cover_all;

    std::queue<xt::xarray<std::size_t>> queue;
  };

 public:
  MaxPooling2D(
      std::size_t kh,
      std::size_t kw,
      std::size_t sy,
      std::size_t sx,
      std::size_t ph,
      std::size_t pw,
      bool cover_all = false)
      : Layer<float>(
            std::make_shared<Impl>(kh, kw, sy, sx, ph, pw, cover_all)) {}

  MaxPooling2D(
      std::size_t k, std::size_t s, std::size_t p, bool cover_all = false)
      : MaxPooling2D(k, k, s, s, p, p, cover_all) {}
};

}  // namespace pooling
}  // namespace layers
}  // namespace xnn

#endif  // __XNN_LAYERS_POOLING_MAX_POOLING_HPP__
