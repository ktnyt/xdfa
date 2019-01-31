#ifndef __XNN_FUNCTIONS_POOLING_AVERAGE_POOLING_HPP__
#define __XNN_FUNCTIONS_POOLING_AVERAGE_POOLING_HPP__

#include "xnn/functions/pooling/pooling.hpp"
#include "xnn/utils/convolution.hpp"
#include "xnn/utils/xtensor.hpp"

#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xview.hpp"

#include <iostream>

namespace xnn {
namespace functions {
namespace pooling {

class AveragePooling2D final : public Pooling2D<float> {
 public:
  AveragePooling2D(
      std::size_t kh,
      std::size_t kw,
      std::size_t sy,
      std::size_t sx,
      std::size_t ph,
      std::size_t pw,
      bool cover_all = true,
      bool return_indices = false)
      : Pooling2D<float>(kh, kw, sy, sx, ph, pw, cover_all, return_indices) {}

  xt::xarray<float> operator()(xt::xarray<float> x) override {
    xt::xarray<float> col = utils::im2col(x, kh, kw, sy, sx, ph, pw);
    return xt::mean(col, {2, 3});
  }
};

class AveragePooling2DGrad final : public Pooling2D<float> {
 public:
  AveragePooling2DGrad(
      std::size_t h,
      std::size_t w,
      std::size_t kh,
      std::size_t kw,
      std::size_t sy,
      std::size_t sx,
      std::size_t ph,
      std::size_t pw,
      bool cover_all = true,
      bool return_indices = false)
      : Pooling2D<float>(kh, kw, sy, sx, ph, pw, cover_all, return_indices),
        h(h),
        w(w) {}

  xt::xarray<float> operator()(xt::xarray<float> x) override {
    xt::xarray<float> tmp =
        xt::view(x, xt::all(), xt::all(), xt::newaxis(), xt::newaxis());
    xt::xarray<float> gcol = utils::tile(tmp, {1, 1, kh, kw, 1, 1});
    std::cout << gcol << std::endl;
    xt::xarray<float> y = utils::col2im(gcol, sy, sx, ph, pw, h, w);
    y /= kh * kw;
    return y;
  }

 private:
  std::size_t h;
  std::size_t w;
};

xt::xarray<float> average_pooling_2d(
    xt::xarray<float> x,
    std::size_t kh,
    std::size_t kw,
    std::size_t sy,
    std::size_t sx,
    std::size_t ph,
    std::size_t pw,
    bool cover_all = true,
    bool return_indices = false) {
  return AveragePooling2D(kh, kw, sy, sx, ph, pw, cover_all, return_indices)(x);
}

xt::xarray<float> average_pooling_2d(
    xt::xarray<float> x,
    std::size_t k,
    std::size_t s,
    std::size_t p,
    bool cover_all = true,
    bool return_indices = false) {
  return average_pooling_2d(x, k, k, s, s, p, p, cover_all, return_indices);
}

xt::xarray<float> average_pooling_2d_grad(
    xt::xarray<float> x,
    std::size_t h,
    std::size_t w,
    std::size_t kh,
    std::size_t kw,
    std::size_t sy,
    std::size_t sx,
    std::size_t ph,
    std::size_t pw,
    bool cover_all = true,
    bool return_indices = false) {
  return AveragePooling2DGrad(
      h, w, kh, kw, sy, sx, ph, pw, cover_all, return_indices)(x);
}

xt::xarray<float> average_pooling_2d_grad(
    xt::xarray<float> x,
    std::size_t h,
    std::size_t w,
    std::size_t k,
    std::size_t s,
    std::size_t p,
    bool cover_all = true,
    bool return_indices = false) {
  return average_pooling_2d_grad(
      x, h, w, k, k, s, s, p, p, cover_all, return_indices);
}

}  // namespace pooling
}  // namespace functions
}  // namespace xnn

#endif  // __XNN_FUNCTIONS_POOLING_AVERAGE_POOLING_HPP__
