#ifndef __XNN_FUNCTIONS_CONNECTION_CONVOLUTION_HPP__
#define __XNN_FUNCTIONS_CONNECTION_CONVOLUTION_HPP__

#include "xnn/function.hpp"
#include "xnn/utils/convolution.hpp"

#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xmanipulation.hpp"

#include <algorithm>
#include <tuple>
#include <vector>

namespace xnn {
namespace functions {
namespace connection {

template <class C, class T>
bool contains(C&& container, const T& value) {
  return container.end() !=
         std::find(container.begin(), container.end(), value);
}

class Convolution2D final : public Function<float> {
 public:
  Convolution2D(
      xt::xarray<float>& W,
      std::size_t sy,
      std::size_t sx,
      std::size_t ph,
      std::size_t pw,
      bool cover_all = false)
      : W(W), sy(sy), sx(sx), ph(ph), pw(pw), cover_all(cover_all) {}

  Convolution2D(
      xt::xarray<float>& W,
      std::size_t s,
      std::size_t p,
      bool cover_all = false)
      : W(W), sy(s), sx(s), ph(p), pw(p), cover_all(cover_all) {}

  xt::xarray<float> operator()(xt::xarray<float> x) override {
    xt::xarray<float> col = utils::im2col(
        x,
        W.shape()[2],
        W.shape()[3],
        sy,
        sx,
        ph,
        pw,
        static_cast<float>(0),
        cover_all);
    xt::xarray<float> y = xt::linalg::tensordot(col, W, {1, 2, 3}, {1, 2, 3});
    return xt::transpose(y, {0, 3, 1, 2});
  }

 private:
  xt::xarray<float>& W;
  std::size_t sy;
  std::size_t sx;
  std::size_t ph;
  std::size_t pw;
  bool cover_all;
};

class Convolution2DGrad final : public Function<float> {
 public:
  Convolution2DGrad(
      xt::xarray<float>& W,
      xt::xarray<float>& dy,
      std::size_t sy,
      std::size_t sx,
      std::size_t ph,
      std::size_t pw,
      bool cover_all = false)
      : W(W), dy(dy), sy(sy), sx(sx), ph(ph), pw(pw), cover_all(cover_all) {}

  Convolution2DGrad(
      xt::xarray<float>& W,
      xt::xarray<float>& dy,
      std::size_t s,
      std::size_t p,
      bool cover_all = false)
      : W(W), dy(dy), sy(s), sx(s), ph(p), pw(p), cover_all(cover_all) {}

  xt::xarray<float> operator()(xt::xarray<float> x) override {
    xt::xarray<float> col = utils::im2col(
        x,
        W.shape()[2],
        W.shape()[3],
        sy,
        sx,
        ph,
        pw,
        static_cast<float>(0),
        cover_all);
    return xt::linalg::tensordot(dy, col, {0, 2, 3}, {0, 4, 5});
  }

 private:
  xt::xarray<float>& W;
  xt::xarray<float>& dy;
  std::size_t sy;
  std::size_t sx;
  std::size_t ph;
  std::size_t pw;
  bool cover_all;
};

class Deconvolution2D final : public Function<float> {
 public:
  Deconvolution2D(
      xt::xarray<float>& W,
      std::size_t sy,
      std::size_t sx,
      std::size_t ph,
      std::size_t pw,
      bool cover_all = false)
      : W(W), sy(sy), sx(sx), ph(ph), pw(pw), cover_all(cover_all) {}

  Deconvolution2D(
      xt::xarray<float>& W,
      std::size_t s,
      std::size_t p,
      bool cover_all = false)
      : W(W), sy(s), sx(s), ph(p), pw(p), cover_all(cover_all) {}

  xt::xarray<float> operator()(xt::xarray<float> x) override {
    xt::xarray<float> tmp = xt::linalg::tensordot(W, x, {0}, {1});
    xt::xarray<float> col = xt::transpose(tmp, {3, 0, 1, 2, 4, 5});
    std::size_t h;
    std::size_t w;
    std::tie(h, w) = calc_out_size(x);
    xt::xarray<float> y = utils::col2im(col, sy, sx, ph, pw, h, w);
    return y;
  }

 private:
  std::tuple<std::size_t, std::size_t> calc_out_size(
      const xt::xarray<float>& x) {
    std::size_t kh = W.shape()[2];
    std::size_t kw = W.shape()[3];
    std::size_t in_h = x.shape()[2];
    std::size_t in_w = x.shape()[3];
    std::size_t out_h = utils::get_deconv_outsize(in_h, kh, sy, ph);
    std::size_t out_w = utils::get_deconv_outsize(in_w, kw, sx, pw);
    return std::tuple<std::size_t, std::size_t>(out_h, out_w);
  }

  xt::xarray<float>& W;
  std::size_t sy;
  std::size_t sx;
  std::size_t ph;
  std::size_t pw;
  bool cover_all;
};

xt::xarray<float> convolution_2d(
    xt::xarray<float> x,
    xt::xarray<float>& W,
    std::size_t sy,
    std::size_t sx,
    std::size_t ph,
    std::size_t pw,
    bool cover_all = false) {
  return Convolution2D(W, sy, sx, ph, pw, cover_all)(x);
}

xt::xarray<float> convolution_2d(
    xt::xarray<float> x,
    xt::xarray<float>& W,
    std::size_t s,
    std::size_t p,
    bool cover_all = false) {
  return Convolution2D(W, s, p, cover_all)(x);
}

xt::xarray<float> convolution_2d_grad(
    xt::xarray<float> x,
    xt::xarray<float>& W,
    xt::xarray<float>& dy,
    std::size_t sy,
    std::size_t sx,
    std::size_t ph,
    std::size_t pw,
    bool cover_all = false) {
  return Convolution2DGrad(W, dy, sy, sx, ph, pw, cover_all)(x);
}

xt::xarray<float> convolution_2d_grad(
    xt::xarray<float> x,
    xt::xarray<float>& W,
    xt::xarray<float>& dy,
    std::size_t s,
    std::size_t p,
    bool cover_all = false) {
  return Convolution2DGrad(W, dy, s, p, cover_all)(x);
}

xt::xarray<float> deconvolution_2d(
    xt::xarray<float> x,
    xt::xarray<float>& W,
    std::size_t sy,
    std::size_t sx,
    std::size_t ph,
    std::size_t pw,
    bool cover_all = false) {
  return Deconvolution2D(W, sy, sx, ph, pw, cover_all)(x);
}

xt::xarray<float> deconvolution_2d(
    xt::xarray<float> x,
    xt::xarray<float>& W,
    std::size_t s,
    std::size_t p,
    bool cover_all = false) {
  return Deconvolution2D(W, s, s, p, p, cover_all)(x);
}

}  // namespace connection
}  // namespace functions
}  // namespace xnn

#endif  // __XNN_FUNCTIONS_CONNECTION_CONVOLUTION_HPP__
