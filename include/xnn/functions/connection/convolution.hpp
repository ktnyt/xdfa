#ifndef __XNN_FUNCTIONS_CONNECTION_CONVOLUTION_HPP__
#define __XNN_FUNCTIONS_CONNECTION_CONVOLUTION_HPP__

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

template <class T>
auto tensordot(
    const xt::xarray<T>& a,
    const xt::xarray<T>& b,
    const std::vector<std::size_t>& axes_a,
    const std::vector<std::size_t>& axes_b) {
  std::vector<std::size_t> as(a.shape().begin(), a.shape().end());
  std::vector<std::size_t> bs(b.shape().begin(), b.shape().end());
  std::size_t nda = as.size();
  std::size_t ndb = bs.size();

  std::vector<std::size_t> notin_a;
  for (std::size_t k = 0; k < nda; ++k) {
    if (!contains(axes_a, k)) {
      notin_a.push_back(k);
    }
  }
  std::vector<std::size_t> newaxes_a(notin_a.begin(), notin_a.end());
  newaxes_a.insert(newaxes_a.end(), axes_a.begin(), axes_a.end());

  std::size_t N2_a = 1;
  for (auto axis : axes_a) {
    N2_a *= as[axis];
  }
  std::size_t N1_a = 1;
  for (auto axis : notin_a) {
    N1_a *= as[axis];
  }
  std::vector<std::size_t> newshape_a = {N1_a, N2_a};

  std::vector<std::size_t> notin_b;
  for (std::size_t k = 0; k < ndb; ++k) {
    if (!contains(axes_b, k)) {
      notin_b.push_back(k);
    }
  }
  std::vector<std::size_t> newaxes_b(notin_b.begin(), notin_b.end());
  newaxes_b.insert(newaxes_b.end(), axes_b.begin(), axes_b.end());

  std::size_t N2_b = 1;
  for (auto axis : axes_b) {
    N2_b *= bs[axis];
  }
  std::size_t N1_b = 1;
  for (auto axis : notin_b) {
    N1_b *= bs[axis];
  }
  std::vector<std::size_t> newshape_b = {N2_b, N1_b};

  std::vector<std::size_t> outshape;
  for (auto axis : notin_a) {
    outshape.push_back(as[axis]);
  }
  for (auto axis : notin_b) {
    outshape.push_back(bs[axis]);
  }

  xt::xarray<T> at = xt::transpose(a, newaxes_a);
  xt::xarray<T> bt = xt::transpose(b, newaxes_b);
  at.reshape(newshape_a);
  bt.reshape(newshape_b);
  xt::xarray<T> res = xt::linalg::dot(at, bt);
  res.reshape(outshape);
  return res;
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

  xt::xarray<float> operator()(xt::xarray<float> x) override {
    xt::xarray<float> tmp = tensordot(W, x, {0}, {1});
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

xt::xarray<float> convolution2d(
    xt::xarray<float> x,
    xt::xarray<float>& W,
    std::size_t sy,
    std::size_t sx,
    std::size_t ph,
    std::size_t pw,
    bool cover_all = false) {
  return Convolution2D(W, sy, sx, ph, pw, cover_all)(x);
}

xt::xarray<float> deconvolution2d(
    xt::xarray<float> x,
    xt::xarray<float>& W,
    std::size_t sy,
    std::size_t sx,
    std::size_t ph,
    std::size_t pw,
    bool cover_all = false) {
  return Deconvolution2D(W, sy, sx, ph, pw, cover_all)(x);
}

}  // namespace connection
}  // namespace functions
}  // namespace xnn

#endif  // __XNN_FUNCTIONS_CONNECTION_CONVOLUTION_HPP__
