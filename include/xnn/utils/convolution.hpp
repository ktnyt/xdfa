#ifndef __XNN_UTILS_CONVOLUTION_HPP__
#define __XNN_UTILS_CONVOLUTION_HPP__

#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xstrided_view.hpp"

#include "xtensor-blas/xlinalg.hpp"

#include <algorithm>
#include <functional>
#include <vector>

namespace xnn {
namespace utils {

std::size_t get_conv_outsize(
    std::size_t size,
    std::size_t k,
    std::size_t s,
    std::size_t p,
    bool cover_all = false) {
  if (cover_all) {
    return (size + p * 2 - k + s - 1) / s + 1;
  }
  return (size + p * 2 - k) / s + 1;
}

std::size_t get_deconv_outsize(
    std::size_t size,
    std::size_t k,
    std::size_t s,
    std::size_t p,
    bool cover_all = false) {
  if (cover_all) {
    return s * (size - 1) + k - s + 1 - 2 * p;
  }
  return s * (size - 1) + k - 2 * p;
}

template <class T>
xt::xarray<T> im2col(
    xt::xarray<T> img,
    std::size_t kh,
    std::size_t kw,
    std::size_t sy,
    std::size_t sx,
    std::size_t ph,
    std::size_t pw,
    T padding_value = static_cast<T>(0),
    bool cover_all = false) {
  auto padded_shape = img.shape();
  std::size_t n = padded_shape[0];
  std::size_t c = padded_shape[1];
  std::size_t h = padded_shape[2];
  std::size_t w = padded_shape[3];

  padded_shape[2] += ph * 2 + (cover_all ? sy - 1 : 0);
  padded_shape[3] += pw * 2 + (cover_all ? sx - 1 : 0);

  xt::xarray<T> tmp = xt::zeros<T>(padded_shape) + padding_value;
  xt::view(
      tmp, xt::all(), xt::all(), xt::range(ph, ph + h), xt::range(pw, pw + w)) =
      img;

  std::size_t out_h = get_conv_outsize(h, kh, sy, ph, cover_all);
  std::size_t out_w = get_conv_outsize(w, kw, sx, pw, cover_all);

  std::vector<std::size_t> out_shape = {n, c, kh, kw, out_h, out_w};

  xt::xarray<T> col(out_shape);

  for (std::size_t j = 0; j < kh; ++j) {
    std::size_t j_lim = j + sy * out_h;
    for (std::size_t i = 0; i < kw; ++i) {
      std::size_t i_lim = i + sx * out_w;
      xt::view(col, xt::all(), xt::all(), j, i, xt::all(), xt::all()) =
          xt::view(
              tmp,
              xt::all(),
              xt::all(),
              xt::range(j, j_lim, sy),
              xt::range(i, i_lim, sx));
    }
  }

  return col;
}

template <class T>
xt::xarray<T> col2im(
    xt::xarray<T> col,
    std::size_t sy,
    std::size_t sx,
    std::size_t ph,
    std::size_t pw,
    std::size_t h,
    std::size_t w) {
  std::size_t n = col.shape()[0];
  std::size_t c = col.shape()[1];
  std::size_t kh = col.shape()[2];
  std::size_t kw = col.shape()[3];
  std::size_t out_h = col.shape()[4];
  std::size_t out_w = col.shape()[5];

  std::vector<std::size_t> shape = {
      n, c, h + 2 * ph + sy - 1, w + 2 * pw + sx - 1};
  xt::xarray<T> img = xt::zeros<T>(shape);

  for (std::size_t j = 0; j < kh; ++j) {
    std::size_t j_lim = j + sy * out_h;
    for (std::size_t i = 0; i < kw; ++i) {
      std::size_t i_lim = i + sx * out_w;
      xt::view(
          img,
          xt::all(),
          xt::all(),
          xt::range(j, j_lim, sy),
          xt::range(i, i_lim, sx)) +=
          xt::view(col, xt::all(), xt::all(), j, i, xt::all(), xt::all());
    }
  }

  xt::xarray<float> ret = xt::view(
      img, xt::all(), xt::all(), xt::range(ph, h + ph), xt::range(pw, w + pw));
  return ret;
}

}  // namespace utils
}  // namespace xnn

#endif  // __XNN_UTILS_CONVOLUTION_HPP__
