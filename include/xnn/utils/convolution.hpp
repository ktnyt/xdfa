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

/* TODO Nd-Convolution
std::vector<std::vector<std::size_t>>
cartesian(std::vector<std::vector<std::size_t>> &v) {
  auto product = [](std::size_t a, std::vector<std::size_t> b) {
    return a * b.size();
  };
  const std::size_t N = std::accumulate(v.begin(), v.end(), 1, std::product);
  std::vector<std::vector<std::size_t>> r(N);
  std::vector<std::size_t> u(v.size());
  for (std::size_t n = 0; n < N; ++n) {
    lldiv_t q{n, 0};
    for (std::size_t j = 0; j < v.size(); ++j) {
      std::size_t i = v.size() - j - 1;
      q = std::div(q.quot, v[i].size());
      u[i] = v[i][q.rem];
    }
    r[n] = u;
  }
  return r;
}

template <class T>
xt::xarray<T> im2col(xt::xarray<T> x, std::vector<std::size_t> kernel_size,
                     std::vector<std::size_t> stride,
                     std::vector<std::size_t> pad, bool cover_all,
                     T pad_value) {
  std::size_t ndim = kernel_size.size();

  auto padded_shape = x.shape();
  xt::strided_slice_vector slices({xt::all(), xt::all()});
  for (std::size_t i = 0; i < ndim; ++i) {
    padded_shape[i + 2] += pad[i] * 2 + (cover_all ? stride[i] - 1 : 0);
    slices.push_back(xt::range(pad[i], pad[i] + x.shape()[i + 2]));
  }
  xt::xarray<T> padded_x = xt::zeros<T>(padded_shape) + pad_value;
  xt::strided_view(padded_x, slices) = x;

  std::vector<std::size_t> out_dims;
  for (std::size_t i = 0; i < ndim; ++i) {
    out_dims.emplace_back(get_conv_out_dim(x.shape()[i + 2], kernel_size[i],
                                           stride[i], pad[i], cover_all));
  }

  std::vector<std::size_t> out_shape({x.shape()[0], x.shape()[1]});
  std::copy(kernel_size.begin(), kernel_size.end(),
            std::back_inserter(out_shape));
  std::copy(out_dims.begin(), out_dims.end(), std::back_inserter(out_shape));
  xt::xarray<T> out(out_shape);

  std::vector<std::vector<std::size_t>> indices = catesian(kernel_size);

  for (std::size_t i = 0; i < product.size(); ++i) {
    xt::strided_slice_vector col_slices({xt::all(), xt::all()});
    for (std::size_t j = 0; j < indices[i].size(); ++j) {
      col_slices.push_back(indices[i][j]);
    }
    for(std::size_t j = 0; j < x.shape().size() - 2; ++j) {
      col_slices.push_back(xt::all());
    }
  }
}
*/

template <class T>
xt::xarray<T> im2col(
    xt::xarray<T> img,
    std::size_t kh,
    std::size_t kw,
    std::size_t sy,
    std::size_t sx,
    std::size_t ph,
    std::size_t pw,
    T padding_value,
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
  xt::xarray<T> img(shape);

  for (std::size_t j = 0; j < kh; ++j) {
    std::size_t j_lim = j + sy * out_h;
    for (std::size_t i = 0; i < kw; ++i) {
      std::size_t i_lim = i + sx * out_w;
      xt::view(
          img,
          xt::all(),
          xt::all(),
          xt::range(j, j_lim, sy),
          xt::range(i, i_lim, sx)) = xt::view(col, xt::all(), xt::all(), j, i);
    }
  }

  return xt::view(
      img, xt::all(), xt::all(), xt::range(ph, h + ph), xt::range(pw, w + pw));
}

}  // namespace utils
}  // namespace xnn

#endif  // __XNN_UTILS_CONVOLUTION_HPP__
