#ifndef __XNN_FUNCTIONS_POOLING_MAX_POOLING_HPP__
#define __XNN_FUNCTIONS_POOLING_MAX_POOLING_HPP__

#include "xnn/functions/pooling/pooling.hpp"
#include "xnn/utils/convolution.hpp"
#include "xnn/utils/xtensor.hpp"

#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xindex_view.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xstrided_view.hpp"

#include <limits>
#include <tuple>

namespace xnn {
namespace functions {
namespace pooling {

class MaxPooling2D final : public Pooling2D<float> {
  static constexpr float lowest = std::numeric_limits<float>::lowest();

 public:
  MaxPooling2D(
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
    xt::xarray<float> col =
        utils::im2col(x, kh, kw, sy, sx, ph, pw, lowest, cover_all);
    std::size_t n = col.shape()[0];
    std::size_t c = col.shape()[1];
    std::size_t kh = col.shape()[2];
    std::size_t kw = col.shape()[3];
    std::size_t out_h = col.shape()[4];
    std::size_t out_w = col.shape()[5];

    col.reshape({n, c, kh * kw, out_h, out_w});

    indices = xt::argmax(col, 2);
    return xt::amax(col, 2);
  }

  xt::xarray<std::size_t> get_indices() const { return indices; }

 private:
  xt::xarray<std::size_t> indices;
};

class MaxPooling2DGrad final : public Pooling2D<float> {
 public:
  MaxPooling2DGrad(
      xt::xarray<std::size_t>& indices,
      std::size_t kh,
      std::size_t kw,
      std::size_t sy,
      std::size_t sx,
      std::size_t ph,
      std::size_t pw,
      bool cover_all = true,
      bool return_indices = false)
      : Pooling2D<float>(kh, kw, sy, sx, ph, pw, cover_all, return_indices),
        indices(indices) {}

  xt::xarray<float> operator()(xt::xarray<float> x) override {
    std::size_t n = x.shape()[0];
    std::size_t c = x.shape()[1];
    std::size_t out_h = x.shape()[2];
    std::size_t out_w = x.shape()[3];

    xt::xarray<float> dcol =
        xt::zeros<float>({n * c * out_h * out_w * kh * kw});

    xt::xarray<std::size_t> flattened =
        xt::flatten(indices) + xt::arange(indices.size()) * (kh * kw);
    xt::index_view(dcol, flattened) =
        xt::ravel<xt::layout_type::column_major>(x);
    dcol.reshape({n, c, out_h, out_w, kh, kw});
    dcol = utils::swapaxes(dcol, 2, 4);
    dcol = utils::swapaxes(dcol, 3, 5);

    std::size_t h;
    std::size_t w;
    std::tie(h, w) = calc_out_size(x);

    return utils::col2im(dcol, sy, sx, ph, pw, h, w);
  }

 private:
  std::tuple<std::size_t, std::size_t> calc_out_size(
      const xt::xarray<float>& x) {
    std::size_t in_h = x.shape()[2];
    std::size_t in_w = x.shape()[3];
    std::size_t out_h = utils::get_deconv_outsize(in_h, kh, sy, ph);
    std::size_t out_w = utils::get_deconv_outsize(in_w, kw, sx, pw);
    return std::tuple<std::size_t, std::size_t>(out_h, out_w);
  }

  xt::xarray<std::size_t>& indices;
};

inline std::pair<xt::xarray<float>, xt::xarray<std::size_t>> max_pooling_2d(
    xt::xarray<float> x,
    std::size_t kh,
    std::size_t kw,
    std::size_t sy,
    std::size_t sx,
    std::size_t ph,
    std::size_t pw,
    bool cover_all = true,
    bool return_indices = false) {
  MaxPooling2D f(kh, kw, sy, sx, ph, pw, cover_all, return_indices);
  xt::xarray<float> y = f(x);
  xt::xarray<float> indices = f.get_indices();
  return {y, indices};
}

inline std::pair<xt::xarray<float>, xt::xarray<std::size_t>> max_pooling_2d(
    xt::xarray<float> x,
    std::size_t k,
    std::size_t s,
    std::size_t p,
    bool cover_all = true,
    bool return_indices = false) {
  return max_pooling_2d(x, k, k, s, s, p, p, cover_all, return_indices);
}

inline xt::xarray<float> max_pooling_2d_grad(
    xt::xarray<float> x,
    xt::xarray<std::size_t>& indices,
    std::size_t kh,
    std::size_t kw,
    std::size_t sy,
    std::size_t sx,
    std::size_t ph,
    std::size_t pw,
    bool cover_all = true,
    bool return_indices = false) {
  return MaxPooling2DGrad(
      indices, kh, kw, sy, sx, ph, pw, cover_all, return_indices)(x);
}

inline xt::xarray<float> max_pooling_2d_grad(
    xt::xarray<float> x,
    xt::xarray<std::size_t>& indices,
    std::size_t k,
    std::size_t s,
    std::size_t p,
    bool cover_all = true,
    bool return_indices = false) {
  return max_pooling_2d_grad(
      x, indices, k, k, s, s, p, p, cover_all, return_indices);
}

}  // namespace pooling
}  // namespace functions
}  // namespace xnn

#endif  // __XNN_FUNCTIONS_POOLING_MAX_POOLING_HPP__
