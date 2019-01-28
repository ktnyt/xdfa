#ifndef __XNN_FUNCTIONS_CONNECTION_CONVOLUTION_HPP__
#define __XNN_FUNCTIONS_CONNECTION_CONVOLUTION_HPP__

#include "xnn/function.hpp"
#include "xnn/initializers.hpp"
#include "xnn/utils/convolution.hpp"

#include "xtensor-blas/xlinalg.h"

#include <algorithm>
#include <numeric>

namespace xnn {
namespace functions {
namespace connection {

class Convolution2D final : public Function<float> {
  class Impl : Function<float>::Impl {
  public:
    Impl(std::size_t in_channels, std::size_t out_channels, std::size_t kh,
         std::size_t kw, std::size_t sy, std::size_t sx, std::size_t ph,
         std::size_t pw, bool cover_all = false)
        : W(initializers::LeCunNormal()({out_channels, in_channels, kh, kw})),
          kh(kh), kw(kw), sy(sy), sx(sx), ph(ph), pw(pw), cover_all(cover_all) {
    }

    xt::xarray<float> forward(xt::xarray<float> x) {
      xt::xarray<float> col =
          utils::im2col(x, kh, kw, sy, sx, ph, pw, 0, cover_all);
      xt::xarray<float> y = xt::linalg::tensordot(col, W, {1, 2, 3}, {1, 2, 4});
      return xt::transpose(y, {0, 3, 1, 2});
    }

  private:
    xt::xarray<float> W;

    std::size_t kh;
    std::size_t kw;
    std::size_t sy;
    std::size_t sx;
    std::size_t ph;
    std::size_t pw;
    bool cover_all;
  };

public:
  Impl(std::size_t kernel, std::size_t stride, std::size_t pad,
       bool cover_all = false)
      : sy(stride), sx(stride), ph(pad), pw(pad), cover_all(cover_all) {}
};

} // namespace connection
} // namespace functions
} // namespace xnn

#endif // __XNN_FUNCTIONS_CONNECTION_CONVOLUTION_HPP__
