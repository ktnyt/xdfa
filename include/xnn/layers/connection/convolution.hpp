#ifndef __XNN_LAYERS_CONNECTION_CONVOLUTION_HPP__
#define __XNN_LAYERS_CONNECTION_CONVOLUTION_HPP__

#include "xnn/functions/connection/convolution.hpp"
#include "xnn/initializers.hpp"
#include "xnn/layer.hpp"

namespace xnn {
namespace layers {
namespace connection {

class Convolution2D final : public Layer<float> {
  class Impl final : public Layer<float>::Impl {
   public:
    Impl(
        std::size_t in_channels,
        std::size_t out_channels,
        std::size_t kh,
        std::size_t kw,
        std::size_t sy,
        std::size_t sx,
        std::size_t ph,
        std::size_t pw,
        bool cover_all = false)
        : W(initializers::LeCunNormal()({out_channels, in_channels, kh, kw})),
          sy(sy),
          sx(sx),
          ph(ph),
          pw(pw),
          cover_all(cover_all) {}

    xt::xarray<float> forward(xt::xarray<float> x) override {
      return functions::connection::convolution2d(
          x, W, sy, sx, ph, pw, cover_all);
    }

    xt::xarray<float> backward(xt::xarray<float> dy) override {
      return functions::connection::deconvolution2d(
          dy, W, sy, sx, ph, pw, cover_all);
    }

   private:
    xt::xarray<float> W;
    std::size_t sy;
    std::size_t sx;
    std::size_t ph;
    std::size_t pw;
    bool cover_all;
  };

 public:
  Convolution2D(
      std::size_t in_channels,
      std::size_t out_channels,
      std::size_t kernel,
      std::size_t stride,
      std::size_t pad,
      bool cover_all = false)
      : Layer<float>(std::make_shared<Impl>(
            in_channels,
            out_channels,
            kernel,
            kernel,
            stride,
            stride,
            pad,
            pad,
            cover_all)) {}
};

}  // namespace connection
}  // namespace layers
}  // namespace xnn

#endif  // __XNN_LAYERS_CONNECTION_CONVOLUTION_HPP__
