#ifndef __XNN_LAYERS_CONNECTION_CONVOLUTION_HPP__
#define __XNN_LAYERS_CONNECTION_CONVOLUTION_HPP__

#include "xnn/functions/connection/convolution.hpp"
#include "xnn/initializers.hpp"
#include "xnn/layer.hpp"
#include "xnn/utils/convolution.hpp"

#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xarray.hpp"

namespace xnn {
namespace layers {
namespace connection {

class Convolution2D final : public Layer<float> {
  class Impl final : public Layer<float>::Impl {
   public:
    Impl(
        std::size_t in_channels,
        std::size_t out_channels,
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
      forward_queue.push(x);
      return functions::connection::convolution2d(
          x, W, sy, sx, ph, pw, cover_all);
    }

    xt::xarray<float> backward(xt::xarray<float> dy) override {
      forward_queue.push(dy);
      return functions::connection::deconvolution2d(
          dy, W, sy, sx, ph, pw, cover_all);
    }

    void update() override {
      xt::xarray<float> x = forward_queue.front();
      xt::xarray<float> dy = backward_queue.front();
      forward_queue.pop();
      backward_queue.pop();
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
      xt::xarray<float> dW =
          xt::linalg::tensordot(dy, col, {0, 2, 3}, {0, 4, 5});
      rule(W, dW);
    }

   private:
    xt::xarray<float> W;
    std::size_t sy;
    std::size_t sx;
    std::size_t ph;
    std::size_t pw;
    bool cover_all;

    Updater<float> rule;

    std::queue<xt::xarray<float>> forward_queue;
    std::queue<xt::xarray<float>> backward_queue;
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
