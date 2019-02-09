#ifndef __XNN_LAYERS_CONNECTION_LINEAR_HPP__
#define __XNN_LAYERS_CONNECTION_LINEAR_HPP__

#include "xnn/functions/connection/linear.hpp"
#include "xnn/initializers.hpp"
#include "xnn/layer.hpp"
#include "xnn/optimizer.hpp"

#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xstrided_view.hpp"

#include <algorithm>
#include <functional>
#include <queue>

namespace xnn {
namespace layers {
namespace connection {

class Linear final : public Layer<float> {
  class Impl final : public Layer<float>::Impl {
   public:
    Impl(std::size_t n_output, Updater<float> rule)
        : b(xt::zeros<float>({n_output})), rule(rule), init(false) {}

    xt::xarray<float> forward(const xt::xarray<float>& x) override {
      forward_queue.push(x);
      if (!init) {
        W = initializers::LeCunNormal()({b.size(), x.shape()[1]});
        init = true;
      }
      return functions::connection::linear(x, W, b);
    }

    xt::xarray<float> backward(const xt::xarray<float>& dy) override {
      backward_queue.push(dy);
      xt::xarray<float> dx = functions::connection::linear_back(dy, W);
      return dx;
    }

    void update() override {
      xt::xarray<float> x = forward_queue.front();
      xt::xarray<float> dy = backward_queue.front();
      forward_queue.pop();
      backward_queue.pop();
      xt::xarray<float> dW = functions::connection::linear_grad(x, dy);
      rule(W, dW);
      rule(b, xt::sum(dy, {0}));
    }

   private:
    xt::xarray<float> W;
    xt::xarray<float> b;

    Updater<float> rule;
    bool init;

    std::queue<xt::xarray<float>> forward_queue;
    std::queue<xt::xarray<float>> backward_queue;
  };

 public:
  Linear(std::size_t n_output, Updater<float> rule)
      : Layer<float>(std::make_shared<Impl>(n_output, rule)) {}
};

}  // namespace connection
}  // namespace layers
}  // namespace xnn

#endif  // __XNN_LAYERS_CONNECTION_LINEAR_HPP__
