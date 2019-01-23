#ifndef __XNN_FUNCTIONS_CONNECTION_LINEAR_HPP__
#define __XNN_FUNCTIONS_CONNECTION_LINEAR_HPP__

#include "xnn/function.hpp"
#include "xnn/initializers/matrix.hpp"
#include "xnn/optimizer.hpp"

#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xmath.hpp"

#include <queue>

namespace xnn {
namespace functions {
namespace connection {

class Linear final : public Function<float> {
public:
  class Impl final : public Function<float>::Impl {
  public:
    Impl(std::size_t n_input, std::size_t n_output, Updater<float> rule)
        : W(initializers::LeCunNormal(n_input, n_output)()),
          b(xt::zeros<float>({n_output})), rule(rule) {}

    xt::xarray<float> forward(xt::xarray<float> x) override {
      forward_queue.push(x);
      return xt::linalg::dot(x, W) + b;
    }

    xt::xarray<float> backward(xt::xarray<float> dy) override {
      backward_queue.push(dy);
      return xt::linalg::dot(dy, xt::transpose(W));
    }

    void update() override {
      xt::xarray<float> x = forward_queue.front();
      xt::xarray<float> dy = backward_queue.front();
      forward_queue.pop();
      backward_queue.pop();
      xt::xarray<float> dW = xt::linalg::dot(xt::transpose(x), dy);
      rule(W, dW);
      rule(b, xt::sum(dy, {0}));
    }

  private:
    xt::xarray<float> W;
    xt::xarray<float> b;

    Updater<float> rule;

    std::queue<xt::xarray<float>> forward_queue;
    std::queue<xt::xarray<float>> backward_queue;
  };

  Linear(std::size_t n_input, std::size_t n_output, Updater<float> rule)
      : Function<float>(std::make_shared<Impl>(n_input, n_output, rule)) {}
};

} // namespace connection
} // namespace functions
} // namespace xnn

#endif // __XNN_FUNCTIONS_CONNECTION_LINEAR_HPP__
