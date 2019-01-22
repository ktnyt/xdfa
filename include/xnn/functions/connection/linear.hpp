#ifndef __XNN_FUNCTIONS_CONNECTION_LINEAR_HPP__
#define __XNN_FUNCTIONS_CONNECTION_LINEAR_HPP__

#include "xnn/function.hpp"
#include "xnn/initializers/matrix.hpp"

#include "xtensor-blas/xlinalg.hpp"

#include <functional>

namespace xnn {
namespace functions {
namespace connection {

class Linear final : public Function<float> {
public:
  Linear(std::size_t n_input, std::size_t n_output,
         std::function<void(xt::xarray<float>&, xt::xarray<float>)> rule)
      : W(initializers::LeCunNormal(n_input, n_output)()), rule(rule) {}

  xt::xarray<float> forward(xt::xarray<float> x) override {
    return xt::linalg::dot(forward_memory = x, W);
  }

  xt::xarray<float> backward(xt::xarray<float> dy) override {
    return xt::linalg::dot(backward_memory = dy, xt::transpose(W));
  }

  void update() override {
    auto dW = xt::linalg::dot(xt::transpose(forward_memory), backward_memory);
    rule(W, dW);
  }

private:
  xt::xarray<float> W;
  xt::xarray<float> forward_memory;
  xt::xarray<float> backward_memory;
  std::function<void(xt::xarray<float>&, xt::xarray<float>)> rule;
};

} // namespace connection
} // namespace functions
} // namespace xnn

#endif // __XNN_FUNCTIONS_CONNECTION_LINEAR_HPP__
