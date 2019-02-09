#ifndef __XNN_LAYERS_CONNECTION_LINEAR_FEEDBACK_HPP__
#define __XNN_LAYERS_CONNECTION_LINEAR_FEEDBACK_HPP__

#include "xnn/initializers.hpp"
#include "xnn/layers/connection/feedback_layer.hpp"
#include "xnn/layers/connection/linear.hpp"

#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xmanipulation.hpp"

namespace xnn {
namespace layers {
namespace connection {
namespace internal {

class linear_feedback {
 public:
  linear_feedback(std::size_t n_output) : n_output(n_output), init(false) {}

  xt::xarray<float> operator()(const xt::xarray<float>& x) {
    if (!init) {
      B = initializers::LeCunNormal()({n_output, x.shape()[1]});
      init = true;
    }
    return xt::linalg::dot(x, xt::transpose(B));
  };

 private:
  xt::xarray<float> B;

  std::size_t n_output;
  bool init;
};

}  // namespace internal

class LinearFeedback final : public FeedbackLayer {
 public:
  template <class A>
  LinearFeedback(std::size_t n_output, Updater<float> rule, A activation)
      : FeedbackLayer(
            Linear(n_output, rule),
            activation,
            internal::linear_feedback(n_output)) {}
};

}  // namespace connection
}  // namespace layers
}  // namespace xnn

#endif  // __XNN_LAYERS_CONNECTION_LINEAR_FEEDBACK_HPP__
