#ifndef __XNN_LAYERS_CONNECTION_LINEAR_FEEDBACK_HPP__
#define __XNN_LAYERS_CONNECTION_LINEAR_FEEDBACK_HPP__

#include "xnn/initializers.hpp"
#include "xnn/layers/connection/linear.hpp"
#include "xnn/layers/miscellaneous/feedback_layer.hpp"

#include "xtensor-blas/xlinalg.hpp"

namespace xnn {
namespace layers {
namespace connection {
namespace internal {

class linear_feedback {
 public:
  linear_feedback(std::size_t n_input, std::size_t n_output)
      : B(initializers::LeCunNormal()({n_input, n_output})) {}

  xt::xarray<float> operator()(const xt::xarray<float>& x) {
    return xt::linalg::dot(x, B);
  };

 private:
  xt::xarray<float> B;
};

}  // namespace internal

class LinearFeedback final : public miscellaneous::FeedbackLayer {
 public:
  template <class A>
  LinearFeedback(
      std::size_t n_input,
      std::size_t n_output,
      std::size_t n_final,
      Updater<float> rule,
      A activation)
      : FeedbackLayer(
            Linear(n_input, n_output, rule),
            activation,
            internal::linear_feedback(n_final, n_output)) {}
};

}  // namespace connection
}  // namespace layers
}  // namespace xnn

#endif  // __XNN_LAYERS_CONNECTION_LINEAR_FEEDBACK_HPP__
