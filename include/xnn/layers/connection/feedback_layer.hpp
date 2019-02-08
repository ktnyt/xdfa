#ifndef __XNN_LAYERS_CONNECTION_FEEDBACK_LAYER_HPP__
#define __XNN_LAYERS_CONNECTION_FEEDBACK_LAYER_HPP__

#include "xnn/layer.hpp"

#include "xtensor/xarray.hpp"

#include <functional>

namespace xnn {
namespace layers {
namespace connection {

class FeedbackLayer : public Layer<float> {
  template <class C, class A>
  class Impl final : public Layer<float>::Impl {
   public:
    Impl(
        C connection,
        A activation,
        std::function<xt::xarray<float>(xt::xarray<float>)> feedback)
        : connection(connection), activation(activation), feedback(feedback) {}

    xt::xarray<float> forward(const xt::xarray<float>& x) override {
      return activation.forward(connection.forward(x));
    };

    xt::xarray<float> backward(const xt::xarray<float>& dy) override {
      return connection.backward(activation.backward(feedback(dy)));
    }

    void update() override { connection.update(); }

   private:
    C connection;
    A activation;
    std::function<xt::xarray<float>(xt::xarray<float>)> feedback;
  };

 public:
  template <class C, class A>
  FeedbackLayer(
      C connection,
      A activation,
      std::function<xt::xarray<float>(xt::xarray<float>)> feedback)
      : Layer<float>(
            std::make_shared<Impl<C, A>>(connection, activation, feedback)) {}
};

}  // namespace miscellaneous
}  // namespace layers
}  // namespace xnn

#endif  // __XNN_LAYERS_CONNECTION_FEEDBACK_LAYER_HPP__
