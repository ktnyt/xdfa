#ifndef __XNN_FUNCTIONS_MISCELLANEOUS_FEEDBACK_LAYER_HPP__
#define __XNN_FUNCTIONS_MISCELLANEOUS_FEEDBACK_LAYER_HPP__

#include "xnn/function.hpp"

#include "xtensor/xarray.hpp"

#include <functional>

namespace xnn {
namespace functions {
namespace miscellaneous {

class FeedbackLayer : public Function<float> {
  template <class C, class A> class Impl final : public Function<float>::Impl {
  public:
    Impl(C connection, A activation,
         std::function<xt::xarray<float>(xt::xarray<float>)> feedback)
        : connection(connection), activation(activation), feedback(feedback) {}

    xt::xarray<float> forward(xt::xarray<float> x) override {
      return activation.forward(connection.forward(x));
    };

    xt::xarray<float> backward(xt::xarray<float> dy) override {
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
  FeedbackLayer(C connection, A activation,
                std::function<xt::xarray<float>(xt::xarray<float>)> feedback)
      : Function<float>(
            std::make_shared<Impl<C, A>>(connection, activation, feedback)) {}
};

} // namespace miscellaneous
} // namespace functions
} // namespace xnn

#endif // __XNN_FUNCTIONS_MISCELLANEOUS_FEEDBACK_LAYER_HPP__
