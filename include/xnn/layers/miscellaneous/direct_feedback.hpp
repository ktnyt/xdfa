#ifndef __XNN_LAYERS_MISCELLANEOUS_DIRECT_FEEDBACK_HPP__
#define __XNN_LAYERS_MISCELLANEOUS_DIRECT_FEEDBACK_HPP__

#include "xnn/layer.hpp"
#include "xnn/utils/helpers.hpp"

namespace xnn {
namespace layers {
namespace miscellaneous {

template <class T>
class DirectFeedback final : public Layer<T> {
  class Impl : public Layer<float>::Impl {
   public:
    template <class Iterable>
    Impl(Iterable iterable) : impls(iterable.begin(), iterable.end()) {}

    xt::xarray<T> forward(xt::xarray<T> x) override {
      xt::xarray<T> y = x;
      for (auto impl = impls.begin(); impl != impls.end(); ++impl) {
        auto f = impl->get();
        y = f->forward(y);
      }
      return y;
    }

    xt::xarray<T> backward(xt::xarray<T> dy) override {
      for (auto impl = impls.begin(); impl != impls.end(); ++impl) {
        auto f = impl->get();
        f->backward(dy);
      }
      return dy;
    }

    void update() override {
      for (auto impl = impls.begin(); impl != impls.end(); ++impl) {
        auto f = impl->get();
        f->update();
      }
    }

   private:
    std::vector<std::shared_ptr<typename Layer<T>::Impl>> impls;
  };

 public:
  template <class... Args>
  DirectFeedback(Args... args)
      : Layer<T>(std::shared_ptr<Impl>(
            new Impl(utils::to_impl<T>(std::forward<Args>(args)...)))) {}
};

}  // namespace miscellaneous
}  // namespace layers
}  // namespace xnn

#endif  // __XNN_LAYERS_MISCELLANEOUS_DIRECT_FEEDBACK_HPP__
