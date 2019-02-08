#ifndef __XNN_LAYERS_NETWORK_SERIAL_HPP__
#define __XNN_LAYERS_NETWORK_SERIAL_HPP__

#include "xnn/layer.hpp"
#include "xnn/utils/helpers.hpp"

#include <memory>
#include <vector>

namespace xnn {
namespace layers {
namespace network {

template <class T>
class Serial final : public Layer<T> {
  class Impl : public Layer<T>::Impl {
   public:
    template <class Iterable>
    Impl(Iterable iterable) : impls(iterable.begin(), iterable.end()) {}

    xt::xarray<T> forward(const xt::xarray<T>& x) override {
      xt::xarray<T> y = x;
      for (auto impl = impls.begin(); impl != impls.end(); ++impl) {
        auto f = impl->get();
        y = f->forward(y);
      }
      return y;
    }

    xt::xarray<T> backward(const xt::xarray<T>& dy) override {
      xt::xarray<T> dx = dy;
      for (auto impl = impls.rbegin(); impl != impls.rend(); ++impl) {
        auto f = impl->get();
        dx = f->backward(dx);
      }
      return dx;
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
  Serial(Args... args)
      : Layer<T>(std::shared_ptr<Impl>(
            new Impl(utils::to_impl<T>(std::forward<Args>(args)...)))) {}
};

}  // namespace network
}  // namespace layers
}  // namespace xnn

#endif  // __XNN_LAYERS_NETWORK_SERIAL_HPP__
