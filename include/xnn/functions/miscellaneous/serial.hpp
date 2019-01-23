#ifndef __XNN_FUNCTIONS_MISCELLANEOUS_SERIAL_HPP__
#define __XNN_FUNCTIONS_MISCELLANEOUS_SERIAL_HPP__

#include "xnn/function.hpp"

#include <array>
#include <list>
#include <memory>
#include <vector>

namespace xnn {
namespace functions {
namespace miscellaneous {

template <class T>
std::list<std::shared_ptr<typename Function<T>::Impl>> convert() {
  return {};
}

template <class T, class Head, class... Tail>
std::list<std::shared_ptr<typename Function<T>::Impl>> convert(Head head,
                                                               Tail... tail) {
  std::list<std::shared_ptr<typename Function<T>::Impl>> list =
      convert<T, Tail...>(tail...);
  list.push_front(
      std::static_pointer_cast<typename Function<T>::Impl>(head.get()));
  return list;
}

template <class T> class Serial final : public Function<T> {
  class Impl : public Function<T>::Impl {
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
    std::vector<std::shared_ptr<typename Function<T>::Impl>> impls;
  };

public:
  template <class... Args>
  Serial(Args... args)
      : Function<T>(std::shared_ptr<Impl>(new Impl(convert<T>(args...)))) {}
};

} // namespace miscellaneous
} // namespace functions
} // namespace xnn

#endif // __XNN_FUNCTIONS_MISCELLANEOUS_SERIAL_HPP__
