#ifndef __XNN_LAYER_HPP__
#define __XNN_LAYER_HPP__

#include "xtensor/xarray.hpp"

#include <functional>
#include <memory>

namespace xnn {

template <class T>
class Layer {
 public:
  class Impl {
   public:
    virtual xt::xarray<T> forward(const xt::xarray<T>&) = 0;
    virtual xt::xarray<T> backward(const xt::xarray<T>&) = 0;
    virtual void update(){};

    xt::xarray<T> forward(xt::xarray<T>&& x) { return forward(x); }
    xt::xarray<T> backward(xt::xarray<T>&& x) { return backward(x); }

    xt::xarray<T> operator()(const xt::xarray<T>& x) { return forward(x); }
    xt::xarray<T> operator()(xt::xarray<T>&& x) {
      return forward(std::forward<xt::xarray<T>>(x));
    }
  };

  template <class U>
  Layer(std::shared_ptr<U> ptr) : ptr(std::static_pointer_cast<Impl>(ptr)) {}
  xt::xarray<T> operator()(const xt::xarray<T>& x) { return forward(x); }
  xt::xarray<T> forward(const xt::xarray<T>& x) {
    return ptr->forward(x);
  }
  xt::xarray<T> forward(xt::xarray<T>&& x) {
    return ptr->forward(std::forward<xt::xarray<T>>(x));
  }
  xt::xarray<T> backward(const xt::xarray<T>& dy) {
    return ptr->backward(dy);
  }
  xt::xarray<T> backward(xt::xarray<T>&& x) {
    return ptr->backward(std::forward<xt::xarray<T>>(x));
  }
  void update() { ptr->update(); };

  std::shared_ptr<Impl> get() const { return ptr; }

 protected:
  std::shared_ptr<Impl> ptr;
};

}  // namespace xnn

#endif  // __XNN_LAYER_HPP__
