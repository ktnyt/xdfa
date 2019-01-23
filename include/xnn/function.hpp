#ifndef __XNN_FUNCTION_HPP__
#define __XNN_FUNCTION_HPP__

#include "xtensor/xarray.hpp"

#include <functional>
#include <memory>

namespace xnn {

template <class T> class Function {
public:
  class Impl {
  public:
    xt::xarray<T> operator()(xt::xarray<T> x) { return forward(x); }
    virtual xt::xarray<T> forward(xt::xarray<T>) = 0;
    virtual xt::xarray<T> backward(xt::xarray<T>) = 0;
    virtual void update(){};
  };

  template <class U>
  Function(std::shared_ptr<U> ptr) : ptr(std::static_pointer_cast<Impl>(ptr)) {}
  xt::xarray<T> forward(xt::xarray<T> x) { return ptr->forward(x); }
  xt::xarray<T> backward(xt::xarray<T> dy) { return ptr->backward(dy); }
  void update() { ptr->update(); };

  std::shared_ptr<Impl> get() const { return ptr; }

protected:
  std::shared_ptr<Impl> ptr;
};

} // namespace xnn

#endif // __XNN_FUNCTION_HPP__
