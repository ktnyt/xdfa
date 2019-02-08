#ifndef __XNN_FUNCTION_HPP__
#define __XNN_FUNCTION_HPP__

#include "xtensor/xarray.hpp"

#include <functional>

namespace xnn {

template <class T>
class Function {
 public:
  virtual xt::xarray<T> operator()(const xt::xarray<T>&) = 0;
  xt::xarray<T> operator()(xt::xarray<T>&& x) { return operator()(x); }
};

}  // namespace xnn

#endif  // __XNN_FUNCTION_HPP__
