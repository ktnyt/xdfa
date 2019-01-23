#ifndef __XNN_INITIALIZER_HPP__
#define __XNN_INITIALIZER_HPP__

#include "xtensor/xarray.hpp"

namespace xnn {

template <class T> class Initializer {
public:
  virtual xt::xarray<T> operator()() = 0;
};

} // namespace xnn

#endif // __XNN_INITIALIZER_HPP__
