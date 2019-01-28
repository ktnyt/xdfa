#ifndef __XNN_OPTIMIZER_HPP__
#define __XNN_OPTIMIZER_HPP__

#include "xtensor/xarray.hpp"

namespace xnn {

template <class T>
using Updater = std::function<void(xt::xarray<T>&, xt::xarray<T>)>;

template <class T>
class UpdateRule {
 public:
  virtual void operator()(xt::xarray<T>& data, xt::xarray<T> grad) = 0;
};

}  // namespace xnn

#endif  // __XNN_OPTIMIZER_HPP__
