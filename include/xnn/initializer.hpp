#ifndef __XNN_INITIALIZER_HPP__
#define __XNN_INITIALIZER_HPP__

#include "xtensor/xarray.hpp"

#include <algorithm>
#include <tuple>

namespace xnn {

template <class T> class Initializer {
public:
  virtual xt::xarray<T> operator()(std::vector<std::size_t> shape) = 0;
};

template <class S> std::tuple<std::size_t, std::size_t> get_fans(S shape) {
  std::size_t receptive_field_size = 1;
  for (std::size_t i = 2; i < shape.size(); ++i) {
    receptive_field_size *= shape[i];
  }
  std::size_t fan_in = shape[1] * receptive_field_size;
  std::size_t fan_out = shape[0] * receptive_field_size;
  return std::tuple<std::size_t, std::size_t>(fan_in, fan_out);
}

} // namespace xnn

#endif // __XNN_INITIALIZER_HPP__
