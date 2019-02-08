#ifndef __XNN_FUNCTIONS_MANIPULATION_FLATTEN_HPP__
#define __XNN_FUNCTIONS_MANIPULATION_FLATTEN_HPP__

#include "xnn/function.hpp"

#include <functional>
#include <numeric>
#include <vector>

namespace xnn {
namespace functions {
namespace manipulation {

template <class T>
class Flatten final : public Function<T> {
 public:
  xt::xarray<T> operator()(const xt::xarray<T>& x) {
    std::vector<std::size_t> shape(x.shape().begin(), x.shape().end());
    std::size_t batch_size = shape[0];
    std::size_t sample_size = static_cast<T>(std::accumulate(
        shape.begin() + 1, shape.end(), 1, std::multiplies<std::size_t>()));
    xt::xarray<T> y = xt::reshape_view(x, {batch_size, sample_size});
    return y;
  }
};

template <class T, class S>
class Unflatten final : public Function<T> {
 public:
  Unflatten(S& shape) : shape(shape) {}

  xt::xarray<T> operator()(const xt::xarray<T>& x) {
    xt::xarray<T> y = xt::reshape_view(x, std::forward<S>(shape));
    return y;
  }

 private:
  S& shape;
};

template <class T>
xt::xarray<T> flatten(const xt::xarray<T>& x) {
  return Flatten<T>()(x);
}

template <class T, class S>
xt::xarray<T> unflatten(const xt::xarray<T>& x, S& shape) {
  return Unflatten<T, S>(shape)(x);
}

}  // namespace manipulation
}  // namespace functions
}  // namespace xnn

#endif  // __XNN_FUNCTIONS_MANIPULATION_FLATTEN_HPP__
