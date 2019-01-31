#ifndef __XNN_UTILS_XTENSOR_HPP__
#define __XNN_UTILS_XTENSOR_HPP__

#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xview.hpp"

#include <numeric>
#include <vector>

namespace xnn {
namespace utils {

template <class E>
inline auto swapaxes(E&& e, std::size_t axis1, std::size_t axis2) {
  std::vector<std::size_t> axes(e.shape().size());
  std::iota(axes.begin(), axes.end(), 0);
  std::swap(axes[axis1], axes[axis2]);
  return xt::transpose(e, axes);
}

template <class T>
xt::xarray<T> repeat0(xt::xarray<T> a, std::size_t n) {
  std::vector<std::size_t> shape(a.shape().begin(), a.shape().end());
  std::size_t leading = shape[0];
  shape[0] *= n;
  xt::xarray<T> out = xt::zeros<T>(shape);
  for (std::size_t i = 0; i < leading; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      xt::view(out, n * i + j) = xt::view(a, i);
    }
  }
  return out;
}

template <class T>
xt::xarray<T> tile(xt::xarray<T> c, std::vector<std::size_t>&& reps) {
  std::vector<std::size_t> tup;
  for (std::size_t i = reps.size(); i < c.shape().size(); ++i) {
    tup.push_back(1);
  }
  tup.insert(tup.end(), reps.begin(), reps.end());
  std::vector<std::size_t> shape_out(tup.size());
  for (std::size_t i = 0; i < tup.size(); ++i) {
    shape_out[i] = c.shape()[i] * tup[i];
  }
  std::size_t n = c.size();
  if (n > 0) {
    for (std::size_t i = 0; i < tup.size(); ++i) {
      std::size_t dim_in = c.shape()[i];
      std::size_t nrep = tup[i];
      if (nrep != 1) {
        xt::xarray<T> tmp = xt::reshape_view(c, {c.size() / n, n});
        c = repeat0(tmp, nrep);
      }
      n /= dim_in;
    }
  }
  c.reshape(shape_out);
  return c;
}

}  // namespace utils
}  // namespace xnn

#endif  // __XNN_UTILS_XTENSOR_HPP__
