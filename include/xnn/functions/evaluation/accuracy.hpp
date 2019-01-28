#ifndef __XNN_FUNCTIONS_EVALUATION_ACCURACY_HPP__
#define __XNN_FUNCTIONS_EVALUATION_ACCURACY_HPP__

#include "xtensor/xarray.hpp"
#include "xtensor/xsort.hpp"

namespace xnn {
namespace functions {
namespace evaluation {

xt::xarray<float> accuracy(xt::xarray<int> t, xt::xarray<float> x) {
  xt::xarray<int> y = xt::argmax(x, 1);
  xt::xarray<float> f = xt::equal(t, y);
  return xt::sum(f) / f.size();
}

}  // namespace evaluation
}  // namespace functions
}  // namespace xnn

#endif  // __XNN_FUNCTIONS_EVALUATION_ACCURACY_HPP__
