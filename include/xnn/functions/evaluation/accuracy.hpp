#ifndef __XNN_FUNCTIONS_EVALUATION_ACCURACY_HPP__
#define __XNN_FUNCTIONS_EVALUATION_ACCURACY_HPP__

#include "xtensor/xarray.hpp"
#include "xtensor/xsort.hpp"

namespace xnn {
namespace functions {
namespace evaluation {

class Accuracy final : public Function<float> {
 public:
  Accuracy(const xt::xarray<int>& t) : t(t) {}

  xt::xarray<float> operator()(const xt::xarray<float>& x) {
    xt::xarray<int> y = xt::argmax(x, 1);
    xt::xarray<float> f = xt::equal(t, y);
    return xt::sum(f) / f.size();
  }

 private:
  const xt::xarray<int>& t;
};

inline xt::xarray<float> accuracy(
    const xt::xarray<int>& t, const xt::xarray<float>& x) {
  return Accuracy(t)(x);
}

}  // namespace evaluation
}  // namespace functions
}  // namespace xnn

#endif  // __XNN_FUNCTIONS_EVALUATION_ACCURACY_HPP__
