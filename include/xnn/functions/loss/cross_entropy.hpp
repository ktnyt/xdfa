#ifndef __XNN_FUNCTIONS_LOSS_CROSS_ENTROPY_HPP__
#define __XNN_FUNCTIONS_LOSS_CROSS_ENTROPY_HPP__

#include "xnn/function.hpp"
#include "xnn/utils/helpers.hpp"

#include "xtensor/xarray.hpp"
#include "xtensor/xindex_view.hpp"
#include "xtensor/xmath.hpp"

namespace xnn {
namespace functions {
namespace loss {

class CrossEntropy final : public Function<float> {
 public:
  CrossEntropy(const xt::xarray<int>& t) : t(t) {}

  xt::xarray<float> operator()(const xt::xarray<float>& x) override {
    auto idx = utils::to_index(t);
    auto y = xt::index_view(x, idx);
    return xt::mean(-xt::log(y));
  }

 private:
  const xt::xarray<int>& t;
};

inline xt::xarray<float> cross_entropy(
    const xt::xarray<float>& x, const xt::xarray<int>& t) {
  return CrossEntropy(t)(x);
}

}  // namespace loss
}  // namespace functions
}  // namespace xnn

#endif  // __XNN_FUNCTIONS_LOSS_CROSS_ENTROPY_HPP__
