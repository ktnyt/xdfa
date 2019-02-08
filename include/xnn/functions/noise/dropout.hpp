#ifndef __XNN_FUNCTIONS_NOISE_DROPOUT_HPP__
#define __XNN_FUNCTIONS_NOISE_DROPOUT_HPP__

#include "xnn/function.hpp"

#include "xtensor/xarray.hpp"
#include "xtensor/xmasked_view.hpp"

namespace xnn {
namespace functions {
namespace noise {

class Dropout final : public Function<float> {
 public:
  Dropout(const xt::xarray<bool>& mask) : mask(mask) {}

  xt::xarray<float> operator()(const xt::xarray<float>& x) override {
    xt::xarray<float> y = x;
    xt::masked_view(y, const_cast<xt::xarray<bool>&>(mask)) = 0.0f;
    return y;
  }

 private:
  const xt::xarray<bool>& mask;
};

inline xt::xarray<float> dropout(
    const xt::xarray<float>& x, const xt::xarray<bool>& mask) {
  return Dropout(mask)(x);
}

}  // namespace noise
}  // namespace functions
}  // namespace xnn

#endif  // __XNN_FUNCTIONS_NOISE_DROPOUT_HPP__
