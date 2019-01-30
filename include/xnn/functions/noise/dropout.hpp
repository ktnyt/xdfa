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
  Dropout(xt::xarray<bool>& mask) : mask(mask) {}

  xt::xarray<float> operator()(xt::xarray<float> x) override {
    xt::masked_view(x, mask) = 0.0;
    return x;
  }

 private:
  xt::xarray<bool>& mask;
};

xt::xarray<float> dropout(xt::xarray<float> x, xt::xarray<bool>& mask) {
  return Dropout(mask)(x);
}

}  // namespace noise
}  // namespace functions
}  // namespace xnn

#endif  // __XNN_FUNCTIONS_NOISE_DROPOUT_HPP__
