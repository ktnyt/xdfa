#ifndef __XNN_OPTIMIZERS_HPP__
#define __XNN_OPTIMIZERS_HPP__

#include "xtensor/xarray.hpp"

#include <limits>

namespace xnn {
namespace optimizers {

template <class T>
class OptimizeRule {
 public:
  virtual xt::xarray<T> diff(xt::xarray<T>) = 0;
};

template <class T>
class AdamRule final : public OptimizeRule<T> {
 public:
  AdamRule(T alpha = 0.001, T beta1 = 0.9, T beta2 = 0.999,
           T eps = std::numeric_limits<T>::epsilon(), T eta = 1.0)
      : alpha(alpha), beta1(beta1), beta2(beta2), eps(eps), eta(eta) {}

  xt::xarray<T> diff(xt::xarray<T> grad) {
    if (grad.shape() != m.shape()) {
      m = xt::xarray<T>(grad.shape());
      v = xt::xarray<T>(grad.shape());
    }

    m += (1.0 - beta1) * (grad - m);
    v += (1.0 - beta2) * (grad * grad - v);

    return eta * (m / (v + eps));
  }

 private:
  T alpha;
  T beta1;
  T beta2;
  T eps;
  T eta;

  xt::xarray<T> m;
  xt::xarray<T> v;
};

}  // namespace optimizers
}  // namespace xnn

#endif  // __XNN_OPTIMIZERS_HPP__
