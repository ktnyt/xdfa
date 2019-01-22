#ifndef __XNN_OPTIMIZERS_ADAM_HPP__
#define __XNN_OPTIMIZERS_ADAM_HPP__

#include "xnn/optimizer.hpp"

#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xmath.hpp"

namespace xnn {
namespace optimizers {

class Adam : public UpdateRule<float> {
public:
  Adam(float alpha = 0.001, float beta1 = 0.9, float beta2 = 0.999,
       float epsilon = 1e-08)
      : alpha(alpha), beta1(beta1), beta2(beta2), epsilon(epsilon) {}

  void operator()(xt::xarray<float> &data, xt::xarray<float> grad) override {
    if (m.size() != grad.size() || v.size() != grad.size()) {
      m = xt::zeros<float>(grad.shape());
      v = xt::zeros<float>(grad.shape());
    }

    m += (1.0 - beta1) * (grad - m);
    v += (1.0 - beta2) * (grad * grad - v);

    data -= alpha * m / (xt::sqrt(v) + epsilon);
  }

private:
  float alpha;
  float beta1;
  float beta2;
  float epsilon;

  xt::xarray<float> m;
  xt::xarray<float> v;
};

} // namespace optimizers
} // namespace xnn

#endif // __XNN_OPTIMIZERS_ADAM_HPP__
