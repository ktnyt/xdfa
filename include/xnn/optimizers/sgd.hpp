#ifndef __XNN_OPTIMIZERS_SGD_HPP__
#define __XNN_OPTIMIZERS_SGD_HPP__

#include "xnn/optimizer.hpp"

#include "xtensor/xarray.hpp"

namespace xnn {
namespace optimizers {

class SGD final : public UpdateRule<float> {
 public:
  SGD(float lr = 0.05) : lr(lr) {}

  void operator()(xt::xarray<float>& data, xt::xarray<float> grad) override {
    data -= grad * lr;
  }

 private:
  float lr;
};

}  // namespace optimizers
}  // namespace xnn

#endif  // __XNN_OPTIMIZERS_SGD_HPP__
