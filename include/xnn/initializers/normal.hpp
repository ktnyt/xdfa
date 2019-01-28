#ifndef __XNN_INITIALIZERS_NORMAL_HPP__
#define __XNN_INITIALIZERS_NORMAL_HPP__

#include "xnn/initializer.hpp"

#include "xtensor/xrandom.hpp"

#include <cmath>

namespace xnn {
namespace initializers {

class LeCunNormal final : public Initializer<float> {
  static constexpr float mean = 0.0;

 public:
  LeCunNormal(float scale = 1.0) : scale(scale) {}

  xt::xarray<float> operator()(std::vector<std::size_t> shape) override {
    std::size_t fan_in;
    std::size_t fan_out;
    std::tie(fan_in, fan_out) = get_fans(shape);
    float stddev = scale * std::sqrt(1.0 / static_cast<float>(fan_in));
    return xt::random::randn(shape, 0.0f, stddev);
  }

 private:
  float scale;
  float stddev;
};

}  // namespace initializers
}  // namespace xnn

#endif  // __XNN_INITIALIZERS_NORMAL_HPP__
