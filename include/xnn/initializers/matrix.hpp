#ifndef __XNN_INITIALIZERS_MATRIX_HPP__
#define __XNN_INITIALIZERS_MATRIX_HPP__

#include "xnn/initializer.hpp"

#include "xtensor/xrandom.hpp"

namespace xnn {
namespace initializers {

class LeCunNormal final : public Initializer<float> {
  static constexpr float mean = 0.0;

public:
  LeCunNormal(std::size_t n_input, std::size_t n_output)
      : n_input(n_input), n_output(n_output),
        stddev(1.0 / std::sqrt(static_cast<float>(n_input))) {}

  xt::xarray<float> operator()() override {
    return xt::random::randn({n_input, n_output}, mean, stddev);
  }

private:
  std::size_t n_input;
  std::size_t n_output;
  float stddev;
};

} // namespace initializers
} // namespace xnn

#endif // __XNN_INITIALIZERS_MATRIX_HPP__
