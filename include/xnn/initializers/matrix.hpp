#ifndef __XNN_INITIALIZER_MATRIX_HPP__
#define __XNN_INITIALIZER_MATRIX_HPP__

namespace xnn {

Arrayf LeCunNormal(std::size_t n_input, std::size_t n_output) {
  static constexpr float mean = 0.0;
  float stddev = 1.0 / std::sqrt(static_cast<float>(n_input));
  return xt::random::randn({n_input, n_output}, mean, stddev);
}

}  // namespace xnn

#endif  // __XNN_INITIALIZER_MATRIX_HPP__
