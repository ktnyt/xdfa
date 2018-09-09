#ifndef __XNN_FUNCTIONS_HPP__
#define __XNN_FUNCTIONS_HPP__

#include <memory>
#include <type_traits>

#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"

#include "xtensor-blas/xlinalg.hpp"

namespace xnn {
namespace functions {

template <class T>
xt::xarray<T> LeCunNormal(std::size_t n_input, std::size_t n_output) {
  static constexpr float mean = 0.0;
  float stddev = 1.0 / std::sqrt(static_cast<float>(n_input));
  return xt::random::randn({n_input, n_output}, mean, stddev);
}

template <class T>
class Function {
 public:
  using value_type = T;
  virtual ~Function() {}
  virtual xt::xarray<T> operator()(xt::xarray<T>) = 0;
};

template <class T>
class Derivable : public Function<T> {
 public:
  virtual ~Derivable() {}
  virtual xt::xarray<T> derivative(xt::xarray<T>) = 0;
};

template <class T>
class Volatile : public Function<T> {
 public:
  virtual ~Volatile() {}
  virtual void update(xt::xarray<T>) = 0;
};

template <class T>
class Sigmoid final : public Derivable<T> {
 public:
  xt::xarray<T> operator()(xt::xarray<T> x) {
    return xt::tanh(x * 0.5) * 0.5 + 0.5;
  }
  xt::xarray<T> derivative(xt::xarray<T> x) { return x * (1 - x); }
};

template <class T>
class Linear final : public Derivable<T>, public Volatile<T> {
 public:
  Linear(std::size_t n_input, std::size_t n_output)
      : W(LeCunNormal<T>(n_input, n_output)) {}
  ~Linear() {}

  xt::xarray<T> operator()(xt::xarray<T> x) {
    return xt::linalg::dot(x, W) + b;
  }

  xt::xarray<T> derivative(xt::xarray<T> x) {
    return xt::linalg::dot(x, W.transpose());
  }
  void update(xt::xarray<T> x) { W += x; }

 private:
  xt::array<T> W;
  xt::array<T> b;
};

}  // namespace functions
}  // namespace xnn

#endif  // __XNN_FUNCTIONS_HPP__
