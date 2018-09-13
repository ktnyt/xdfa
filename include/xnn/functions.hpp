#ifndef __XNN_FUNCTIONS_HPP__
#define __XNN_FUNCTIONS_HPP__

#include <memory>
#include <type_traits>

#include "xtensor/xarray.hpp"
#include "xtensor/xindex_view.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xsort.hpp"

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
    return xt::linalg::dot(x, xt::transpose(W));
  }

  void update(xt::xarray<T> x) {
    W += x;
    b += xt::sum(x, 1);
  }

 private:
  xt::xarray<T> W;
  xt::xarray<T> b;
};

template <class T>
std::vector<std::vector<int>> indices(xt::xarray<T> t) {
  std::vector<std::vector<int>> ret;
  for (int i = 0; i < t.size(); ++i) {
    ret.push_back({i, t[i]});
  }
  return ret;
}

template <class T>
xt::xarray<T> softmax(xt::xarray<T> x) {
  xt::transpose(x) -= xt::amax(x, {1});
  xt::xarray<T> e = xt::exp(x);
  xt::transpose(e) /= xt::sum(e, {1});
  return e;
}

template <class T, class U>
class SoftmaxCrossEntropy final : public Function<T> {
 public:
  xt::xarray<T> operator()(xt::xarray<T> x) {
    memory = softmax(x);
    auto idx = indices(labels);
    auto y = xt::index_view(memory, idx);
    return xt::mean(-xt::log(y));
  }

  SoftmaxCrossEntropy& with(xt::xarray<U> t) {
    labels = t;
    return *this;
  }

  xt::xarray<T> grads() {
    auto idx = indices(labels);
    auto p = memory;
    xt::index_view(p, idx) -= 1;
    return p / labels.shape()[0];
  }

 private:
  xt::xarray<U> labels;
  xt::xarray<T> memory;
};

template <class T, class U>
xt::xarray<T> accuracy(xt::xarray<U> t, xt::xarray<T> x) {
  xt::xarray<U> y = xt::argmax(x, 1);
  xt::xarray<T> f = xt::equal(t, y);
  return xt::sum(f) / f.size();
}

}  // namespace functions
}  // namespace xnn

#endif  // __XNN_FUNCTIONS_HPP__
