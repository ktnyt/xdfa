#define XTENSOR_USE_XSIMD
#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xindex_view.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xview.hpp"

#include "xtensor-blas/xlinalg.hpp"

#include <cmath>
#include <iostream>

#include "mnist.hpp"

using Expression = xt::xexpression<float>;
using Arrayf = xt::xarray<float>;
using Arrayi = xt::xarray<int>;

class Function {
 public:
  virtual Arrayf forward(Arrayf) = 0;
  virtual Arrayf backward(Arrayf) = 0;
};

std::vector<std::vector<int>> indices(Arrayi t) {
  std::vector<std::vector<int>> ret;
  for (int i = 0; i < t.size(); ++i) {
    ret.push_back({i, t[i]});
  }
  return ret;
}

Arrayf sigmoid(Arrayf x) { return xt::tanh(x * 0.5) * 0.5 + 0.5; }
Arrayf dsigmoid(Arrayf y) { return y * (1.0 - y); }

Arrayf softmax(Arrayf x) {
  xt::transpose(x) -= xt::amax(x, {1});
  Arrayf e = xt::exp(x);
  xt::transpose(e) /= xt::sum(e, {1});
  return e;
}

class Sigmoid final : public Function {
 public:
  Arrayf forward(Arrayf x) { return memory = sigmoid(x); }
  Arrayf backward(Arrayf d) { return d * dsigmoid(memory); }

 private:
  Arrayf memory;
};

class SoftmaxCrossEntropy final : public Function {
 public:
  void set_labels(Arrayi t) { labels = t; }

  Arrayf forward(Arrayf x) {
    memory = softmax(x);
    auto idx = indices(labels);
    auto y = xt::index_view(memory, idx);
    return xt::mean(-xt::log(y));
  }

  Arrayf backward(Arrayf dy) {
    auto idx = indices(labels);
    auto p = memory;
    xt::index_view(p, idx) -= 1;
    return p / labels.shape()[0];
  }

 private:
  Arrayi labels;
  Arrayf memory;
};

Arrayf LeCunNormal(std::size_t n_input, std::size_t n_output) {
  static constexpr float mean = 0.0;
  float stddev = 1.0 / std::sqrt(static_cast<float>(n_input));
  return xt::random::randn({n_input, n_output}, mean, stddev);
}

class Layer final : public Function {
 public:
  Layer(std::size_t n_input, std::size_t n_output, float lr = 0.05)
      : w(LeCunNormal(n_input, n_output)), lr(lr) {}

  Arrayf forward(Arrayf x) { return xt::linalg::dot(memory = x, w); }

  Arrayf backward(Arrayf dy) {
    auto dw = xt::linalg::dot(xt::transpose(memory), dy);
    auto dx = xt::linalg::dot(dy, xt::transpose(w));
    w -= dw * lr;
    return dx;
  }

 private:
  Arrayf w;
  Arrayf b;
  float lr;
  Arrayf memory;
};

class Feedback final : public Function {
 public:
  Feedback(std::size_t n_input, std::size_t n_output)
      : w(LeCunNormal(n_input, n_output)) {}
  Arrayf forward(Arrayf x) { return x; }
  Arrayf backward(Arrayf dy) { return xt::linalg::dot(dy, w); }

 private:
  Arrayf w;
};

class Combine final : public Function {
 public:
  Combine(std::initializer_list<Function*> init) : functions(init) {}

  Arrayf forward(Arrayf x) {
    Arrayf y = x;
    for (std::size_t i = 0; i < functions.size(); ++i) {
      y = functions[i]->forward(y);
    }
    return y;
  }

  Arrayf backward(Arrayf dy) {
    Arrayf dx = dy;
    for (std::size_t i = 0; i < functions.size(); ++i) {
      dx = functions[functions.size() - i - 1]->backward(dx);
    }
    return dx;
  }

 private:
  std::vector<Function*> functions;
};

int main() {
  Arrayf train = mnist::read_images<float>("mnist/train-images-idx3-ubyte");
  std::cout << "hoge" << std::endl;
  std::cout << xt::view(train, 1) << std::endl;
  std::cout << "hoge" << std::endl;

  return 0;
}
