#ifndef __XNN_ACTIVATION_HPP__
#define __XNN_ACTIVATION_HPP__

#include "xnn/function.hpp"
#include "xnn/utils.hpp"

#include "xtensor/xindex_view.hpp"
#include "xtensor/xmath.hpp"

namespace xnn {

Arrayf sigmoid(Arrayf x) { return xt::tanh(x * 0.5) * 0.5 + 0.5; }
Arrayf dsigmoid(Arrayf y) { return y * (1.0 - y); }

class Sigmoid final : public Function {
 public:
  Arrayf forward(Arrayf x) { return memory = sigmoid(x); }
  Arrayf backward(Arrayf d) { return d * dsigmoid(memory); }

 private:
  Arrayf memory;
};

Arrayf softmax(Arrayf x) {
  xt::transpose(x) -= xt::amax(x, {1});
  Arrayf e = xt::exp(x);
  xt::transpose(e) /= xt::sum(e, {1});
  return e;
}

class SoftmaxCrossEntropy final : public Function {
 public:
  void set_labels(Arrayi t) { labels = t; }

  Arrayf forward(Arrayf x) {
    memory = softmax(x);
    auto idx = indices(labels);
    auto y = xt::index_view(memory, idx);
    return xt::mean(-xt::log(y));
  }

  SoftmaxCrossEntropy& with(Arrayi t) {
    set_labels(t);
    return *this;
  }

  Arrayf backward(Arrayf dy) {
    auto idx = indices(labels);
    auto p = memory;
    xt::index_view(p, idx) -= 1;
    return p / labels.shape()[0];
  }

  Arrayf grads() { return backward(0.0); }

 private:
  Arrayi labels;
  Arrayf memory;
};

}  // namespace xnn

#endif  // __XNN_ACTIVATION_HPP__
