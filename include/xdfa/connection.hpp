#ifndef __XDFA_CONNECTION_HPP__
#define __XDFA_CONNECTION_HPP__

#include "xdfa/function.hpp"
#include "xdfa/types.hpp"
#include "xdfa/utils.hpp"

#include "xtensor-blas/xlinalg.hpp"

namespace xdfa {

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

}  // namespace xdfa

#endif  // __XDFA_CONNECTION_HPP__
