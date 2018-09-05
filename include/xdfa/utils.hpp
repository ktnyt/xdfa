#ifndef __XDFA_UTILS_HPP__
#define __XDFA_UTILS_HPP__

#include "xdfa/function.hpp"
#include "xdfa/types.hpp"

#include "xtensor/xrandom.hpp"

namespace xdfa {

std::vector<std::vector<int>> indices(Arrayi t) {
  std::vector<std::vector<int>> ret;
  for (int i = 0; i < t.size(); ++i) {
    ret.push_back({i, t[i]});
  }
  return ret;
}

Arrayf LeCunNormal(std::size_t n_input, std::size_t n_output) {
  static constexpr float mean = 0.0;
  float stddev = 1.0 / std::sqrt(static_cast<float>(n_input));
  return xt::random::randn({n_input, n_output}, mean, stddev);
}

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

}  // namespace xdfa

#endif  // __XDFA_UTILS_HPP__
