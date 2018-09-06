#ifndef __XNN_UTILS_HPP__
#define __XNN_UTILS_HPP__

#include "xnn/function.hpp"
#include "xnn/types.hpp"

#include "xtensor/xrandom.hpp"

namespace xnn {

std::vector<std::vector<int>> indices(Arrayi t) {
  std::vector<std::vector<int>> ret;
  for (int i = 0; i < t.size(); ++i) {
    ret.push_back({i, t[i]});
  }
  return ret;
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

}  // namespace xnn

#endif  // __XNN_UTILS_HPP__
