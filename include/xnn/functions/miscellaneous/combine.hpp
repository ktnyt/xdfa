#ifndef __XNN_FUNCTIONS_MISCELLANEOUS_COMBINE_HPP__
#define __XNN_FUNCTIONS_MISCELLANEOUS_COMBINE_HPP__

#include "xnn/function.hpp"

#include <vector>

namespace xnn {
namespace functions {
namespace miscellaneous {

template <class T> class Combine final : public Function<T> {
public:
  Combine(std::initializer_list<Function<T> *> init) : functions(init) {}

  xt::xarray<T> forward(xt::xarray<T> x) override {
    xt::xarray<T> y = x;
    for (std::size_t i = 0; i < functions.size(); ++i) {
        Function<T>* f = functions[i];
        y = f->forward(y);
    }
    return y;
  }

  xt::xarray<T> backward(xt::xarray<T> dy) override {
    xt::xarray<T> dx = dy;
    for (std::size_t i = 0; i < functions.size(); ++i) {
        Function<T>* f = functions[functions.size() - (i + 1)];
        dx = f->backward(dx);
    }
    return dx;
  }

  void update() override {
    for (std::size_t i = 0; i < functions.size(); ++i) {
        Function<T>* f = functions[i];
      f->update();
    }
  }

private:
  std::vector<Function<T> *> functions;
};

} // namespace miscellaneous
} // namespace functions
} // namespace xnn
#endif // __XNN_FUNCTIONS_MISCELLANEOUS_COMBINE_HPP__
