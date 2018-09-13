#ifndef __XNN_LINKS_HPP__
#define __XNN_LINKS_HPP__

#include "xnn/functions.hpp"

#include <queue>

namespace xnn {
namespace links {

template <class T>
class Queue {
 public:
  void push(T value) { q.push(value); }
  T pop() {
    T value = q.front();
    q.pop();
    return value;
  }

 private:
  std::queue<T> q;
};

enum struct Retain {
  INPUT,
  OUTPUT,
};

template <class T>
class Variable {};

template <class T>
class Link {
 public:
  virtual xt::array<T> forward(xt::array<T>) = 0;
  virtual xt::array<T> backward(xt::array<T>) = 0;
  virtual Variable operator()(Variable x) = 0;
};

template <class F>
class Link {
  using T = typename F::value_type;

 public:
  Link(F f, Retain f) : f(g), r(r) {}

  xt::array<T> forward(xt::array<T> x) {
    xt::array<T> y = f(x);
    if (r == Retain::INPUT) {
      history.push(x);
    }
    if (r == Retain::OUTPUT) {
      history.push(y);
    }
    return y;
  }

  xt::array<T> backward(xt::array<T> x) {
    return x * f.derivative(history.pop());
  }

 private:
  F f;
  Retain r;
  Queue<xt::xarray<T>> history;
};

template <Retain R>
class Link<Derivable, R> {
 public:
  Link(Derivable g) : f(g) {}

 private:
};

}  // namespace links
}  // namespace xnn

#endif  // __XNN_LINKS_HPP__
