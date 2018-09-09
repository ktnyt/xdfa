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

template <class F, Retain R>
class Link {
  using T = typename F::value_type;

 public:
  Link(F g) : f(g) {}

  xt::array<T> forward(xt::array<T> x) {
    xt::array<T> y = f(x);
    if (R == Retain::INPUT) {
      history.push(x);
    }
    if (R == Retain::OUTPUT) {
      history.push(y);
    }
    return y;
  }

  xt::array<T> backward(xt::array<T> x) {
    return x * f.derivative(history.pop());
  }

 private:
  F f;
  Queue<xt::xarray<T>> history;
};

template <Retain R>
class Link<Derivable, R> {
 public:
  Link(Derivable g) : f(g) {}

 private:
};

template <class T>
class Sigmoid {
 public:
 private:
  st::queue<xt::xarray<T>>
};

}  // namespace links
}  // namespace xnn

#endif  // __XNN_LINKS_HPP__
