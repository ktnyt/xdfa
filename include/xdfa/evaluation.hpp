#ifndef __XDFA_EVALUATION_HPP__
#define __XDFA_EVALUATION_HPP__

#include "xdfa/types.hpp"

#include "xtensor/xmath.hpp"
#include "xtensor/xsort.hpp"

namespace xdfa {

Arrayf accuracy(Arrayi t, Arrayf x) {
  Arrayi y = xt::argmax(x, 1);
  Arrayf f = xt::equal(t, y);
  return xt::sum(f) / f.size();
}

}  // namespace xdfa

#endif  // __XDFA_EVALUATION_HPP__
