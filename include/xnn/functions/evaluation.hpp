#ifndef __XNN_EVALUATION_HPP__
#define __XNN_EVALUATION_HPP__

#include "xnn/types.hpp"

#include "xtensor/xmath.hpp"
#include "xtensor/xsort.hpp"

namespace xnn {

Arrayf accuracy(Arrayi t, Arrayf x) {
  Arrayi y = xt::argmax(x, 1);
  Arrayf f = xt::equal(t, y);
  return xt::sum(f) / f.size();
}

}  // namespace xnn

#endif  // __XNN_EVALUATION_HPP__
