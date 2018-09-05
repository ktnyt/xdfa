#ifndef __XDFA_FUNCTION_HPP__
#define __XDFA_FUNCTION_HPP__

#include "xdfa/types.hpp"

namespace xdfa {

class Function {
 public:
  virtual Arrayf forward(Arrayf) = 0;
  virtual Arrayf backward(Arrayf) = 0;
};

}  // namespace xdfa

#endif  // __XDFA_FUNCTION_HPP__
