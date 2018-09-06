#ifndef __XNN_FUNCTION_HPP__
#define __XNN_FUNCTION_HPP__

#include "xnn/types.hpp"

namespace xnn {

class Function {
 public:
  virtual Arrayf forward(Arrayf) = 0;
  virtual Arrayf backward(Arrayf) = 0;
};

}  // namespace xnn

#endif  // __XNN_FUNCTION_HPP__
