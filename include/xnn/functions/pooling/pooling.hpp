#ifndef __XNN_FUNCTIONS_POOLING_POOLING_HPP__
#define __XNN_FUNCTIONS_POOLING_POOLING_HPP__

#include "xnn/function.hpp"

namespace xnn {
namespace functions {
namespace pooling {

template<class T>
class Pooling2D : public Function<T> {
 public:
  Pooling2D(
      std::size_t kh,
      std::size_t kw,
      std::size_t sy,
      std::size_t sx,
      std::size_t ph,
      std::size_t pw,
      bool cover_all = true,
      bool return_indices = false)
      : kh(kh),
        kw(kw),
        sy(sy),
        sx(sx),
        ph(ph),
        pw(pw),
        cover_all(cover_all),
        return_indices(return_indices) {}

 protected:
  std::size_t kh;
  std::size_t kw;
  std::size_t sy;
  std::size_t sx;
  std::size_t ph;
  std::size_t pw;
  bool cover_all;
  bool return_indices;
};

}  // namespace pooling
}  // namespace functions
}  // namespace xnn

#endif  // __XNN_FUNCTIONS_POOLING_POOLING_HPP__
