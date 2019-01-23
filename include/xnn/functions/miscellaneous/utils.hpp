#ifndef __XNN_FUNCTIONS_MISCELLANEOUS_UTILS_HPP__
#define __XNN_FUNCTIONS_MISCELLANEOUS_UTILS_HPP__

#include "xnn/function.hpp"

#include <list>

namespace xnn {
namespace functions {
namespace miscellaneous {

template <class T>
std::list<std::shared_ptr<typename Function<T>::Impl>> to_impl() {
  return {};
}

template <class T, class Head, class... Tail>
std::list<std::shared_ptr<typename Function<T>::Impl>> to_impl(Head head,
                                                               Tail... tail) {
  std::list<std::shared_ptr<typename Function<T>::Impl>> list =
      to_impl<T, Tail...>(tail...);
  list.push_front(
      std::static_pointer_cast<typename Function<T>::Impl>(head.get()));
  return list;
}

} // namespace miscellaneous
} // namespace functions
} // namespace xnn

#endif // __XNN_FUNCTIONS_MISCELLANEOUS_UTILS_HPP__
