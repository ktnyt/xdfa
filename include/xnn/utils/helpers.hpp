#ifndef __XNN_UTILS_HELPERS_HPP__
#define __XNN_UTILS_HELPERS_HPP__

#include "xnn/layer.hpp"

#include "xtensor/xarray.hpp"

#include <list>
#include <memory>
#include <vector>

namespace xnn {
namespace utils {

std::vector<std::vector<int>> to_index(xt::xarray<int> t) {
  std::vector<std::vector<int>> ret;
  for (int i = 0; i < t.size(); ++i) {
    ret.push_back({i, t[i]});
  }
  return ret;
}

template <class T>
std::list<std::shared_ptr<class Layer<T>::Impl>> to_impl() {
  return {};
}

template <class T, class Head, class... Tail>
std::list<std::shared_ptr<class Layer<T>::Impl>> to_impl(
    Head head, Tail... tail) {
  std::list<std::shared_ptr<class Layer<T>::Impl>> list =
      to_impl<T, Tail...>(tail...);
  list.push_front(std::static_pointer_cast<class Layer<T>::Impl>(head.get()));
  return list;
}

}  // namespace utils
}  // namespace xnn

#endif  // __XNN_UTILS_HELPERS_HPP__
