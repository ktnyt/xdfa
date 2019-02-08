#ifndef __XNN_LAYERS_MANIPULATION_FLATTEN_HPP__
#define __XNN_LAYERS_MANIPULATION_FLATTEN_HPP__

#include "xnn/functions/manipulation/flatten.hpp"
#include "xnn/layer.hpp"

#include <queue>

namespace xnn {
namespace layers {
namespace manipulation {

template <class T>
class Flatten final : public Layer<T> {
  class Impl final : public Layer<T>::Impl {
   public:
    xt::xarray<T> forward(const xt::xarray<T>& x) {
      queue.emplace(x.shape().begin(), x.shape().end());
      return functions::manipulation::flatten(x);
    }

    xt::xarray<T> backward(const xt::xarray<T>& dy) {
      std::vector<std::size_t> shape = queue.front();
      queue.pop();
      return functions::manipulation::unflatten(dy, shape);
    }

   private:
    std::queue<std::vector<std::size_t>> queue;
  };

 public:
  Flatten() : Layer<T>(std::make_shared<Impl>()) {}
};

}  // namespace manipulation
}  // namespace layers
}  // namespace xnn

#endif  // __XNN_LAYERS_MANIPULATION_FLATTEN_HPP__
