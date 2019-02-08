#ifndef __XNN_DATASETS_CIFAR10_HPP__
#define __XNN_DATASETS_CIFAR10_HPP__

#include "xnn/dataset.hpp"

#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xview.hpp"

#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>

#define CIFAR10_BATCH_SIZE 10000

namespace xnn {
namespace datasets {
namespace cifar10 {
namespace internal {

template <class T1, class T2>
std::pair<xt::xarray<T1>, xt::xarray<T2>> load_batch(
    std::string path, bool flatten) {
  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file.is_open()) {
    std::runtime_error("failed to open image file: " + path);
  }

  xt::xarray<T1> x(std::vector<std::size_t>({CIFAR10_BATCH_SIZE, 3, 32, 32}));
  xt::xarray<T2> t(std::vector<std::size_t>({CIFAR10_BATCH_SIZE}));

  unsigned char b;
  for (std::size_t i = 0; i < CIFAR10_BATCH_SIZE; ++i) {
    file.get(reinterpret_cast<char&>(b));
    t(i) = static_cast<T2>(b);
    for (std::size_t j = 0; j < 3; ++j) {
      for (std::size_t r = 0; r < 32; ++r) {
        for (std::size_t c = 0; c < 32; ++c) {
          file.get(reinterpret_cast<char&>(b));
          x(i, j, r, c) = static_cast<T1>(b);
        }
      }
    }
  }

  if (flatten) {
    x.reshape({CIFAR10_BATCH_SIZE, 3072});
  }

  return std::pair<xt::xarray<T1>, xt::xarray<T2>>(x, t);
}

template <class T1, class T2>
std::pair<xt::xarray<T1>, xt::xarray<T2>> load_training(
    std::string path, bool flatten) {
  xt::xarray<T1> x1;
  xt::xarray<T2> t1;
  xt::xarray<T1> x2;
  xt::xarray<T2> t2;
  xt::xarray<T1> x3;
  xt::xarray<T2> t3;
  xt::xarray<T1> x4;
  xt::xarray<T2> t4;
  xt::xarray<T1> x5;
  xt::xarray<T2> t5;

  std::tie(x1, t1) = load_batch<T1, T2>(path + "/data_batch_1.bin", flatten);
  std::tie(x2, t2) = load_batch<T1, T2>(path + "/data_batch_2.bin", flatten);
  std::tie(x3, t3) = load_batch<T1, T2>(path + "/data_batch_3.bin", flatten);
  std::tie(x4, t4) = load_batch<T1, T2>(path + "/data_batch_4.bin", flatten);
  std::tie(x5, t5) = load_batch<T1, T2>(path + "/data_batch_5.bin", flatten);

  xt::xarray<T1> x = xt::stack(std::make_tuple(x1, x2, x3, x4, x5));
  xt::xarray<T2> t = xt::stack(std::make_tuple(t1, t2, t3, t4, t5));

  x.reshape({CIFAR10_BATCH_SIZE * 5, 3, 32, 32});
  t.reshape({CIFAR10_BATCH_SIZE * 5});

  return std::pair<xt::xarray<T1>, xt::xarray<T2>>(x, t);
}

}  // namespace internal

template <class T1, class T2>
class Training : public Dataset<T1, T2> {
 public:
  Training(std::string path, bool flatten)
      : Dataset<T1, T2>(internal::load_training<T1, T2>(path, flatten)) {}

  template <class S>
  Training(std::string path, bool flatten, S seed)
      : Dataset<T1, T2>(internal::load_training<T1, T2>(path, flatten), seed) {}
};

}  // namespace cifar10
}  // namespace datasets
}  // namespace xnn

#endif  // __XNN_DATASETS_CIFAR10_HPP__
