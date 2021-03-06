#ifndef __XNN_DATASET_HPP__
#define __XNN_DATASET_HPP__

#include "xnn/utils/helpers.hpp"

#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xview.hpp"

#include <algorithm>
#include <iterator>
#include <random>
#include <tuple>
#include <vector>

#include <iostream>

namespace xnn {

template <class T1, class T2>
class Dataset {
 public:
  Dataset(xt::xarray<T1>&& x, xt::xarray<T2>&& t) : x(x), t(t) {}

  Dataset(std::pair<xt::xarray<T1>, xt::xarray<T2>>&& data)
      : x(data.first), t(data.second) {}

  template <class S>
  Dataset(xt::xarray<T1>&& x, xt::xarray<T2>&& t, S seed)
      : x(x), t(t), rng1(seed), rng2(seed) {}

  template <class S>
  Dataset(std::pair<xt::xarray<T1>, xt::xarray<T2>>&& data, S seed)
      : x(data.first), t(data.second), rng1(seed), rng2(seed) {}

  Dataset(const Dataset&) = default;
  Dataset(Dataset&&) = default;
  Dataset& operator=(const Dataset&) = default;
  Dataset& operator=(Dataset&&) = default;

  void shuffle() {
    xt::random::shuffle(x, rng1);
    xt::random::shuffle(t, rng2);
  }

  std::size_t size() const {
    return x.shape()[0];
  }

  std::size_t leading() const {
    return x.shape()[1];
  }

  xt::xarray<T1>& x_data() { return x; }
  xt::xarray<T2>& t_data() { return t; }

  void for_each(
      std::size_t batchsize,
      std::function<void(xt::xarray<T1>&, xt::xarray<T2>&)> f) {
    for (std::size_t i = 0; i < utils::len(x); i += batchsize) {
      xt::xarray<T1> x_batch = xt::view(x, xt::range(i, i + batchsize));
      xt::xarray<T2> t_batch = xt::view(t, xt::range(i, i + batchsize));
      f(x_batch, t_batch);
    }
  }

 private:
  xt::xarray<T1> x;
  xt::xarray<T2> t;

  std::mt19937 rng1;
  std::mt19937 rng2;
};

}  // namespace xnn

#endif  // __XNN_DATASET_HPP__
